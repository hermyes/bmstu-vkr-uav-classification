from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.data.audio_loader import load_audio
from src.data.manifest import invert_label_map
from src.features.feature_extractor import extract_features
from src.models.model_io import load_model_artifacts
from src.models.svm_model import predict_with_probabilities
from src.signal.heuristic_detector import analyze_windows, guess_class_from_window_stats
from src.signal.preprocessing import preprocess_audio
from src.signal.segmentation import split_into_windows


class UAVClassificationPipeline:
    def __init__(
        self,
        model,
        scaler,
        label_map: Dict[str, int],
        config: Dict[str, Any],
    ) -> None:
        self.model = model
        self.scaler = scaler
        self.label_map = {str(k): int(v) for k, v in label_map.items()}
        self.id_to_label = invert_label_map(self.label_map)

        self.preprocessing_cfg = dict(config.get("preprocessing", {}))
        self.segmentation_cfg = dict(config.get("segmentation", {}))
        self.heuristic_cfg = dict(config.get("heuristic", {}))
        self.features_cfg = dict(config.get("features", {}))
        self.decision_cfg = dict(config.get("decision", {}))

    @classmethod
    def from_paths(
        cls,
        model_path: str | Path,
        scaler_path: str | Path,
        label_map_path: str | Path,
        config: Dict[str, Any],
    ) -> "UAVClassificationPipeline":
        model, scaler, label_map = load_model_artifacts(
            model_path=model_path,
            scaler_path=scaler_path,
            label_map_path=label_map_path,
        )
        return cls(model=model, scaler=scaler, label_map=label_map, config=config)

    def _segment_audio(self, audio: np.ndarray, sr: int):
        window_sec = float(self.segmentation_cfg.get("window_sec", 1.0))
        hop_sec = float(self.segmentation_cfg.get("hop_sec", 0.5))
        pad_end = bool(self.segmentation_cfg.get("pad_end", False))
        return split_into_windows(
            audio=audio,
            sr=sr,
            window_sec=window_sec,
            hop_sec=hop_sec,
            pad_end=pad_end,
        )

    def _build_class_probabilities(self, probs: np.ndarray) -> Dict[str, float]:
        if probs.ndim != 2 or probs.shape[0] == 0:
            return {}

        model_classes = getattr(self.model, "classes_", None)
        if model_classes is None or len(model_classes) != probs.shape[1]:
            model_classes = np.arange(probs.shape[1], dtype=np.int64)
        model_classes = [int(c) for c in model_classes]

        prob_pairs = []
        for class_idx, class_id in enumerate(model_classes):
            label_name = self.id_to_label.get(class_id, f"class_{class_id}")
            prob_pairs.append((label_name, float(probs[0, class_idx])))

        prob_pairs.sort(key=lambda item: item[1], reverse=True)
        return {name: prob for name, prob in prob_pairs}

    def _build_guidance_profile_map(self) -> Dict[str, str]:
        defaults = {
            "background": "noise_reject_profile",
            "drone": "uav_multirotor_profile",
            "helicopter": "rotorcraft_profile",
            "airplane": "fixed_wing_profile",
            "unknown": "unknown_profile",
        }
        cfg_profiles = self.decision_cfg.get("guidance_profiles", {})
        if isinstance(cfg_profiles, dict):
            for key, value in cfg_profiles.items():
                defaults[str(key)] = str(value)
        return defaults

    def _build_integration_payloads(
        self,
        accepted: bool,
        label_name: str,
        label_id: int,
        confidence: float,
        threshold: float,
        signal_level: float,
        time_start: float,
        time_end: float,
    ) -> Dict[str, Dict[str, Any]]:
        decision = "accepted" if accepted else "rejected"
        detected_type = label_name if accepted else "unknown"
        detected_label_id = label_id if accepted else -1
        guidance_profiles = self._build_guidance_profile_map()
        guidance_profile_key = guidance_profiles.get(detected_type, guidance_profiles.get("unknown", "unknown_profile"))

        guidance_payload = {
            "message_type": "guidance_v1",
            "target_detected": bool(accepted),
            "target_type": detected_type,
            "target_label_id": int(detected_label_id),
            "raw_model_type": str(label_name),
            "raw_model_label_id": int(label_id),
            "confidence": float(confidence),
            "confidence_threshold": float(threshold),
            "decision": decision,
            "decision_reason": "target_confirmed" if accepted else "confidence_below_threshold",
            "guidance_profile_key": str(guidance_profile_key),
            "time_start": float(time_start),
            "time_end": float(time_end),
        }

        localization_payload = {
            "message_type": "localization_v1",
            "target_detected": bool(accepted),
            "signal_level": float(signal_level),
            "time_start": float(time_start),
            "time_end": float(time_end),
            "channel_levels": None,
            "tdoa_ready": False,
        }

        return {
            "guidance": guidance_payload,
            "localization": localization_payload,
        }

    def infer_audio_array(
        self,
        audio: np.ndarray,
        sr: int,
        include_diagnostics: bool = False,
    ) -> Dict[str, Any]:
        processed = preprocess_audio(audio, sr=sr, cfg=self.preprocessing_cfg)
        windows, bounds = self._segment_audio(processed, sr)
        heur = analyze_windows(windows=windows, sr=sr, bounds=bounds, cfg=self.heuristic_cfg)

        index = heur.get("index")
        threshold = float(self.decision_cfg.get("confidence_threshold", 0.65))
        time_start = float(heur.get("time_start", 0.0))
        time_end = float(heur.get("time_end", 0.0))
        signal_level = float(heur.get("signal_level", 0.0))
        selected_windows = int(heur.get("selected_windows", 0))

        label_id = -1
        label_name = "unknown"
        conf = 0.0
        class_probabilities: Dict[str, float] = {}
        empty_reason: str | None = None

        if index is None:
            accepted = False
            empty_reason = "empty_audio"
        else:
            selected_window = windows[int(index)]
            feats = extract_features(selected_window, sr=sr, config=self.features_cfg)
            x = feats.reshape(1, -1)
            if self.scaler is not None:
                x = self.scaler.transform(x)
            y_pred, probs = predict_with_probabilities(self.model, x)
            label_id = int(y_pred[0])
            label_name = self.id_to_label.get(label_id, "unknown")
            conf = float(np.max(probs[0])) if probs.size else 0.0
            class_probabilities = self._build_class_probabilities(probs)
            accepted = conf >= threshold

        decision = "accepted" if accepted else "rejected"
        target_type = label_name if accepted else "unknown"
        output_label_id = label_id if accepted else -1
        selected_window_index = int(index) if index is not None else -1
        decision_reason = empty_reason or ("target_confirmed" if accepted else "confidence_below_threshold")

        integration_payloads = self._build_integration_payloads(
            accepted=accepted,
            label_name=label_name,
            label_id=label_id,
            confidence=conf,
            threshold=threshold,
            signal_level=signal_level,
            time_start=time_start,
            time_end=time_end,
        )

        result = {
            "schema_version": "2.0",
            "target_type": target_type,
            "label_id": output_label_id,
            "raw_label_id": label_id,
            "raw_label_name": label_name,
            "confidence": conf,
            "signal_level": signal_level,
            "time_start": time_start,
            "time_end": time_end,
            "selected_windows": selected_windows,
            "selected_window_index": selected_window_index,
            "decision": decision,
            "reason": decision_reason,
            "detection_summary": {
                "detected": bool(accepted),
                "decision": decision,
                "decision_reason": decision_reason,
                "object_type": target_type,
                "object_label_id": int(output_label_id),
                "raw_model_type": str(label_name),
                "raw_model_label_id": int(label_id),
                "confidence": float(conf),
                "confidence_threshold": float(threshold),
                "signal_level": float(signal_level),
                "time_start": float(time_start),
                "time_end": float(time_end),
            },
            "integration_payloads": integration_payloads,
            # Backward-compatible aliases used by existing scripts/UI.
            "guidance_message": integration_payloads["guidance"],
            "localization_message": integration_payloads["localization"],
        }

        if include_diagnostics:
            selected_window_details = {}
            heuristic_guess = {"label": "unknown", "reason": "no_selected_window", "scores": {}}
            if index is not None:
                for detail in heur.get("window_details", []):
                    if int(detail.get("index", -1)) == int(index):
                        selected_window_details = detail
                        break
                heuristic_guess = guess_class_from_window_stats(
                    selected_window_details,
                    cfg=dict(self.heuristic_cfg.get("type_guess", {})),
                )

            result["diagnostics"] = {
                "sample_rate": int(sr),
                "duration_sec": float(len(audio) / sr) if sr > 0 else 0.0,
                "num_windows": int(len(windows)),
                "window_sec": float(self.segmentation_cfg.get("window_sec", 1.0)),
                "hop_sec": float(self.segmentation_cfg.get("hop_sec", 0.5)),
                "class_probabilities": class_probabilities,
                "heuristic_guess_label": str(heuristic_guess.get("label", "unknown")),
                "heuristic_guess_reason": str(heuristic_guess.get("reason", "unknown")),
                "heuristic_guess_scores": dict(heuristic_guess.get("scores", {})),
                "heuristic": {
                    "selected": bool(heur.get("selected", False)),
                    "energy_threshold": float(heur.get("energy_threshold", 0.0)),
                    "band_threshold": float(heur.get("band_threshold", 0.0)),
                    "selected_window_details": selected_window_details,
                    "window_details": heur.get("window_details", []),
                },
            }
        return result

    def infer_file(self, audio_path: str | Path, include_diagnostics: bool = False) -> Dict[str, Any]:
        target_sr = self.preprocessing_cfg.get("target_sample_rate", 22050)
        mono = bool(self.preprocessing_cfg.get("mono", True))
        normalize = bool(self.preprocessing_cfg.get("normalize", True))

        audio, sr = load_audio(
            path=audio_path,
            target_sr=int(target_sr) if target_sr is not None else None,
            mono=mono,
            normalize=normalize,
        )
        result = self.infer_audio_array(audio=audio, sr=sr, include_diagnostics=include_diagnostics)
        result["audio_path"] = str(Path(audio_path))
        return result
