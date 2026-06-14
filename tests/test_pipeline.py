from __future__ import annotations

import numpy as np

from src.app.pipeline import UAVClassificationPipeline


class DummyScaler:
    def transform(self, x):
        return x


class DummyModel:
    def predict(self, x):
        return np.ones((x.shape[0],), dtype=np.int64)

    def predict_proba(self, x):
        probs = np.zeros((x.shape[0], 4), dtype=np.float32)
        probs[:, 1] = 0.9
        probs[:, 0] = 0.1
        return probs


def test_pipeline_infer_audio_array() -> None:
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    audio = (0.7 * np.sin(2 * np.pi * 350 * t)).astype(np.float32)

    label_map = {
        "background": 0,
        "drone": 1,
        "helicopter": 2,
        "airplane": 3,
    }
    cfg = {
        "preprocessing": {
            "target_sample_rate": 22050,
            "mono": True,
            "normalize": True,
            "apply_bandpass": True,
            "low_freq": 80,
            "high_freq": 8000,
            "filter_order": 4,
        },
        "segmentation": {"window_sec": 1.0, "hop_sec": 0.5, "pad_end": False},
        "heuristic": {
            "alpha": 1.5,
            "min_peaks": 3,
            "band_low_freq": 80,
            "band_high_freq": 8000,
            "min_peak_height_ratio": 0.2,
        },
        "features": {
            "mfcc": {"n_mfcc": 13, "use_delta": True, "use_delta_delta": True},
            "rms": {"enabled": True},
            "spectral": {"use_centroid": True, "use_bandwidth": True, "use_rolloff": True, "use_zcr": True},
            "band_energy": {"enabled": True, "bands": [[80, 300], [300, 1000], [1000, 3000], [3000, 8000]]},
            "spectral_peaks": {"enabled": True, "n_peaks": 3, "min_peak_height_ratio": 0.2},
        },
        "decision": {"confidence_threshold": 0.5},
    }

    pipeline = UAVClassificationPipeline(
        model=DummyModel(),
        scaler=DummyScaler(),
        label_map=label_map,
        config=cfg,
    )
    result = pipeline.infer_audio_array(audio=audio, sr=sr, include_diagnostics=True)
    assert result["decision"] == "accepted"
    assert result["target_type"] == "drone"
    assert "confidence" in result
    assert "diagnostics" in result
    assert "class_probabilities" in result["diagnostics"]
    assert result["diagnostics"]["class_probabilities"].get("drone", 0.0) > 0.5
    assert result["diagnostics"]["heuristic_guess_label"] in {"background", "drone", "helicopter", "airplane"}
    assert "guidance_message" in result
    assert "localization_message" in result
    assert "integration_payloads" in result
    assert result["integration_payloads"]["guidance"]["target_type"] in {"unknown", "drone"}
    assert "guidance_profile_key" in result["integration_payloads"]["guidance"]
    assert result["integration_payloads"]["localization"]["message_type"] == "localization_v1"
