from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

from src.data.audio_loader import load_audio
from src.data.manifest import load_manifest, resolve_audio_paths
from src.evaluation.metrics import compute_classification_metrics
from src.signal.heuristic_detector import analyze_windows, guess_class_from_window_stats
from src.signal.preprocessing import preprocess_audio
from src.signal.segmentation import split_into_windows


def _sample_manifest(
    manifest: List[dict],
    limit: int | None,
    random_seed: int = 42,
) -> List[dict]:
    if limit is None or limit >= len(manifest):
        return manifest
    rng = random.Random(random_seed)
    idx = sorted(rng.sample(range(len(manifest)), k=limit))
    return [manifest[i] for i in idx]


def evaluate_heuristic_type_guess(
    manifest_path: str | Path,
    label_map: Dict[str, int],
    preprocessing_cfg: Dict[str, Any],
    segmentation_cfg: Dict[str, Any] | None = None,
    heuristic_cfg: Dict[str, Any] | None = None,
    show_progress: bool = True,
    limit: int | None = None,
    random_seed: int = 42,
) -> Dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    manifest = resolve_audio_paths(manifest, manifest_path.parent)
    manifest = _sample_manifest(manifest, limit=limit, random_seed=random_seed)

    segmentation_cfg = dict(segmentation_cfg or {})
    heuristic_cfg = dict(heuristic_cfg or {})
    type_guess_cfg = dict(heuristic_cfg.get("type_guess", {}))

    target_sr = int(preprocessing_cfg.get("target_sample_rate", 22050))
    mono = bool(preprocessing_cfg.get("mono", True))
    normalize = bool(preprocessing_cfg.get("normalize", True))
    window_sec = float(segmentation_cfg.get("window_sec", 1.0))
    hop_sec = float(segmentation_cfg.get("hop_sec", 0.5))
    pad_end = bool(segmentation_cfg.get("pad_end", False))

    label_map = {str(k): int(v) for k, v in label_map.items()}
    sorted_labels = sorted(label_map.items(), key=lambda x: x[1])
    label_names = [name for name, _ in sorted_labels]
    label_ids = [idx for _, idx in sorted_labels]
    valid_labels = set(label_map.keys())

    y_true: List[int] = []
    y_pred: List[int] = []

    selected_by_threshold = 0
    fallback_selected = 0
    reason_counter: Counter[str] = Counter()

    iterator = manifest
    if show_progress:
        iterator = tqdm(manifest, desc=f"HeuristicType {manifest_path.parent.name}", unit="file")

    for item in iterator:
        audio_path = Path(str(item["resolved_path"]))
        audio, sr = load_audio(
            path=audio_path,
            target_sr=target_sr,
            mono=mono,
            normalize=normalize,
        )
        processed = preprocess_audio(audio, sr=sr, cfg=preprocessing_cfg)
        windows, bounds = split_into_windows(
            audio=processed,
            sr=sr,
            window_sec=window_sec,
            hop_sec=hop_sec,
            pad_end=pad_end,
        )
        heur = analyze_windows(windows=windows, sr=sr, bounds=bounds, cfg=heuristic_cfg)
        idx = heur.get("index")
        if idx is None:
            pred_label_name = "background"
            reason_counter["empty_audio_fallback"] += 1
            fallback_selected += 1
        else:
            if bool(heur.get("selected", False)):
                selected_by_threshold += 1
            else:
                fallback_selected += 1

            selected_detail = {}
            for detail in heur.get("window_details", []):
                if int(detail.get("index", -1)) == int(idx):
                    selected_detail = detail
                    break

            guess = guess_class_from_window_stats(selected_detail, cfg=type_guess_cfg)
            pred_label_name = str(guess["label"])
            reason_counter[str(guess.get("reason", "unknown"))] += 1

        if pred_label_name not in valid_labels:
            pred_label_name = "background"
            reason_counter["invalid_label_fallback"] += 1

        y_true.append(int(item["label_id"]))
        y_pred.append(int(label_map[pred_label_name]))

    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        label_ids=label_ids,
        label_names=label_names,
    )
    total = len(y_true)
    metrics["n_tested"] = int(total)
    metrics["selected_by_threshold"] = int(selected_by_threshold)
    metrics["selected_by_fallback"] = int(fallback_selected)
    metrics["selected_by_threshold_ratio"] = float(selected_by_threshold / total) if total > 0 else 0.0
    metrics["reason_counts"] = dict(reason_counter)

    return metrics
