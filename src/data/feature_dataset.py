from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from src.data.audio_loader import load_audio
from src.data.manifest import load_manifest, resolve_audio_paths
from src.features.feature_extractor import extract_features, extract_features_with_names
from src.signal.preprocessing import preprocess_audio


def build_feature_matrix_from_manifest(
    manifest_path: str | Path,
    preprocessing_cfg: Dict[str, Any],
    features_cfg: Dict[str, Any],
    show_progress: bool = True,
    limit: int | None = None,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[dict], List[str]]:
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    manifest = resolve_audio_paths(manifest, manifest_path.parent)
    if limit is not None:
        if limit < len(manifest):
            rng = random.Random(random_seed)
            indices = sorted(rng.sample(range(len(manifest)), k=limit))
            manifest = [manifest[i] for i in indices]

    target_sr = int(preprocessing_cfg.get("target_sample_rate", 22050))
    mono = bool(preprocessing_cfg.get("mono", True))
    normalize = bool(preprocessing_cfg.get("normalize", True))

    x_rows: List[np.ndarray] = []
    y_rows: List[int] = []
    meta_rows: List[dict] = []
    feature_names: List[str] = []

    iterator = manifest
    if show_progress:
        iterator = tqdm(manifest, desc=f"Extracting {manifest_path.parent.name}", unit="file")

    for item in iterator:
        audio_path = Path(item["resolved_path"])
        audio, sr = load_audio(
            path=audio_path,
            target_sr=target_sr,
            mono=mono,
            normalize=normalize,
        )
        audio = preprocess_audio(audio, sr=sr, cfg=preprocessing_cfg)

        if not feature_names:
            vector, feature_names = extract_features_with_names(audio, sr=sr, config=features_cfg)
        else:
            vector = extract_features(audio, sr=sr, config=features_cfg)

        x_rows.append(vector)
        y_rows.append(int(item["label_id"]))
        meta_rows.append(
            {
                "path": str(audio_path),
                "label_id": int(item["label_id"]),
                "label": str(item.get("label", "")),
                "source_id": str(item.get("source_id", "")),
            }
        )

    if not x_rows:
        raise ValueError(f"No samples extracted from manifest: {manifest_path}")

    x = np.vstack(x_rows).astype(np.float32)
    y = np.asarray(y_rows, dtype=np.int64)
    return x, y, meta_rows, feature_names
