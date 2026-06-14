from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib

from src.config_utils import load_json, save_json


def save_model_artifacts(
    model,
    scaler,
    label_map: Dict[str, int],
    output_dir: str | Path,
    model_name: str = "svm_baseline.pkl",
    scaler_name: str = "scaler.pkl",
    label_map_name: str = "label_map.json",
) -> Tuple[Path, Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / model_name
    scaler_path = output_dir / scaler_name
    label_map_path = output_dir / label_map_name

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    save_json(label_map_path, label_map)

    return model_path, scaler_path, label_map_path


def load_model_artifacts(
    model_path: str | Path,
    scaler_path: str | Path,
    label_map_path: str | Path,
):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_map = load_json(label_map_path)
    return model, scaler, label_map
