#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_utils import ensure_dir, load_yaml, save_json
from src.data.feature_dataset import build_feature_matrix_from_manifest
from src.data.manifest import load_label_map
from src.evaluation.metrics import compute_classification_metrics, save_metrics
from src.models.model_io import save_model_artifacts
from src.models.svm_model import predict_with_confidence, train_svm_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SVM baseline for UAV sound classification")
    parser.add_argument("--config", type=str, default="configs/svm_baseline.yaml", help="Path to YAML config")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset root")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit per split for quick debug")
    return parser.parse_args()


def _resolve_dataset_paths(cfg: Dict[str, object], dataset_override: str | None):
    dataset_cfg = dict(cfg.get("dataset", {}))
    root_dir = Path(dataset_override or dataset_cfg.get("root_dir", "train_sounds/dataset_out"))
    train_manifest = root_dir / str(dataset_cfg.get("train_manifest", "train/manifest.json"))
    val_manifest = root_dir / str(dataset_cfg.get("val_manifest", "val/manifest.json"))
    label_map_path = root_dir / str(dataset_cfg.get("label_map", "label_map.json"))
    return root_dir, train_manifest, val_manifest, label_map_path


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    root_dir, train_manifest, val_manifest, label_map_path = _resolve_dataset_paths(cfg, args.dataset)
    preprocessing_cfg = dict(cfg.get("preprocessing", {}))
    features_cfg = dict(cfg.get("features", {}))
    model_cfg = dict(cfg.get("model", {}))
    training_cfg = dict(cfg.get("training", {}))
    output_cfg = dict(cfg.get("output", {}))

    x_train, y_train, _, feature_names = build_feature_matrix_from_manifest(
        manifest_path=train_manifest,
        preprocessing_cfg=preprocessing_cfg,
        features_cfg=features_cfg,
        limit=args.limit,
    )
    x_val, y_val, _, _ = build_feature_matrix_from_manifest(
        manifest_path=val_manifest,
        preprocessing_cfg=preprocessing_cfg,
        features_cfg=features_cfg,
        limit=args.limit,
    )

    classifier, scaler, training_summary = train_svm_classifier(
        x_train=x_train,
        y_train=y_train,
        model_cfg=model_cfg,
        random_seed=int(training_cfg.get("random_seed", 42)),
        n_jobs=int(training_cfg.get("n_jobs", -1)),
    )

    x_val_scaled = scaler.transform(x_val)
    y_val_pred, y_val_conf = predict_with_confidence(classifier, x_val_scaled)

    label_map = load_label_map(label_map_path)
    labels_sorted = sorted(label_map.items(), key=lambda x: x[1])
    label_names = [name for name, _ in labels_sorted]
    label_ids = [idx for _, idx in labels_sorted]

    metrics = compute_classification_metrics(
        y_true=y_val,
        y_pred=y_val_pred,
        label_ids=label_ids,
        label_names=label_names,
    )
    metrics["mean_confidence"] = float(np.mean(y_val_conf))
    metrics["training_summary"] = training_summary
    metrics["dataset_root"] = str(root_dir)
    metrics["n_features"] = int(x_train.shape[1])
    metrics["n_train"] = int(len(y_train))
    metrics["n_val"] = int(len(y_val))

    models_dir = ensure_dir(output_cfg.get("models_dir", "models"))
    reports_dir = ensure_dir(output_cfg.get("reports_dir", "reports"))
    metrics_dir = ensure_dir(reports_dir / "metrics")

    model_path, scaler_path, label_map_out = save_model_artifacts(
        model=classifier,
        scaler=scaler,
        label_map=label_map,
        output_dir=models_dir,
    )
    save_json(models_dir / "feature_names.json", feature_names)
    save_metrics(metrics, metrics_dir / "svm_val_metrics.json")
    save_json(metrics_dir / "svm_training_summary.json", training_summary)

    print("Training completed.")
    print(f"Model: {model_path}")
    print(f"Scaler: {scaler_path}")
    print(f"Label map: {label_map_out}")
    print(f"Val macro F1: {metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
