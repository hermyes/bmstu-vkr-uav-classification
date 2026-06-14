#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_utils import ensure_dir, load_yaml
from src.data.feature_dataset import build_feature_matrix_from_manifest
from src.data.manifest import load_label_map
from src.evaluation.heuristic_stats import evaluate_heuristic_type_guess
from src.evaluation.metrics import compute_classification_metrics, save_metrics
from src.evaluation.plots import save_confusion_matrix_plot
from src.models.model_io import load_model_artifacts
from src.models.svm_model import predict_with_confidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained model on test split")
    parser.add_argument("--config", type=str, default="configs/svm_baseline.yaml", help="Path to YAML config")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset root")
    parser.add_argument("--model", type=str, default="models/svm_baseline.pkl", help="Path to model artifact")
    parser.add_argument("--scaler", type=str, default="models/scaler.pkl", help="Path to scaler artifact")
    parser.add_argument("--label-map", type=str, default="models/label_map.json", help="Path to label map artifact")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for quick debug")
    parser.add_argument(
        "--skip-heuristic-type-stats",
        action="store_true",
        help="Skip heuristic-only class guessing statistics",
    )
    return parser.parse_args()


def _resolve_test_manifest(cfg: Dict[str, object], dataset_override: str | None) -> Path:
    dataset_cfg = dict(cfg.get("dataset", {}))
    root_dir = Path(dataset_override or dataset_cfg.get("root_dir", "train_sounds/dataset_out"))
    return root_dir / str(dataset_cfg.get("test_manifest", "test/manifest.json"))


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    preprocessing_cfg = dict(cfg.get("preprocessing", {}))
    features_cfg = dict(cfg.get("features", {}))
    segmentation_cfg = dict(cfg.get("segmentation", {}))
    heuristic_cfg = dict(cfg.get("heuristic", {}))
    output_cfg = dict(cfg.get("output", {}))

    test_manifest = _resolve_test_manifest(cfg, args.dataset)

    x_test, y_test, _, _ = build_feature_matrix_from_manifest(
        manifest_path=test_manifest,
        preprocessing_cfg=preprocessing_cfg,
        features_cfg=features_cfg,
        limit=args.limit,
    )

    model, scaler, label_map = load_model_artifacts(
        model_path=args.model,
        scaler_path=args.scaler,
        label_map_path=args.label_map,
    )
    label_map = {str(k): int(v) for k, v in label_map.items()}
    labels_sorted = sorted(label_map.items(), key=lambda x: x[1])
    label_names = [name for name, _ in labels_sorted]
    label_ids = [idx for _, idx in labels_sorted]

    x_test_scaled = scaler.transform(x_test)
    y_pred, confidence = predict_with_confidence(model, x_test_scaled)

    metrics = compute_classification_metrics(
        y_true=y_test,
        y_pred=y_pred,
        label_ids=label_ids,
        label_names=label_names,
    )
    metrics["mean_confidence"] = float(np.mean(confidence))
    metrics["n_test"] = int(len(y_test))

    if not args.skip_heuristic_type_stats:
        heuristic_metrics = evaluate_heuristic_type_guess(
            manifest_path=test_manifest,
            label_map=label_map,
            preprocessing_cfg=preprocessing_cfg,
            segmentation_cfg=segmentation_cfg,
            heuristic_cfg=heuristic_cfg,
            show_progress=True,
            limit=args.limit,
        )
        metrics["heuristic_type_guess"] = heuristic_metrics

    reports_dir = ensure_dir(output_cfg.get("reports_dir", "reports"))
    metrics_dir = ensure_dir(reports_dir / "metrics")
    figures_dir = ensure_dir(reports_dir / "figures")

    metrics_path = metrics_dir / "test_metrics.json"
    figure_path = figures_dir / "confusion_matrix.png"

    save_metrics(metrics, metrics_path)
    if "heuristic_type_guess" in metrics:
        save_metrics(metrics["heuristic_type_guess"], metrics_dir / "heuristic_type_guess_metrics.json")
    save_confusion_matrix_plot(
        confusion=metrics["confusion_matrix"],
        labels=label_names,
        output_path=figure_path,
        normalize=False,
        title="Test Confusion Matrix",
    )

    print("Evaluation completed.")
    print(f"Metrics: {metrics_path}")
    print(f"Figure: {figure_path}")
    print(f"Test macro F1: {metrics['f1_macro']:.4f}")
    if "heuristic_type_guess" in metrics:
        print(
            "Heuristic type guess macro F1: "
            f"{metrics['heuristic_type_guess']['f1_macro']:.4f}"
        )


if __name__ == "__main__":
    main()
