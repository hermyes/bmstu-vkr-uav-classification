#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_utils import ensure_dir, load_yaml, save_json
from src.data.feature_dataset import build_feature_matrix_from_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract handcrafted features from a manifest split")
    parser.add_argument("--config", type=str, default="configs/svm_baseline.yaml", help="Path to config")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset root")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Split name")
    parser.add_argument("--out-dir", type=str, default="reports/features", help="Directory for feature files")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    dataset_cfg = dict(cfg.get("dataset", {}))
    preprocessing_cfg = dict(cfg.get("preprocessing", {}))
    features_cfg = dict(cfg.get("features", {}))

    root_dir = Path(args.dataset or dataset_cfg.get("root_dir", "train_sounds/dataset_out"))
    manifest_key = f"{args.split}_manifest"
    manifest_rel = str(dataset_cfg.get(manifest_key, f"{args.split}/manifest.json"))
    manifest_path = root_dir / manifest_rel

    x, y, meta, feature_names = build_feature_matrix_from_manifest(
        manifest_path=manifest_path,
        preprocessing_cfg=preprocessing_cfg,
        features_cfg=features_cfg,
        limit=args.limit,
    )

    out_dir = ensure_dir(args.out_dir)
    npz_path = out_dir / f"{args.split}_features.npz"
    meta_path = out_dir / f"{args.split}_meta.json"
    names_path = out_dir / "feature_names.json"

    np.savez_compressed(npz_path, X=x, y=y)
    save_json(meta_path, meta)
    save_json(names_path, feature_names)

    print(f"Saved feature matrix: {npz_path}")
    print(f"Saved metadata: {meta_path}")
    print(f"Saved feature names: {names_path}")


if __name__ == "__main__":
    main()
