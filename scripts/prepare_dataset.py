#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.manifest import load_label_map, load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate dataset manifests and print class stats")
    parser.add_argument("--dataset", type=str, default="train_sounds/dataset_out", help="Dataset root directory")
    return parser.parse_args()


def summarize_manifest(manifest_path: Path) -> Counter:
    manifest = load_manifest(manifest_path)
    counter = Counter([item["label"] for item in manifest])
    return counter


def main() -> None:
    args = parse_args()
    root = Path(args.dataset)
    label_map = load_label_map(root / "label_map.json")
    print("Label map:", label_map)

    for split in ["train", "val", "test"]:
        manifest_path = root / split / "manifest.json"
        counts = summarize_manifest(manifest_path)
        total = sum(counts.values())
        print(f"\n[{split}] total={total}")
        for cls, cnt in sorted(counts.items()):
            print(f"  {cls:<12} {cnt}")


if __name__ == "__main__":
    main()
