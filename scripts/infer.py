#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.app.pipeline import UAVClassificationPipeline
from src.config_utils import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for a single audio file")
    parser.add_argument("--audio", type=str, required=True, help="Path to .wav file")
    parser.add_argument("--model", type=str, default="models/svm_baseline.pkl", help="Path to model artifact")
    parser.add_argument("--scaler", type=str, default="models/scaler.pkl", help="Path to scaler artifact")
    parser.add_argument("--label-map", type=str, default="models/label_map.json", help="Path to label map")
    parser.add_argument("--config", type=str, default="configs/inference.yaml", help="Path to inference YAML config")
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Include detailed diagnostics: window stats and class probabilities",
    )
    parser.add_argument(
        "--save-guidance-json",
        type=str,
        default="",
        help="Optional path to save JSON payload for guidance subsystem",
    )
    parser.add_argument(
        "--save-localization-json",
        type=str,
        default="",
        help="Optional path to save JSON payload for localization subsystem",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    pipeline = UAVClassificationPipeline.from_paths(
        model_path=args.model,
        scaler_path=args.scaler,
        label_map_path=args.label_map,
        config=cfg,
    )
    result = pipeline.infer_file(args.audio, include_diagnostics=args.diagnostics)

    if args.save_guidance_json:
        guidance_path = Path(args.save_guidance_json)
        guidance_path.parent.mkdir(parents=True, exist_ok=True)
        guidance_payload = result.get("integration_payloads", {}).get("guidance", result.get("guidance_message", {}))
        guidance_path.write_text(json.dumps(guidance_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.save_localization_json:
        localization_path = Path(args.save_localization_json)
        localization_path.parent.mkdir(parents=True, exist_ok=True)
        localization_payload = result.get("integration_payloads", {}).get(
            "localization",
            result.get("localization_message", {}),
        )
        localization_path.write_text(json.dumps(localization_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
