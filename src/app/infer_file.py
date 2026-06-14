from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.app.pipeline import UAVClassificationPipeline


def infer_single_file(
    pipeline: UAVClassificationPipeline,
    audio_path: str | Path,
) -> Dict[str, Any]:
    return pipeline.infer_file(audio_path)
