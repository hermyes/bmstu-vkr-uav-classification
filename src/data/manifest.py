from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping

from src.config_utils import load_json


REQUIRED_MANIFEST_FIELDS = {
    "id",
    "file",
    "label",
    "label_id",
    "source_id",
}


def load_manifest(path: str | Path) -> List[dict]:
    path = Path(path)
    payload = load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Manifest must be a list: {path}")
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Manifest item {idx} is not an object: {path}")
        missing = REQUIRED_MANIFEST_FIELDS - set(item.keys())
        if missing:
            raise ValueError(f"Manifest item {idx} missing fields {missing}: {path}")
    return payload


def load_label_map(path: str | Path) -> Dict[str, int]:
    path = Path(path)
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Label map must be a dict: {path}")

    label_map: Dict[str, int] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            raise ValueError(f"Label map keys must be strings: {path}")
        if not isinstance(value, int):
            raise ValueError(f"Label map values must be integers: {path}")
        label_map[key] = value
    return label_map


def invert_label_map(label_map: Mapping[str, int]) -> Dict[int, str]:
    return {value: key for key, value in label_map.items()}


def resolve_audio_paths(
    manifest: Iterable[Mapping[str, object]],
    split_dir: str | Path,
) -> List[dict]:
    split_dir = Path(split_dir)
    resolved: List[dict] = []
    for item in manifest:
        entry = dict(item)
        file_field = str(entry["file"])
        file_path = Path(file_field)
        if not file_path.is_absolute():
            file_path = (split_dir / file_path).resolve()
        entry["resolved_path"] = str(file_path)
        resolved.append(entry)
    return resolved
