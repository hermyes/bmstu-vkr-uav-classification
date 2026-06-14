from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

import numpy as np

from src.data.audio_loader import load_audio
from src.data.manifest import invert_label_map, load_label_map, load_manifest, resolve_audio_paths


AudioTransform = Callable[[np.ndarray, int], np.ndarray]


@dataclass
class DatasetItem:
    path: str
    label: int
    label_name: str
    audio: np.ndarray
    sample_rate: int
    source_id: str


class ManifestAudioDataset:
    def __init__(
        self,
        manifest_path: str | Path,
        label_map_path: str | Path,
        target_sr: int = 22050,
        mono: bool = True,
        normalize: bool = True,
        audio_transform: Optional[AudioTransform] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.label_map_path = Path(label_map_path)
        self.target_sr = target_sr
        self.mono = mono
        self.normalize = normalize
        self.audio_transform = audio_transform

        raw_manifest = load_manifest(self.manifest_path)
        self.manifest = resolve_audio_paths(raw_manifest, self.manifest_path.parent)
        self.label_map: Dict[str, int] = load_label_map(self.label_map_path)
        self.id_to_label: Dict[int, str] = invert_label_map(self.label_map)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> DatasetItem:
        record = self.manifest[index]
        path = Path(record["resolved_path"])
        audio, sr = load_audio(
            path=path,
            target_sr=self.target_sr,
            mono=self.mono,
            normalize=self.normalize,
        )
        if self.audio_transform is not None:
            audio = self.audio_transform(audio, sr)

        label_id = int(record["label_id"])
        label_name = str(record.get("label", self.id_to_label.get(label_id, "unknown")))

        return DatasetItem(
            path=str(path),
            label=label_id,
            label_name=label_name,
            audio=audio,
            sample_rate=sr,
            source_id=str(record.get("source_id", "")),
        )

    def iter_items(self) -> Iterator[DatasetItem]:
        for idx in range(len(self)):
            yield self[idx]


def build_dataset(
    dataset_root: str | Path,
    split: str,
    target_sr: int = 22050,
    mono: bool = True,
    normalize: bool = True,
    audio_transform: Optional[AudioTransform] = None,
) -> ManifestAudioDataset:
    dataset_root = Path(dataset_root)
    manifest_path = dataset_root / split / "manifest.json"
    label_map_path = dataset_root / "label_map.json"
    return ManifestAudioDataset(
        manifest_path=manifest_path,
        label_map_path=label_map_path,
        target_sr=target_sr,
        mono=mono,
        normalize=normalize,
        audio_transform=audio_transform,
    )


def collect_labels(dataset: ManifestAudioDataset) -> List[str]:
    labels = sorted(dataset.label_map.items(), key=lambda x: x[1])
    return [name for name, _ in labels]
