#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import soundfile as sf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.audio_loader import load_audio


CHUNK_RE = re.compile(r"^(?P<base>.+)_chunk_(?P<idx>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build 3-second dataset from existing train_sounds/dataset_out "
            "that currently contains 1-second chunks."
        )
    )
    parser.add_argument(
        "--input-dataset",
        type=str,
        default="train_sounds/dataset_out",
        help="Path to source dataset root with train/val/test manifests.",
    )
    parser.add_argument(
        "--output-dataset",
        type=str,
        default="train_sounds/dataset_out_3sec",
        help="Path to output 3-second dataset root.",
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=22050,
        help="Target sample rate for output audio.",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=3.0,
        help="Output sample duration in seconds.",
    )
    parser.add_argument(
        "--source-chunk-sec",
        type=float,
        default=1.0,
        help="Expected duration of input chunks in seconds.",
    )
    parser.add_argument(
        "--stride-chunks",
        type=int,
        default=3,
        help="Step in chunks while building windows (3 = non-overlapping 3 sec windows).",
    )
    parser.add_argument(
        "--pad-last",
        action="store_true",
        help="Pad final incomplete window with zeros instead of dropping it.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output directory if it already exists.",
    )
    return parser.parse_args()


def _load_manifest(path: Path) -> List[Dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Manifest is not a JSON list: {path}")
    return data


def _parse_chunk_idx(item: Dict[str, object]) -> int:
    orig_path = str(item.get("orig_path", ""))
    stem = Path(orig_path).stem if orig_path else ""
    match = CHUNK_RE.match(stem)
    if match:
        return int(match.group("idx"))

    file_name = str(item.get("file", ""))
    fallback_stem = Path(file_name).stem
    match = CHUNK_RE.match(fallback_stem)
    if match:
        return int(match.group("idx"))

    # Fallback: preserve deterministic order even if index is unknown.
    item_id = int(item.get("id", -1))
    return max(item_id, 0)


def _ensure_exact_len(audio: np.ndarray, n_samples: int) -> np.ndarray:
    out = np.asarray(audio, dtype=np.float32)
    if len(out) == n_samples:
        return out
    if len(out) > n_samples:
        return out[:n_samples]
    padded = np.zeros(n_samples, dtype=np.float32)
    padded[: len(out)] = out
    return padded


def _iter_source_windows(
    records: List[Dict[str, object]],
    window_chunks: int,
    stride_chunks: int,
    pad_last: bool,
) -> Iterable[List[Dict[str, object]]]:
    n = len(records)
    i = 0
    while i < n:
        end = i + window_chunks
        if end <= n:
            yield records[i:end]
        elif pad_last:
            yield records[i:n]
        else:
            break
        i += stride_chunks


def _copy_label_map(input_root: Path, output_root: Path) -> None:
    src = input_root / "label_map.json"
    dst = output_root / "label_map.json"
    if not src.exists():
        raise FileNotFoundError(f"label_map.json not found: {src}")
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> None:
    args = parse_args()

    input_root = Path(args.input_dataset)
    output_root = Path(args.output_dataset)
    if not input_root.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_root}")

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output_root}. Use --overwrite to rebuild."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    window_chunks = int(round(float(args.window_sec) / float(args.source_chunk_sec)))
    if window_chunks <= 0:
        raise ValueError("window_chunks must be positive")
    if args.stride_chunks <= 0:
        raise ValueError("--stride-chunks must be >= 1")

    chunk_samples = int(round(float(args.source_chunk_sec) * int(args.target_sr)))
    out_samples = window_chunks * chunk_samples
    out_duration = float(out_samples) / float(args.target_sr)

    split_stats: Dict[str, Counter] = {split: Counter() for split in ("train", "val", "test")}
    split_counts: Dict[str, int] = {split: 0 for split in ("train", "val", "test")}

    for split in ("train", "val", "test"):
        in_manifest_path = input_root / split / "manifest.json"
        in_audio_root = input_root / split
        if not in_manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {in_manifest_path}")

        items = _load_manifest(in_manifest_path)
        groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for item in items:
            source_id = str(item.get("source_id", ""))
            groups[source_id].append(item)

        out_split_root = output_root / split
        out_audio_root = out_split_root / "audio"
        out_audio_root.mkdir(parents=True, exist_ok=True)
        out_manifest: List[Dict[str, object]] = []

        sample_id = 0
        for source_id, source_items in tqdm(
            sorted(groups.items(), key=lambda kv: kv[0]),
            desc=f"Building {split}",
            unit="source",
        ):
            sorted_items = sorted(source_items, key=_parse_chunk_idx)

            # Preload source chunks once.
            loaded_chunks: List[Tuple[Dict[str, object], np.ndarray]] = []
            for item in sorted_items:
                rel_file = str(item.get("file", ""))
                abs_path = in_audio_root / rel_file
                audio, _ = load_audio(
                    path=abs_path,
                    target_sr=int(args.target_sr),
                    mono=True,
                    normalize=False,
                )
                audio = _ensure_exact_len(audio, chunk_samples)
                loaded_chunks.append((item, audio))

            for window in _iter_source_windows(
                [x[0] for x in loaded_chunks],
                window_chunks=window_chunks,
                stride_chunks=int(args.stride_chunks),
                pad_last=bool(args.pad_last),
            ):
                window_audios: List[np.ndarray] = []
                window_items: List[Dict[str, object]] = []
                # Map for O(1) access by item id.
                by_id = {int(itm.get("id", idx)): aud for idx, (itm, aud) in enumerate(loaded_chunks)}
                for itm in window:
                    key = int(itm.get("id", -1))
                    if key in by_id:
                        window_audios.append(by_id[key])
                        window_items.append(itm)

                if not window_audios:
                    continue

                merged = np.concatenate(window_audios, axis=0).astype(np.float32)
                merged = _ensure_exact_len(merged, out_samples)

                out_name = f"{sample_id:06d}.wav"
                out_path = out_audio_root / out_name
                sf.write(str(out_path), merged, samplerate=int(args.target_sr), subtype="PCM_16")

                first = window_items[0]
                label = str(first.get("label", "unknown"))
                label_id = int(first.get("label_id", -1))
                orig_paths = [str(it.get("orig_path", "")) for it in window_items]
                start_idx = _parse_chunk_idx(window_items[0])
                end_idx = _parse_chunk_idx(window_items[-1])

                out_manifest.append(
                    {
                        "id": sample_id,
                        "file": f"audio/{out_name}",
                        "label": label,
                        "label_id": label_id,
                        "source_id": source_id,
                        "duration_sec": out_duration,
                        "sample_rate": int(args.target_sr),
                        "orig_path": orig_paths[0] if orig_paths else "",
                        "orig_paths": orig_paths,
                        "source_chunk_start": int(start_idx),
                        "source_chunk_end": int(end_idx),
                        "source_window_size_chunks": int(window_chunks),
                    }
                )
                split_stats[split][label] += 1
                split_counts[split] += 1
                sample_id += 1

        (out_split_root / "manifest.json").write_text(
            json.dumps(out_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    _copy_label_map(input_root=input_root, output_root=output_root)

    print("\nDone building 3-second dataset.")
    print(f"Input : {input_root}")
    print(f"Output: {output_root}")
    print(f"target_sr={args.target_sr}, window_sec={out_duration:.3f}, stride_chunks={args.stride_chunks}, pad_last={args.pad_last}")
    for split in ("train", "val", "test"):
        print(f"\n{split.upper()} total: {split_counts[split]}")
        for label, count in sorted(split_stats[split].items(), key=lambda kv: kv[0]):
            print(f"  {label:<12} {count}")


if __name__ == "__main__":
    main()
