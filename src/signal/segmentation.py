from __future__ import annotations

from typing import List, Tuple

import numpy as np


def window_to_time_bounds(index: int, window_sec: float, hop_sec: float) -> Tuple[float, float]:
    start = float(index) * float(hop_sec)
    end = start + float(window_sec)
    return start, end


def split_into_windows(
    audio: np.ndarray,
    sr: int,
    window_sec: float = 1.0,
    hop_sec: float = 0.5,
    pad_end: bool = False,
) -> Tuple[List[np.ndarray], List[Tuple[float, float]]]:
    if audio.ndim != 1:
        raise ValueError("split_into_windows expects 1D mono signal")

    n_samples = len(audio)
    window_size = max(1, int(round(window_sec * sr)))
    hop_size = max(1, int(round(hop_sec * sr)))

    windows: List[np.ndarray] = []
    bounds: List[Tuple[float, float]] = []

    if n_samples == 0:
        return windows, bounds

    if n_samples < window_size:
        if pad_end:
            padded = np.zeros(window_size, dtype=np.float32)
            padded[:n_samples] = audio
            windows.append(padded)
            bounds.append((0.0, window_sec))
        else:
            windows.append(audio.astype(np.float32))
            bounds.append((0.0, n_samples / sr))
        return windows, bounds

    start = 0
    while start + window_size <= n_samples:
        end = start + window_size
        windows.append(audio[start:end].astype(np.float32))
        bounds.append((start / sr, end / sr))
        start += hop_size

    if pad_end and start < n_samples:
        tail = audio[start:n_samples]
        padded = np.zeros(window_size, dtype=np.float32)
        padded[: len(tail)] = tail
        windows.append(padded)
        bounds.append((start / sr, (start + window_size) / sr))

    return windows, bounds
