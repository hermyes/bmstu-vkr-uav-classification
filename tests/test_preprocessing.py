from __future__ import annotations

import numpy as np

from src.signal.preprocessing import bandpass_filter, compute_energy, compute_rms, normalize_audio
from src.signal.segmentation import split_into_windows


def test_normalize_audio() -> None:
    audio = np.array([0.0, 0.5, -2.0], dtype=np.float32)
    out = normalize_audio(audio)
    assert np.isclose(np.max(np.abs(out)), 1.0)


def test_bandpass_and_metrics() -> None:
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    audio = np.sin(2 * np.pi * 500 * t).astype(np.float32)
    filtered = bandpass_filter(audio, sr=sr, low_freq=80, high_freq=8000, order=4)
    assert filtered.shape == audio.shape
    assert compute_rms(filtered) > 0
    assert compute_energy(filtered) > 0


def test_split_into_windows() -> None:
    sr = 100
    audio = np.arange(0, 250, dtype=np.float32)
    windows, bounds = split_into_windows(audio, sr=sr, window_sec=1.0, hop_sec=0.5)
    assert len(windows) == len(bounds)
    assert len(windows) > 0
