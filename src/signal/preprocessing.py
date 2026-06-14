from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.signal import butter, sosfiltfilt


def normalize_audio(audio: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak < eps:
        return audio
    return audio / peak


def bandpass_filter(
    audio: np.ndarray,
    sr: int,
    low_freq: float = 80.0,
    high_freq: float = 8000.0,
    order: int = 5,
) -> np.ndarray:
    if audio.size == 0:
        return audio

    nyquist = sr / 2.0
    low = max(1.0, float(low_freq))
    high = min(float(high_freq), nyquist - 1.0)
    if low >= high:
        return audio

    sos = butter(order, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    return sosfiltfilt(sos, audio).astype(np.float32)


def compute_rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


def compute_energy(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.mean(np.square(audio)))


def preprocess_audio(audio: np.ndarray, sr: int, cfg: Dict[str, object]) -> np.ndarray:
    out = np.asarray(audio, dtype=np.float32)
    if bool(cfg.get("apply_bandpass", False)):
        out = bandpass_filter(
            out,
            sr=sr,
            low_freq=float(cfg.get("low_freq", 80.0)),
            high_freq=float(cfg.get("high_freq", 8000.0)),
            order=int(cfg.get("filter_order", 5)),
        )
    if bool(cfg.get("normalize", True)):
        out = normalize_audio(out)
    return out.astype(np.float32)
