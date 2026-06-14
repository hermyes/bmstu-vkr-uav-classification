from __future__ import annotations

from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import soundfile as sf


def validate_audio(path: str | Path) -> bool:
    path = Path(path)
    try:
        info = sf.info(str(path))
    except Exception:
        return False
    return info.frames > 0 and info.samplerate > 0


def get_audio_duration(path: str | Path) -> float:
    info = sf.info(str(path))
    return float(info.duration)


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    if audio.ndim != 2:
        raise ValueError(f"Expected 1D or 2D audio array, got shape {audio.shape}")
    # SoundFile returns shape (n_samples, n_channels) when always_2d=True.
    return np.mean(audio, axis=1)


def _normalize(audio: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak < eps:
        return audio
    return audio / peak


def load_audio(
    path: str | Path,
    target_sr: int | None = 22050,
    mono: bool = True,
    normalize: bool = True,
) -> Tuple[np.ndarray, int]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {path}")

    audio, sr = sf.read(str(path), dtype="float32", always_2d=not mono)
    if mono:
        audio = _to_mono(audio)
    else:
        if audio.ndim == 2:
            audio = audio.T

    if target_sr is not None and int(sr) != int(target_sr):
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = int(target_sr)

    if normalize:
        audio = _normalize(audio)

    return np.asarray(audio, dtype=np.float32), int(sr)
