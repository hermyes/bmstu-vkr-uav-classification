from __future__ import annotations

from typing import List, Tuple

import librosa
import numpy as np


def _safe_fft_size(audio_len: int) -> int:
    if audio_len <= 512:
        return max(64, audio_len)
    if audio_len <= 1024:
        return 512
    if audio_len <= 2048:
        return 1024
    return 2048


def extract_mfcc_features(
    audio: np.ndarray,
    sr: int,
    n_mfcc: int = 20,
    use_delta: bool = True,
    use_delta_delta: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    n_fft = _safe_fft_size(len(audio))
    hop_length = max(1, n_fft // 4)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    vectors: List[float] = []
    names: List[str] = []

    for i in range(n_mfcc):
        vectors.append(float(np.mean(mfcc[i])))
        names.append(f"mfcc_{i+1}_mean")
        vectors.append(float(np.std(mfcc[i])))
        names.append(f"mfcc_{i+1}_std")

    if use_delta:
        delta = librosa.feature.delta(mfcc)
        for i in range(n_mfcc):
            vectors.append(float(np.mean(delta[i])))
            names.append(f"mfcc_delta_{i+1}_mean")
            vectors.append(float(np.std(delta[i])))
            names.append(f"mfcc_delta_{i+1}_std")

    if use_delta_delta:
        delta2 = librosa.feature.delta(mfcc, order=2)
        for i in range(n_mfcc):
            vectors.append(float(np.mean(delta2[i])))
            names.append(f"mfcc_delta2_{i+1}_mean")
            vectors.append(float(np.std(delta2[i])))
            names.append(f"mfcc_delta2_{i+1}_std")

    return np.asarray(vectors, dtype=np.float32), names
