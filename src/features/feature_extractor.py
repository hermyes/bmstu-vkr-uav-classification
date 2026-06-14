from __future__ import annotations

from typing import Dict, List, Tuple

import librosa
import numpy as np

from src.features.mfcc import extract_mfcc_features
from src.features.spectral import (
    extract_band_energy_features,
    extract_spectral_peaks_features,
    extract_spectral_summary_features,
)


def _safe_fft_size(audio_len: int) -> int:
    if audio_len <= 512:
        return max(64, audio_len)
    if audio_len <= 1024:
        return 512
    if audio_len <= 2048:
        return 1024
    return 2048


def _extract_rms_features(audio: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    n_fft = _safe_fft_size(len(audio))
    hop_length = max(1, n_fft // 4)
    rms = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop_length)[0]
    features = np.asarray([float(np.mean(rms)), float(np.std(rms))], dtype=np.float32)
    names = ["rms_mean", "rms_std"]
    return features, names


def extract_features_with_names(
    audio: np.ndarray,
    sr: int,
    config: Dict[str, object],
) -> Tuple[np.ndarray, List[str]]:
    features: List[np.ndarray] = []
    names: List[str] = []

    mfcc_cfg = dict(config.get("mfcc", {}))
    mfcc_values, mfcc_names = extract_mfcc_features(
        audio=audio,
        sr=sr,
        n_mfcc=int(mfcc_cfg.get("n_mfcc", 20)),
        use_delta=bool(mfcc_cfg.get("use_delta", True)),
        use_delta_delta=bool(mfcc_cfg.get("use_delta_delta", True)),
    )
    features.append(mfcc_values)
    names.extend(mfcc_names)

    rms_cfg = dict(config.get("rms", {}))
    if bool(rms_cfg.get("enabled", True)):
        rms_values, rms_names = _extract_rms_features(audio)
        features.append(rms_values)
        names.extend(rms_names)

    spectral_cfg = dict(config.get("spectral", {}))
    spectral_values, spectral_names = extract_spectral_summary_features(
        audio=audio,
        sr=sr,
        use_centroid=bool(spectral_cfg.get("use_centroid", True)),
        use_bandwidth=bool(spectral_cfg.get("use_bandwidth", True)),
        use_rolloff=bool(spectral_cfg.get("use_rolloff", True)),
        use_zcr=bool(spectral_cfg.get("use_zcr", True)),
    )
    if spectral_values.size > 0:
        features.append(spectral_values)
        names.extend(spectral_names)

    band_cfg = dict(config.get("band_energy", {}))
    if bool(band_cfg.get("enabled", True)):
        bands = band_cfg.get("bands", [[80, 300], [300, 1000], [1000, 3000], [3000, 8000]])
        band_values, band_names = extract_band_energy_features(audio=audio, sr=sr, bands=bands)
        features.append(band_values)
        names.extend(band_names)

    peaks_cfg = dict(config.get("spectral_peaks", {}))
    if bool(peaks_cfg.get("enabled", True)):
        peak_values, peak_names = extract_spectral_peaks_features(
            audio=audio,
            sr=sr,
            n_peaks=int(peaks_cfg.get("n_peaks", 5)),
            min_peak_height_ratio=float(peaks_cfg.get("min_peak_height_ratio", 0.2)),
        )
        features.append(peak_values)
        names.extend(peak_names)

    if not features:
        raise ValueError("No features enabled. Check features config.")

    vector = np.concatenate(features).astype(np.float32)
    return vector, names


def extract_features(audio: np.ndarray, sr: int, config: Dict[str, object]) -> np.ndarray:
    vector, _ = extract_features_with_names(audio, sr, config)
    return vector
