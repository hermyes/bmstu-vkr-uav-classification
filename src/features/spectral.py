from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import librosa
import numpy as np
from scipy.signal import find_peaks


def _safe_fft_size(audio_len: int) -> int:
    if audio_len <= 512:
        return max(64, audio_len)
    if audio_len <= 1024:
        return 512
    if audio_len <= 2048:
        return 1024
    return 2048


def _append_mean_std(values: np.ndarray, base_name: str, out_values: List[float], out_names: List[str]) -> None:
    out_values.append(float(np.mean(values)))
    out_names.append(f"{base_name}_mean")
    out_values.append(float(np.std(values)))
    out_names.append(f"{base_name}_std")


def extract_spectral_summary_features(
    audio: np.ndarray,
    sr: int,
    use_centroid: bool = True,
    use_bandwidth: bool = True,
    use_rolloff: bool = True,
    use_zcr: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    n_fft = _safe_fft_size(len(audio))
    hop_length = max(1, n_fft // 4)
    values: List[float] = []
    names: List[str] = []

    if use_centroid:
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        _append_mean_std(centroid, "spectral_centroid", values, names)

    if use_bandwidth:
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        _append_mean_std(bandwidth, "spectral_bandwidth", values, names)

    if use_rolloff:
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        _append_mean_std(rolloff, "spectral_rolloff", values, names)

    if use_zcr:
        zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=n_fft, hop_length=hop_length)[0]
        _append_mean_std(zcr, "zero_crossing_rate", values, names)

    return np.asarray(values, dtype=np.float32), names


def extract_band_energy_features(
    audio: np.ndarray,
    sr: int,
    bands: Sequence[Sequence[float]],
) -> Tuple[np.ndarray, List[str]]:
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / sr)

    values: List[float] = []
    names: List[str] = []
    for low, high in bands:
        mask = (freqs >= float(low)) & (freqs < float(high))
        energy = float(np.mean(np.square(spectrum[mask]))) if np.any(mask) else 0.0
        values.append(energy)
        names.append(f"band_energy_{int(low)}_{int(high)}")

    return np.asarray(values, dtype=np.float32), names


def extract_spectral_peaks_features(
    audio: np.ndarray,
    sr: int,
    n_peaks: int = 5,
    min_peak_height_ratio: float = 0.2,
) -> Tuple[np.ndarray, List[str]]:
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / sr)
    values: List[float] = []
    names: List[str] = []

    if spectrum.size == 0:
        spectrum = np.zeros(1, dtype=np.float32)
        freqs = np.zeros(1, dtype=np.float32)

    height = float(np.max(spectrum)) * float(min_peak_height_ratio)
    peaks, props = find_peaks(spectrum, height=height)
    peak_heights = props.get("peak_heights", np.array([], dtype=np.float32))

    if peaks.size > 0:
        order = np.argsort(peak_heights)[::-1]
        peaks = peaks[order]
        peak_heights = peak_heights[order]

    for i in range(n_peaks):
        if i < len(peaks):
            freq_val = float(freqs[peaks[i]])
            amp_val = float(peak_heights[i])
        else:
            freq_val = 0.0
            amp_val = 0.0

        values.append(freq_val)
        names.append(f"spectral_peak_{i+1}_freq")
        values.append(amp_val)
        names.append(f"spectral_peak_{i+1}_amp")

    return np.asarray(values, dtype=np.float32), names
