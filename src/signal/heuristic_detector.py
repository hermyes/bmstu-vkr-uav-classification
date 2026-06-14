from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.signal import find_peaks

from src.signal.preprocessing import compute_energy, compute_rms


def _compute_spectrum(audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    if audio.size == 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    spectrum = np.abs(np.fft.rfft(audio)).astype(np.float64)
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / sr).astype(np.float64)
    return spectrum, freqs


def _band_energy_from_spectrum(spectrum: np.ndarray, freqs: np.ndarray, low_freq: float, high_freq: float) -> float:
    if spectrum.size == 0:
        return 0.0
    mask = (freqs >= float(low_freq)) & (freqs <= float(high_freq))
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.square(spectrum[mask])))


def _band_power_sum_from_spectrum(spectrum: np.ndarray, freqs: np.ndarray, low_freq: float, high_freq: float) -> float:
    if spectrum.size == 0:
        return 0.0
    mask = (freqs >= float(low_freq)) & (freqs < float(high_freq))
    if not np.any(mask):
        return 0.0
    return float(np.sum(np.square(spectrum[mask])))


def _spectral_peak_count(audio: np.ndarray, min_peak_height_ratio: float = 0.2) -> int:
    if audio.size == 0:
        return 0
    spectrum = np.abs(np.fft.rfft(audio))
    if spectrum.size == 0:
        return 0
    threshold = float(np.max(spectrum)) * float(min_peak_height_ratio)
    peaks, _ = find_peaks(spectrum, height=threshold)
    return int(len(peaks))


def evaluate_window(audio: np.ndarray, sr: int, cfg: Dict[str, object]) -> Dict[str, float]:
    low_freq = float(cfg.get("band_low_freq", 80.0))
    high_freq = float(cfg.get("band_high_freq", 8000.0))
    min_peak_height_ratio = float(cfg.get("min_peak_height_ratio", 0.2))
    low_split_hz = float(cfg.get("low_band_split_hz", 300.0))
    mid_split_hz = float(cfg.get("mid_band_split_hz", 1200.0))

    spectrum, freqs = _compute_spectrum(audio, sr)
    power = np.square(spectrum)
    total_power = float(np.sum(power)) + 1e-12
    dominant_freq = float(freqs[np.argmax(power)]) if power.size > 0 else 0.0
    spectral_centroid = float(np.sum(freqs * power) / total_power) if power.size > 0 else 0.0

    low_power = _band_power_sum_from_spectrum(spectrum, freqs, low_freq, low_split_hz)
    mid_power = _band_power_sum_from_spectrum(spectrum, freqs, low_split_hz, mid_split_hz)
    high_power = _band_power_sum_from_spectrum(spectrum, freqs, mid_split_hz, high_freq)

    return {
        "rms": compute_rms(audio),
        "energy": compute_energy(audio),
        "band_energy": _band_energy_from_spectrum(spectrum, freqs, low_freq, high_freq),
        "num_spectral_peaks": float(_spectral_peak_count(audio, min_peak_height_ratio)),
        "dominant_freq": dominant_freq,
        "spectral_centroid": spectral_centroid,
        "low_band_ratio": float(low_power / total_power),
        "mid_band_ratio": float(mid_power / total_power),
        "high_band_ratio": float(high_power / total_power),
    }


def _zscore(values: np.ndarray) -> np.ndarray:
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std < 1e-9:
        return np.zeros_like(values)
    return (values - mean) / std


def analyze_windows(
    windows: Sequence[np.ndarray],
    sr: int,
    bounds: Sequence[Tuple[float, float]],
    cfg: Dict[str, object],
) -> Dict[str, object]:
    if len(windows) == 0:
        return {
            "selected": False,
            "index": None,
            "time_start": 0.0,
            "time_end": 0.0,
            "signal_level": 0.0,
            "selected_windows": 0,
            "reason": "empty_input",
            "energy_threshold": 0.0,
            "band_threshold": 0.0,
            "window_details": [],
        }

    stats = [evaluate_window(w, sr, cfg) for w in windows]
    energies = np.asarray([s["energy"] for s in stats], dtype=np.float64)
    band_energies = np.asarray([s["band_energy"] for s in stats], dtype=np.float64)
    peaks = np.asarray([s["num_spectral_peaks"] for s in stats], dtype=np.float64)

    alpha = float(cfg.get("alpha", 1.5))
    min_peaks = float(cfg.get("min_peaks", 3))

    energy_thr = float(np.mean(energies) + alpha * np.std(energies))
    band_thr = float(np.mean(band_energies) + alpha * np.std(band_energies))
    mask = (energies > energy_thr) & (band_energies > band_thr) & (peaks >= min_peaks)

    score = _zscore(energies) + _zscore(band_energies) + (peaks / max(min_peaks, 1.0))
    candidate_indices = np.where(mask)[0]

    if candidate_indices.size > 0:
        chosen_idx = int(candidate_indices[np.argmax(score[candidate_indices])])
        selected = True
    else:
        chosen_idx = int(np.argmax(score))
        selected = False

    time_start, time_end = bounds[chosen_idx]
    details: List[Dict[str, float | int | bool]] = []
    for i, ((start, end), st, sc, passed) in enumerate(zip(bounds, stats, score, mask)):
        details.append(
            {
                "index": int(i),
                "time_start": float(start),
                "time_end": float(end),
                "duration_sec": float(end - start),
                "rms": float(st["rms"]),
                "energy": float(st["energy"]),
                "band_energy": float(st["band_energy"]),
                "num_spectral_peaks": int(st["num_spectral_peaks"]),
                "dominant_freq": float(st["dominant_freq"]),
                "spectral_centroid": float(st["spectral_centroid"]),
                "low_band_ratio": float(st["low_band_ratio"]),
                "mid_band_ratio": float(st["mid_band_ratio"]),
                "high_band_ratio": float(st["high_band_ratio"]),
                "score": float(sc),
                "passed_thresholds": bool(passed),
                "is_selected": bool(i == chosen_idx),
            }
        )

    return {
        "selected": bool(selected),
        "index": chosen_idx,
        "time_start": float(time_start),
        "time_end": float(time_end),
        "signal_level": float(stats[chosen_idx]["rms"]),
        "selected_windows": int(candidate_indices.size),
        "energy_threshold": energy_thr,
        "band_threshold": band_thr,
        "window_details": details,
    }


def select_informative_window(
    windows: Sequence[np.ndarray],
    sr: int,
    bounds: Sequence[Tuple[float, float]],
    cfg: Dict[str, object],
) -> Dict[str, object]:
    analyzed = analyze_windows(windows=windows, sr=sr, bounds=bounds, cfg=cfg)
    return {
        "selected": analyzed["selected"],
        "index": analyzed["index"],
        "time_start": analyzed["time_start"],
        "time_end": analyzed["time_end"],
        "signal_level": analyzed["signal_level"],
        "selected_windows": analyzed["selected_windows"],
        "energy_threshold": analyzed["energy_threshold"],
        "band_threshold": analyzed["band_threshold"],
    }


def guess_class_from_window_stats(
    stats: Dict[str, float | int | bool],
    cfg: Dict[str, object] | None = None,
) -> Dict[str, object]:
    cfg = dict(cfg or {})

    centroid = float(stats.get("spectral_centroid", 0.0))
    dom_freq = float(stats.get("dominant_freq", 0.0))
    low_ratio = float(stats.get("low_band_ratio", 0.0))
    mid_ratio = float(stats.get("mid_band_ratio", 0.0))
    high_ratio = float(stats.get("high_band_ratio", 0.0))
    peaks = float(stats.get("num_spectral_peaks", 0.0))
    energy = float(stats.get("energy", 0.0))

    silence_energy_threshold = float(cfg.get("silence_energy_threshold", 1e-4))
    drone_high_ratio_threshold = float(cfg.get("drone_high_ratio_threshold", 0.45))
    drone_centroid_threshold = float(cfg.get("drone_centroid_threshold", 1600.0))
    drone_mid_ratio_max = float(cfg.get("drone_mid_ratio_max", 0.30))
    background_centroid_threshold = float(cfg.get("background_centroid_threshold", 1050.0))
    background_high_ratio_threshold = float(cfg.get("background_high_ratio_threshold", 0.20))
    background_peaks_threshold = float(cfg.get("background_peaks_threshold", 260.0))
    helicopter_low_ratio_threshold = float(cfg.get("helicopter_low_ratio_threshold", 0.48))
    helicopter_dom_freq_threshold = float(cfg.get("helicopter_dom_freq_threshold", 260.0))
    airplane_mid_ratio_threshold = float(cfg.get("airplane_mid_ratio_threshold", 0.40))

    label = "background"
    reason = "fallback_background"

    if energy <= silence_energy_threshold:
        label = "background"
        reason = "very_low_energy"
    elif (high_ratio >= drone_high_ratio_threshold and mid_ratio <= drone_mid_ratio_max) or (
        centroid >= drone_centroid_threshold and high_ratio >= background_high_ratio_threshold
    ):
        label = "drone"
        reason = "high_frequency_signature"
    elif centroid >= background_centroid_threshold and (
        high_ratio >= background_high_ratio_threshold or peaks >= background_peaks_threshold
    ):
        label = "background"
        reason = "broadband_noise_signature"
    elif low_ratio >= helicopter_low_ratio_threshold and dom_freq <= helicopter_dom_freq_threshold:
        label = "helicopter"
        reason = "low_frequency_rotor_signature"
    elif mid_ratio >= airplane_mid_ratio_threshold and high_ratio < drone_high_ratio_threshold:
        label = "airplane"
        reason = "mid_band_aircraft_signature"
    elif low_ratio >= mid_ratio:
        label = "helicopter"
        reason = "low_band_dominance_fallback"
    else:
        label = "airplane"
        reason = "mid_band_dominance_fallback"

    return {
        "label": label,
        "reason": reason,
        "scores": {
            "centroid": centroid,
            "dominant_freq": dom_freq,
            "low_band_ratio": low_ratio,
            "mid_band_ratio": mid_ratio,
            "high_band_ratio": high_ratio,
            "num_spectral_peaks": peaks,
            "energy": energy,
        },
    }
