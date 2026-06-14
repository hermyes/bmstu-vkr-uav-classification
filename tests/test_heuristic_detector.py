from __future__ import annotations

from src.signal.heuristic_detector import guess_class_from_window_stats


def test_guess_class_from_window_stats_labels() -> None:
    drone_guess = guess_class_from_window_stats(
        {
            "energy": 0.05,
            "spectral_centroid": 2200.0,
            "dominant_freq": 340.0,
            "low_band_ratio": 0.12,
            "mid_band_ratio": 0.18,
            "high_band_ratio": 0.62,
            "num_spectral_peaks": 120,
        }
    )
    assert drone_guess["label"] == "drone"

    heli_guess = guess_class_from_window_stats(
        {
            "energy": 0.05,
            "spectral_centroid": 520.0,
            "dominant_freq": 160.0,
            "low_band_ratio": 0.62,
            "mid_band_ratio": 0.24,
            "high_band_ratio": 0.09,
            "num_spectral_peaks": 80,
        }
    )
    assert heli_guess["label"] == "helicopter"

    bg_guess = guess_class_from_window_stats(
        {
            "energy": 0.05,
            "spectral_centroid": 1450.0,
            "dominant_freq": 170.0,
            "low_band_ratio": 0.20,
            "mid_band_ratio": 0.35,
            "high_band_ratio": 0.35,
            "num_spectral_peaks": 330,
        }
    )
    assert bg_guess["label"] == "background"
