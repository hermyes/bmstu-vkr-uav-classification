from __future__ import annotations

import numpy as np

from src.features.feature_extractor import extract_features, extract_features_with_names


def test_feature_extraction_shapes() -> None:
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    audio = (0.8 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    cfg = {
        "mfcc": {"n_mfcc": 13, "use_delta": True, "use_delta_delta": True},
        "rms": {"enabled": True},
        "spectral": {"use_centroid": True, "use_bandwidth": True, "use_rolloff": True, "use_zcr": True},
        "band_energy": {"enabled": True, "bands": [[80, 300], [300, 1000], [1000, 3000], [3000, 8000]]},
        "spectral_peaks": {"enabled": True, "n_peaks": 3, "min_peak_height_ratio": 0.2},
    }

    vector, names = extract_features_with_names(audio, sr=sr, config=cfg)
    vector2 = extract_features(audio, sr=sr, config=cfg)

    assert vector.ndim == 1
    assert vector.shape == vector2.shape
    assert len(names) == len(vector)
    assert len(vector) > 20
