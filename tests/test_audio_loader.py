from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from src.data.audio_loader import get_audio_duration, load_audio, validate_audio


def test_load_audio_resample_and_mono(tmp_path: Path) -> None:
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sig_left = 0.5 * np.sin(2 * np.pi * 440 * t)
    sig_right = 0.3 * np.sin(2 * np.pi * 660 * t)
    stereo = np.vstack([sig_left, sig_right]).T.astype(np.float32)

    wav_path = tmp_path / "stereo.wav"
    sf.write(wav_path, stereo, samplerate=sr)

    audio, out_sr = load_audio(wav_path, target_sr=22050, mono=True, normalize=True)
    assert out_sr == 22050
    assert audio.ndim == 1
    assert np.max(np.abs(audio)) <= 1.0 + 1e-6
    assert validate_audio(wav_path)
    assert get_audio_duration(wav_path) > 0
