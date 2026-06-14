from __future__ import annotations

import io
import json
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st

from src.app.pipeline import UAVClassificationPipeline
from src.config_utils import load_json, load_yaml
from src.data.audio_loader import load_audio
from src.features.feature_extractor import extract_features, extract_features_with_names
from src.models.svm_model import predict_with_probabilities
from src.signal.heuristic_detector import analyze_windows, guess_class_from_window_stats
from src.signal.preprocessing import preprocess_audio
from src.signal.segmentation import split_into_windows


st.set_page_config(
    page_title="Демо классификации БПЛА по звуку",
    page_icon=":material/graphic_eq:",
    layout="wide",
)


def _read_json_if_exists(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        data = load_json(path)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


@st.cache_resource(show_spinner=False)
def _load_pipeline_cached(
    model_path: str,
    scaler_path: str,
    label_map_path: str,
    config_path: str,
) -> Tuple[UAVClassificationPipeline, Dict[str, Any]]:
    cfg = load_yaml(config_path)
    pipeline = UAVClassificationPipeline.from_paths(
        model_path=model_path,
        scaler_path=scaler_path,
        label_map_path=label_map_path,
        config=cfg,
    )
    return pipeline, cfg


def _build_probabilities(
    probs: np.ndarray,
    model: Any,
    id_to_label: Dict[int, str],
) -> Dict[str, float]:
    if probs.ndim != 2 or probs.shape[0] == 0:
        return {}
    classes = getattr(model, "classes_", None)
    if classes is None or len(classes) != probs.shape[1]:
        classes = np.arange(probs.shape[1], dtype=np.int64)
    classes = [int(c) for c in classes]
    pairs = []
    for class_idx, class_id in enumerate(classes):
        pairs.append((id_to_label.get(class_id, f"class_{class_id}"), float(probs[0, class_idx])))
    pairs.sort(key=lambda item: item[1], reverse=True)
    return {name: value for name, value in pairs}


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _safe_int(x: Any) -> int:
    try:
        return int(x)
    except Exception:
        return 0


def _analyze_audio(
    audio_path: str | Path,
    pipeline: UAVClassificationPipeline,
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any], np.ndarray, np.ndarray, int]:
    preprocessing_cfg = dict(cfg.get("preprocessing", {}))
    segmentation_cfg = dict(cfg.get("segmentation", {}))
    heuristic_cfg = dict(cfg.get("heuristic", {}))
    features_cfg = dict(cfg.get("features", {}))
    decision_cfg = dict(cfg.get("decision", {}))

    target_sr = preprocessing_cfg.get("target_sample_rate", 22050)
    mono = bool(preprocessing_cfg.get("mono", True))
    normalize = bool(preprocessing_cfg.get("normalize", True))
    audio, sr = load_audio(
        path=audio_path,
        target_sr=int(target_sr) if target_sr is not None else None,
        mono=mono,
        normalize=normalize,
    )
    processed = preprocess_audio(audio, sr=sr, cfg=preprocessing_cfg)

    window_sec = float(segmentation_cfg.get("window_sec", 1.0))
    hop_sec = float(segmentation_cfg.get("hop_sec", 0.5))
    pad_end = bool(segmentation_cfg.get("pad_end", False))
    windows, bounds = split_into_windows(
        audio=processed,
        sr=sr,
        window_sec=window_sec,
        hop_sec=hop_sec,
        pad_end=pad_end,
    )
    heur = analyze_windows(windows=windows, sr=sr, bounds=bounds, cfg=heuristic_cfg)
    details_by_idx = {_safe_int(d.get("index")): d for d in heur.get("window_details", [])}

    threshold = float(decision_cfg.get("confidence_threshold", 0.65))
    rows: List[Dict[str, Any]] = []

    for i, (window, (start_t, end_t)) in enumerate(zip(windows, bounds)):
        feats = extract_features(window, sr=sr, config=features_cfg)
        x = feats.reshape(1, -1)
        if pipeline.scaler is not None:
            x = pipeline.scaler.transform(x)
        y_pred, probs = predict_with_probabilities(pipeline.model, x)
        label_id = _safe_int(y_pred[0]) if len(y_pred) > 0 else -1
        label_name = pipeline.id_to_label.get(label_id, "unknown")
        conf = float(np.max(probs[0])) if probs.size else 0.0
        accepted = conf >= threshold

        probs_map = _build_probabilities(probs=probs, model=pipeline.model, id_to_label=pipeline.id_to_label)

        detail = details_by_idx.get(i, {})
        heuristic_guess = guess_class_from_window_stats(
            detail,
            cfg=dict(heuristic_cfg.get("type_guess", {})),
        )

        row = {
            "chunk_idx": i,
            "time_start": float(start_t),
            "time_end": float(end_t),
            "duration_sec": float(end_t - start_t),
            "pred_label": label_name,
            "pred_label_id": label_id,
            "confidence": conf,
            "decision": "accepted" if accepted else "rejected",
            "heur_selected_by_threshold": bool(heur.get("selected", False) and _safe_int(heur.get("index")) == i),
            "heur_is_selected_window": bool(_safe_int(heur.get("index")) == i),
            "heur_passed_thresholds": bool(detail.get("passed_thresholds", False)),
            "heur_guess_label": str(heuristic_guess.get("label", "unknown")),
            "heur_guess_reason": str(heuristic_guess.get("reason", "unknown")),
            "rms": _safe_float(detail.get("rms", 0.0)),
            "energy": _safe_float(detail.get("energy", 0.0)),
            "band_energy": _safe_float(detail.get("band_energy", 0.0)),
            "num_spectral_peaks": _safe_float(detail.get("num_spectral_peaks", 0.0)),
            "spectral_centroid": _safe_float(detail.get("spectral_centroid", 0.0)),
            "dominant_freq": _safe_float(detail.get("dominant_freq", 0.0)),
            "low_band_ratio": _safe_float(detail.get("low_band_ratio", 0.0)),
            "mid_band_ratio": _safe_float(detail.get("mid_band_ratio", 0.0)),
            "high_band_ratio": _safe_float(detail.get("high_band_ratio", 0.0)),
        }

        for class_name, class_prob in probs_map.items():
            row[f"prob_{class_name}"] = float(class_prob)
        rows.append(row)

    chunks_df = pd.DataFrame(rows)
    final_result = pipeline.infer_file(audio_path, include_diagnostics=True)
    final_result["heuristic_analysis"] = {
        "energy_threshold": float(heur.get("energy_threshold", 0.0)),
        "band_threshold": float(heur.get("band_threshold", 0.0)),
        "selected_window_index": _safe_int(heur.get("index", -1)),
        "selected_windows": _safe_int(heur.get("selected_windows", 0)),
    }
    return chunks_df, final_result, audio, processed, sr


def _render_waveform_plot(raw_audio: np.ndarray, processed_audio: np.ndarray, sr: int, result: Dict[str, Any]) -> None:
    t_raw = np.arange(len(raw_audio), dtype=np.float64) / float(sr)
    t_proc = np.arange(len(processed_audio), dtype=np.float64) / float(sr)
    start_t = float(result.get("time_start", 0.0))
    end_t = float(result.get("time_end", 0.0))

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(t_raw, raw_audio, linewidth=0.7)
    axes[0].axvspan(start_t, end_t, alpha=0.25, color="orange")
    axes[0].set_title("Исходная волновая форма")
    axes[0].set_ylabel("Амплитуда")
    axes[0].grid(alpha=0.2)

    axes[1].plot(t_proc, processed_audio, linewidth=0.7, color="tab:green")
    axes[1].axvspan(start_t, end_t, alpha=0.25, color="orange")
    axes[1].set_title("Волновая форма после предобработки")
    axes[1].set_xlabel("Время (сек)")
    axes[1].set_ylabel("Амплитуда")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)


def _compute_spectrum_db(audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    if audio.size == 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    win = np.hanning(len(audio)).astype(np.float64)
    sig = audio.astype(np.float64) * win
    spectrum = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(len(sig), d=1.0 / float(sr))
    db = 20.0 * np.log10(spectrum + 1e-12)
    return freqs, db


def _band_power(audio: np.ndarray, sr: int, low_hz: float, high_hz: float) -> float:
    if audio.size == 0:
        return 0.0
    spec = np.abs(np.fft.rfft(audio.astype(np.float64))) ** 2
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / float(sr))
    mask = (freqs >= float(low_hz)) & (freqs < float(high_hz))
    if not np.any(mask):
        return 0.0
    return float(np.sum(spec[mask]))


def _render_preprocess_metrics(
    raw_audio: np.ndarray,
    processed_audio: np.ndarray,
    sr: int,
    band_low_hz: float,
    band_high_hz: float,
) -> None:
    n = min(len(raw_audio), len(processed_audio))
    if n == 0:
        return
    raw = raw_audio[:n].astype(np.float64)
    proc = processed_audio[:n].astype(np.float64)
    diff = proc - raw
    eps = 1e-12

    corr = float(np.corrcoef(raw, proc)[0, 1]) if n > 1 else 1.0
    mae = float(np.mean(np.abs(diff)))
    rms_raw = float(np.sqrt(np.mean(np.square(raw))))
    rms_proc = float(np.sqrt(np.mean(np.square(proc))))

    low_raw = _band_power(raw, sr=sr, low_hz=0.0, high_hz=max(1.0, band_low_hz))
    low_proc = _band_power(proc, sr=sr, low_hz=0.0, high_hz=max(1.0, band_low_hz))
    high_raw = _band_power(raw, sr=sr, low_hz=band_high_hz, high_hz=sr * 0.5)
    high_proc = _band_power(proc, sr=sr, low_hz=band_high_hz, high_hz=sr * 0.5)

    low_reduction_db = float(10.0 * np.log10((low_raw + eps) / (low_proc + eps)))
    high_reduction_db = float(10.0 * np.log10((high_raw + eps) / (high_proc + eps)))

    st.caption(
        "Почему визуально почти одинаково: полосовая фильтрация меняет в первую очередь частотный состав, "
        "а не общую форму сигнала во времени."
    )
    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
    mc1.metric("Корреляция raw/proc", f"{corr:.4f}")
    mc2.metric("Средн. |Δ|", f"{mae:.6f}")
    mc3.metric("RMS raw", f"{rms_raw:.6f}")
    mc4.metric("RMS proc", f"{rms_proc:.6f}")
    mc5.metric(f"Ослабление < {band_low_hz:.0f} Гц", f"{low_reduction_db:.2f} дБ")
    mc6.metric(f"Ослабление > {band_high_hz:.0f} Гц", f"{high_reduction_db:.2f} дБ")


def _render_spectrum_comparison(
    raw_audio: np.ndarray,
    processed_audio: np.ndarray,
    sr: int,
    band_low_hz: float,
    band_high_hz: float,
) -> None:
    freqs_raw, raw_db = _compute_spectrum_db(raw_audio, sr=sr)
    freqs_proc, proc_db = _compute_spectrum_db(processed_audio, sr=sr)
    if freqs_raw.size == 0 or freqs_proc.size == 0:
        return

    max_hz = min(float(sr) * 0.5, 10000.0)
    mask_raw = freqs_raw <= max_hz
    mask_proc = freqs_proc <= max_hz

    common_n = min(np.sum(mask_raw), np.sum(mask_proc))
    fr = freqs_raw[mask_raw][:common_n]
    rd = raw_db[mask_raw][:common_n]
    pd = proc_db[mask_proc][:common_n]
    delta_db = pd - rd

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(fr, rd, label="До предобработки", linewidth=0.9, alpha=0.85)
    axes[0].plot(fr, pd, label="После предобработки", linewidth=0.9, alpha=0.85)
    axes[0].axvspan(band_low_hz, min(band_high_hz, max_hz), color="green", alpha=0.10, label="Полоса пропускания")
    axes[0].set_title("Сравнение спектров (дБ)")
    axes[0].set_ylabel("Амплитуда, дБ")
    axes[0].grid(alpha=0.2)
    axes[0].legend(loc="upper right")

    axes[1].plot(fr, delta_db, color="tab:red", linewidth=0.9)
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axes[1].axvspan(band_low_hz, min(band_high_hz, max_hz), color="green", alpha=0.10)
    axes[1].set_title("Разница спектров: после - до")
    axes[1].set_xlabel("Частота, Гц")
    axes[1].set_ylabel("Δ, дБ")
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)


def _safe_fft_size(audio_len: int) -> int:
    if audio_len <= 512:
        return max(64, audio_len)
    if audio_len <= 1024:
        return 512
    if audio_len <= 2048:
        return 1024
    return 2048


def _extract_selected_window_audio(
    audio: np.ndarray,
    sr: int,
    result: Dict[str, Any],
) -> Tuple[np.ndarray, float, float]:
    if audio.size == 0 or sr <= 0:
        return np.zeros((0,), dtype=np.float32), 0.0, 0.0

    duration = float(len(audio) / sr)
    start_t = float(result.get("time_start", 0.0))
    end_t = float(result.get("time_end", 0.0))
    start_t = max(0.0, min(start_t, duration))
    end_t = max(start_t + (1.0 / float(sr)), min(end_t, duration))

    start_idx = int(round(start_t * sr))
    end_idx = int(round(end_t * sr))
    start_idx = max(0, min(start_idx, len(audio) - 1))
    end_idx = max(start_idx + 1, min(end_idx, len(audio)))
    return audio[start_idx:end_idx].astype(np.float32), float(start_t), float(end_t)


def _slice_context_segment(
    audio: np.ndarray,
    sr: int,
    selected_start_t: float,
    selected_end_t: float,
    context_sec: float = 4.0,
    max_total_sec: float = 30.0,
) -> Tuple[np.ndarray, float, float]:
    duration = float(len(audio) / sr) if sr > 0 else 0.0
    if duration <= 0.0:
        return np.zeros((0,), dtype=np.float32), 0.0, 0.0

    left = max(0.0, selected_start_t - float(context_sec))
    right = min(duration, selected_end_t + float(context_sec))

    if (right - left) > float(max_total_sec):
        center = 0.5 * (selected_start_t + selected_end_t)
        half = 0.5 * float(max_total_sec)
        left = max(0.0, center - half)
        right = min(duration, center + half)

    start_idx = int(round(left * sr))
    end_idx = int(round(right * sr))
    start_idx = max(0, min(start_idx, len(audio) - 1))
    end_idx = max(start_idx + 1, min(end_idx, len(audio)))
    return audio[start_idx:end_idx].astype(np.float32), float(left), float(right)


def _render_time_frequency_analysis(processed_audio: np.ndarray, sr: int, result: Dict[str, Any]) -> None:
    selected_audio, sel_start, sel_end = _extract_selected_window_audio(processed_audio, sr=sr, result=result)
    if selected_audio.size == 0:
        st.warning("Не удалось выделить окно для временно-частотного анализа.")
        return

    context_audio, ctx_start, ctx_end = _slice_context_segment(
        processed_audio,
        sr=sr,
        selected_start_t=sel_start,
        selected_end_t=sel_end,
        context_sec=4.0,
        max_total_sec=30.0,
    )
    if context_audio.size == 0:
        st.warning("Не удалось построить контекстный фрагмент для STFT/MFCC.")
        return

    n_fft = _safe_fft_size(len(context_audio))
    hop_length = max(1, n_fft // 4)
    stft = librosa.stft(context_audio, n_fft=n_fft, hop_length=hop_length)
    spec_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    mfcc = librosa.feature.mfcc(y=context_audio, sr=sr, n_mfcc=20, n_fft=n_fft, hop_length=hop_length)

    local_sel_start = max(0.0, sel_start - ctx_start)
    local_sel_end = min(float(len(context_audio) / sr), sel_end - ctx_start)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    img0 = librosa.display.specshow(
        spec_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="hz",
        ax=axes[0],
    )
    axes[0].axvspan(local_sel_start, local_sel_end, color="orange", alpha=0.25)
    axes[0].set_title("STFT-спектрограмма (дБ)")
    axes[0].grid(alpha=0.2)
    fig.colorbar(img0, ax=axes[0], format="%+2.0f dB")

    img1 = librosa.display.specshow(
        mfcc,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        ax=axes[1],
    )
    axes[1].axvspan(local_sel_start, local_sel_end, color="orange", alpha=0.25)
    axes[1].set_title("MFCC-карта (20 коэффициентов)")
    axes[1].set_ylabel("MFCC index")
    axes[1].grid(alpha=0.2)
    fig.colorbar(img1, ax=axes[1])

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.caption(
        f"Показан контекстный фрагмент [{ctx_start:.2f}; {ctx_end:.2f}] сек, "
        f"выбранное окно: [{sel_start:.2f}; {sel_end:.2f}] сек."
    )


def _feature_group_name(feature_name: str) -> str:
    if feature_name.startswith("mfcc_delta2_"):
        return "mfcc_delta2"
    if feature_name.startswith("mfcc_delta_"):
        return "mfcc_delta"
    if feature_name.startswith("mfcc_"):
        return "mfcc"
    if feature_name.startswith("rms_"):
        return "rms"
    if feature_name.startswith("spectral_peak_"):
        return "spectral_peaks"
    if feature_name.startswith("spectral_"):
        return "spectral"
    if feature_name.startswith("zero_crossing_"):
        return "zcr"
    if feature_name.startswith("band_energy_"):
        return "band_energy"
    return "other"


def _render_selected_window_features(
    processed_audio: np.ndarray,
    sr: int,
    result: Dict[str, Any],
    features_cfg: Dict[str, Any],
    heuristic_cfg: Dict[str, Any],
) -> None:
    selected_audio, sel_start_t, sel_end_t = _extract_selected_window_audio(processed_audio, sr=sr, result=result)
    if selected_audio.size < 64:
        st.warning("Слишком короткое выбранное окно для извлечения признаков.")
        return

    diagnostics = dict(result.get("diagnostics", {}))
    heur = dict(diagnostics.get("heuristic", {}))
    selected_details = dict(heur.get("selected_window_details", {}))
    heur_selected = bool(heur.get("selected", False))
    selected_idx = _safe_int(selected_details.get("index", result.get("selected_window_index", -1)))

    energy = _safe_float(selected_details.get("energy", 0.0))
    band_energy = _safe_float(selected_details.get("band_energy", 0.0))
    peaks = _safe_int(selected_details.get("num_spectral_peaks", 0))
    energy_thr = _safe_float(heur.get("energy_threshold", 0.0))
    band_thr = _safe_float(heur.get("band_threshold", 0.0))
    min_peaks = _safe_float(heuristic_cfg.get("min_peaks", 3.0))
    passed_energy = energy > energy_thr
    passed_band = band_energy > band_thr
    passed_peaks = float(peaks) >= float(min_peaks)

    reason_map = {
        "very_low_energy": "очень низкая энергия сигнала",
        "high_frequency_signature": "преобладают высокочастотные компоненты",
        "broadband_noise_signature": "широкополосный шумовой спектр",
        "low_frequency_rotor_signature": "доминируют низкочастотные роторные компоненты",
        "mid_band_aircraft_signature": "доминирует среднечастотная полоса",
        "low_band_dominance_fallback": "эвристическое правило fallback по доминированию низкой полосы",
        "mid_band_dominance_fallback": "эвристическое правило fallback по доминированию средней полосы",
        "no_selected_window": "окно не было выбрано эвристикой",
        "unknown": "причина не определена",
    }
    heur_guess_reason_code = str(diagnostics.get("heuristic_guess_reason", "unknown"))
    heur_guess_reason_text = reason_map.get(heur_guess_reason_code, heur_guess_reason_code)

    st.info(
        "Рассматривается окно "
        f"№{selected_idx} [{sel_start_t:.2f}; {sel_end_t:.2f}] сек, "
        f"потому что: energy={energy:.6f} {'>' if passed_energy else '<='} {energy_thr:.6f}, "
        f"band_energy={band_energy:.6f} {'>' if passed_band else '<='} {band_thr:.6f}, "
        f"peaks={peaks} {'>=' if passed_peaks else '<'} {min_peaks:.0f}. "
        f"Режим выбора: {'прошло все пороги' if heur_selected else 'fallback по максимальному score'}. "
        f"Причина эвристического типа: {heur_guess_reason_text}."
    )

    feature_vector, feature_names = extract_features_with_names(selected_audio, sr=sr, config=features_cfg)
    features_df = pd.DataFrame(
        {
            "feature": feature_names,
            "value": feature_vector.astype(np.float64),
        }
    )
    features_df["group"] = features_df["feature"].apply(_feature_group_name)

    c1, c2 = st.columns(2)
    c1.metric("Размер вектора признаков", str(len(features_df)))
    c2.metric("Групп признаков", str(features_df["group"].nunique()))

    st.markdown("**Распределение количества признаков по группам**")
    group_counts = features_df["group"].value_counts().sort_values(ascending=False)
    st.bar_chart(group_counts, height=220)

    group_options = sorted(features_df["group"].unique().tolist())
    selected_groups = st.multiselect(
        "Фильтр групп признаков",
        options=group_options,
        default=group_options,
        help="По умолчанию показаны все признаки выбранного окна.",
    )
    visible_df = features_df[features_df["group"].isin(selected_groups)].copy()
    st.caption(f"Показано признаков: {len(visible_df)} из {len(features_df)}.")

    st.markdown("**Полный вектор признаков выбранного окна**")
    st.dataframe(visible_df[["group", "feature", "value"]], use_container_width=True, height=480)

    features_csv = features_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Скачать вектор признаков выбранного окна (CSV)",
        data=features_csv,
        file_name="selected_window_features.csv",
        mime="text/csv",
        use_container_width=False,
    )

    if selected_details:
        st.markdown("**Ключевые эвристические признаки выбранного окна**")
        e1, e2, e3, e4, e5 = st.columns(5)
        e1.metric("Энергия", f"{_safe_float(selected_details.get('energy')):.6f}")
        e2.metric("Энергия в полосе", f"{_safe_float(selected_details.get('band_energy')):.6f}")
        e3.metric("Спектр. центроид, Гц", f"{_safe_float(selected_details.get('spectral_centroid')):.1f}")
        e4.metric("Доминирующая частота, Гц", f"{_safe_float(selected_details.get('dominant_freq')):.1f}")
        e5.metric("Число спектр. пиков", f"{_safe_int(selected_details.get('num_spectral_peaks'))}")


def _render_overview_metrics(
    training_summary: Dict[str, Any],
    test_metrics: Dict[str, Any],
    heuristic_metrics: Dict[str, Any],
) -> None:
    model_name = "SVM"
    accuracy_test = _safe_float(test_metrics.get("accuracy", 0.0))
    f1_test = _safe_float(test_metrics.get("f1_macro", 0.0))
    heur_acc = _safe_float(heuristic_metrics.get("accuracy", 0.0))
    heur_f1 = _safe_float(heuristic_metrics.get("f1_macro", 0.0))
    heur_n = _safe_int(heuristic_metrics.get("n_tested", 0))
    f1_gap = f1_test - heur_f1

    st.markdown(
        """
        <style>
        .uav-card {
            padding: 0.9rem 1rem;
            border-radius: 14px;
            border: 1px solid rgba(16, 42, 67, 0.15);
            background: linear-gradient(180deg, #f8fbff 0%, #eef5fb 100%);
        }
        .uav-card-title {
            font-size: 0.9rem;
            color: #486581;
            margin-bottom: 0.35rem;
        }
        .uav-card-value {
            font-size: 2.1rem;
            color: #102a43;
            font-weight: 700;
            line-height: 1.05;
        }
        .uav-card-note {
            font-size: 0.8rem;
            color: #627d98;
            margin-top: 0.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"""
            <div class="uav-card">
              <div class="uav-card-title">Модель (по умолчанию)</div>
              <div class="uav-card-value">{model_name}</div>
              <div class="uav-card-note">Базовая модель классификации</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
            <div class="uav-card">
              <div class="uav-card-title">Точность на тестовой выборке</div>
              <div class="uav-card-value">{accuracy_test:.4f}</div>
              <div class="uav-card-note">Метрика Accuracy</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
            <div class="uav-card">
              <div class="uav-card-title">F1-macro на тестовой выборке</div>
              <div class="uav-card-value">{f1_test:.4f}</div>
              <div class="uav-card-note">Качество ML-модели</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""
            <div class="uav-card">
              <div class="uav-card-title">F1-macro эвристики</div>
              <div class="uav-card-value">{heur_f1:.4f}</div>
              <div class="uav-card-note">Без ML-классификатора</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption("Разница F1-macro (ML - эвристика): {:.4f}".format(f1_gap))
    st.progress(max(0.0, min(1.0, f1_test)), text=f"Качество ML-модели (F1): {f1_test:.4f}")
    st.progress(max(0.0, min(1.0, heur_f1)), text=f"Качество эвристики (F1): {heur_f1:.4f}")

    best_params = training_summary.get("best_params", {})
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Лучшие гиперпараметры SVM")
        if isinstance(best_params, dict) and best_params:
            params_df = pd.DataFrame(
                [{"Параметр": str(k), "Значение": str(v)} for k, v in best_params.items()]
            )
            st.table(params_df)
        else:
            st.info("Параметры не найдены в отчете обучения.")

    with right:
        st.subheader("Качество только эвристики")
        h1, h2, h3 = st.columns(3)
        h1.metric("Accuracy", f"{heur_acc:.4f}")
        h2.metric("F1-macro", f"{heur_f1:.4f}")
        h3.metric("Объектов в тесте", f"{heur_n}")

        compare_df = pd.DataFrame(
            {
                "Источник": ["ML-модель (F1)", "Эвристика (F1)"],
                "Значение": [f1_test, heur_f1],
            }
        ).set_index("Источник")
        st.bar_chart(compare_df, height=220)

def _render_live_simulation(chunks_df: pd.DataFrame, playback_delay: float, display_every_n: int) -> None:
    if chunks_df.empty:
        st.warning("Нет чанков для отображения.")
        return

    status = st.status("Идет инференс в режиме реального времени по чанкам...", expanded=True)
    progress = st.progress(0, text="Запуск...")
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    m1 = mcol1.empty()
    m2 = mcol2.empty()
    m3 = mcol3.empty()
    m4 = mcol4.empty()
    st.caption("Live-графики тоже разделены: разные масштабы метрик не смешиваются.")
    live_chart_col1, live_chart_col2, live_chart_col3 = st.columns(3)
    conf_chart_placeholder = live_chart_col1.empty()
    energy_chart_placeholder = live_chart_col2.empty()
    band_energy_chart_placeholder = live_chart_col3.empty()
    log_placeholder = st.empty()

    total = len(chunks_df)
    log_lines: List[str] = []
    timeline_rows: List[Dict[str, Any]] = []

    for idx, row in chunks_df.iterrows():
        timeline_rows.append(
            {
                "chunk_idx": int(row["chunk_idx"]),
                "confidence": _safe_float(row["confidence"]),
                "energy": _safe_float(row["energy"]),
                "band_energy": _safe_float(row["band_energy"]),
            }
        )

        if idx % max(display_every_n, 1) == 0 or idx == total - 1:
            m1.metric("Чанк", f"{int(row['chunk_idx']) + 1}/{total}")
            m2.metric("Текущее предсказание", str(row["pred_label"]))
            m3.metric("Уверенность", f"{_safe_float(row['confidence']):.3f}")
            m4.metric("Эвристическое предположение", str(row["heur_guess_label"]))

            log_lines.append(
                f"[{_safe_float(row['time_start']):7.2f}с .. {_safe_float(row['time_end']):7.2f}с] "
                f"модель={row['pred_label']:<11} увер={_safe_float(row['confidence']):.3f} "
                f"эвр={row['heur_guess_label']:<11} порог={bool(row['heur_passed_thresholds'])}"
            )
            log_placeholder.code("\n".join(log_lines[-20:]), language="text")

            timeline_df = pd.DataFrame(timeline_rows).set_index("chunk_idx")
            conf_chart_placeholder.line_chart(
                timeline_df[["confidence"]].rename(columns={"confidence": "Уверенность"}),
                height=220,
            )
            energy_chart_placeholder.line_chart(
                timeline_df[["energy"]].rename(columns={"energy": "Энергия"}),
                height=220,
            )
            band_energy_chart_placeholder.line_chart(
                timeline_df[["band_energy"]].rename(columns={"band_energy": "Энергия в полосе"}),
                height=220,
            )

        ratio = int(round(100.0 * float(idx + 1) / float(total)))
        progress.progress(ratio, text=f"Обработано {idx + 1}/{total} чанков")
        if playback_delay > 0.0:
            time.sleep(playback_delay)

    status.update(label="Симуляция реального времени завершена.", state="complete", expanded=False)


def _render_chunk_tables_and_charts(chunks_df: pd.DataFrame) -> None:
    if chunks_df.empty:
        st.warning("Нет аналитики по чанкам.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Всего чанков", str(len(chunks_df)))
    col2.metric("Принятых чанков", str(int((chunks_df["decision"] == "accepted").sum())))
    col3.metric("Уникальных предсказанных классов", str(chunks_df["pred_label"].nunique()))

    st.caption("Графики разделены по метрикам, чтобы разные масштабы не скрывали важные изменения.")

    plot_df = chunks_df[["chunk_idx", "confidence", "energy", "band_energy"]].copy()
    plot_df = plot_df.sort_values("chunk_idx").set_index("chunk_idx")

    gcol1, gcol2 = st.columns(2)
    with gcol1:
        st.markdown("**Уверенность модели по чанкам**")
        st.line_chart(plot_df[["confidence"]].rename(columns={"confidence": "Уверенность"}), height=240)
    with gcol2:
        st.markdown("**Энергия по чанкам**")
        st.line_chart(plot_df[["energy"]].rename(columns={"energy": "Энергия"}), height=240)

    st.markdown("**Энергия в полосе по чанкам**")
    st.line_chart(plot_df[["band_energy"]].rename(columns={"band_energy": "Энергия в полосе"}), height=240)

    st.markdown("**Распределение предсказанных классов**")
    label_counts = chunks_df["pred_label"].value_counts().sort_values(ascending=False)
    st.bar_chart(label_counts, height=260)

    st.subheader("Предсказания по каждому чанку")
    view_cols = [
        "chunk_idx",
        "time_start",
        "time_end",
        "pred_label",
        "confidence",
        "heur_guess_label",
        "heur_guess_reason",
        "energy",
        "band_energy",
        "spectral_centroid",
        "dominant_freq",
        "num_spectral_peaks",
    ]
    safe_cols = [c for c in view_cols if c in chunks_df.columns]
    display_df = chunks_df[safe_cols].rename(
        columns={
            "chunk_idx": "Индекс чанка",
            "time_start": "Начало, с",
            "time_end": "Конец, с",
            "pred_label": "Предсказание модели",
            "confidence": "Уверенность",
            "heur_guess_label": "Предположение эвристики",
            "heur_guess_reason": "Причина предположения",
            "energy": "Энергия",
            "band_energy": "Энергия в полосе",
            "spectral_centroid": "Спектральный центроид",
            "dominant_freq": "Доминирующая частота",
            "num_spectral_peaks": "Число спектральных пиков",
        }
    )
    st.dataframe(display_df, use_container_width=True, height=380)

    csv_bytes = chunks_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Скачать аналитику по чанкам (CSV)",
        data=csv_bytes,
        file_name="chunk_analytics.csv",
        mime="text/csv",
        use_container_width=False,
    )


def _render_result_json(result: Dict[str, Any]) -> None:
    st.subheader("Итоговый JSON результата")
    st.json(result)
    json_bytes = json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button(
        label="Скачать JSON инференса",
        data=json_bytes,
        file_name="inference_result.json",
        mime="application/json",
        use_container_width=False,
    )


def _save_uploaded_audio(uploaded_file: Any) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(prefix="uav_demo_", suffix=suffix, delete=False)
    tmp.write(uploaded_file.getvalue())
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def _audio_to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    """Serialize mono float audio to WAV bytes for Streamlit playback/download."""
    if audio.ndim != 1:
        raise ValueError("Expected mono 1D audio array")
    clipped = np.clip(audio.astype(np.float32), -1.0, 1.0)
    buf = io.BytesIO()
    sf.write(buf, clipped, samplerate=int(sr), format="WAV", subtype="PCM_16")
    return buf.getvalue()


def main() -> None:
    st.markdown(
        """
        <div style="padding: 0.8rem 1rem; border-radius: 14px; background: linear-gradient(135deg, #102a43 0%, #334e68 100%); color: #f0f4f8;">
          <h2 style="margin: 0 0 0.4rem 0;">Классификация БПЛА по звуку — презентационная консоль</h2>
          <div style="margin: 0; font-size: 0.95rem;">Инференс по чанкам в реальном времени, диагностика модели, статистика эвристики и экспорт отчетов.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    default_model = "models/svm_baseline.pkl"
    default_scaler = "models/scaler.pkl"
    default_label_map = "models/label_map.json"
    default_config = "configs/inference.yaml"

    with st.sidebar:
        st.header("Настройка запуска")
        st.caption("Пути по умолчанию указывают на лучшую модель SVM.")
        model_path = st.text_input("Путь к модели", value=default_model)
        scaler_path = st.text_input("Путь к нормализатору (scaler)", value=default_scaler)
        label_map_path = st.text_input("Путь к карте классов", value=default_label_map)
        config_path = st.text_input("Путь к конфигу инференса", value=default_config)
        playback_delay = st.slider("Задержка между чанками (сек)", 0.0, 0.4, 0.03, 0.01)
        display_every_n = st.slider("Обновлять интерфейс каждые N чанков", 1, 15, 1, 1)
        st.caption("Поставь delay=0.0 для максимальной скорости.")

    try:
        pipeline, cfg = _load_pipeline_cached(
            model_path=model_path,
            scaler_path=scaler_path,
            label_map_path=label_map_path,
            config_path=config_path,
        )
    except Exception as exc:
        st.error(f"Не удалось загрузить пайплайн: {exc}")
        st.stop()

    training_summary = _read_json_if_exists("reports/metrics/svm_training_summary.json")
    test_metrics = _read_json_if_exists("reports/metrics/test_metrics.json")
    heuristic_metrics = _read_json_if_exists("reports/metrics/heuristic_type_guess_metrics.json")
    if not heuristic_metrics and isinstance(test_metrics.get("heuristic_type_guess"), dict):
        heuristic_metrics = dict(test_metrics["heuristic_type_guess"])

    overview_tab, live_tab, chunks_tab, output_tab = st.tabs(
        ["Обзор", "Инференс в реальном времени", "Аналитика чанков", "Итоговый вывод"]
    )

    with overview_tab:
        _render_overview_metrics(
            training_summary=training_summary,
            test_metrics=test_metrics,
            heuristic_metrics=heuristic_metrics,
        )
        cm_path = Path("reports/figures/confusion_matrix.png")
        if cm_path.exists():
            st.subheader("Матрица ошибок модели (тестовая выборка)")
            st.image(str(cm_path), use_container_width=True)
        else:
            st.info("Изображение матрицы ошибок не найдено: запусти scripts/evaluate.py для генерации.")

    with live_tab:
        st.subheader("Загрузка аудио и запуск инференса в реальном времени")
        uploaded = st.file_uploader("Загрузите .wav файл", type=["wav"])
        run_live = st.button("Запустить анализ в реальном времени", type="primary", disabled=uploaded is None)

        if uploaded is not None:
            st.audio(uploaded.getvalue(), format="audio/wav")
            st.caption(
                f"Файл: {uploaded.name} | Размер: {len(uploaded.getvalue()) / (1024 * 1024):.2f} МБ"
            )

        if run_live and uploaded is not None:
            audio_path = _save_uploaded_audio(uploaded)
            with st.spinner("Подготавливаю анализ по чанкам..."):
                chunks_df, result, raw_audio, processed_audio, sr = _analyze_audio(
                    audio_path=audio_path,
                    pipeline=pipeline,
                    cfg=cfg,
                )

            st.session_state["uav_chunks_df"] = chunks_df
            st.session_state["uav_result"] = result
            st.session_state["uav_raw_audio"] = raw_audio
            st.session_state["uav_processed_audio"] = processed_audio
            st.session_state["uav_sr"] = sr
            st.session_state["uav_last_file"] = uploaded.name
            st.session_state["uav_last_run_ts"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            _render_live_simulation(
                chunks_df=chunks_df,
                playback_delay=playback_delay,
                display_every_n=display_every_n,
            )
            st.success("Анализ завершен. Открой вкладки «Аналитика чанков» и «Итоговый вывод».")

    chunks_df = st.session_state.get("uav_chunks_df")
    result = st.session_state.get("uav_result")
    raw_audio = st.session_state.get("uav_raw_audio")
    processed_audio = st.session_state.get("uav_processed_audio")
    sr = st.session_state.get("uav_sr")

    with chunks_tab:
        if isinstance(chunks_df, pd.DataFrame) and not chunks_df.empty:
            st.caption(
                f"Последний запуск: {st.session_state.get('uav_last_run_ts', '-')}, "
                f"файл: {st.session_state.get('uav_last_file', '-')}"
            )
            _render_chunk_tables_and_charts(chunks_df)
        else:
            st.info("Пока нет аналитики по чанкам. Сначала запусти анализ в реальном времени.")

    with output_tab:
        if isinstance(result, dict):
            if isinstance(raw_audio, np.ndarray) and isinstance(processed_audio, np.ndarray) and isinstance(sr, int):
                st.subheader("Прослушивание сигналов")
                preprocessing_cfg = dict(cfg.get("preprocessing", {}))
                try:
                    raw_wav_bytes = _audio_to_wav_bytes(raw_audio, sr=sr)
                    proc_wav_bytes = _audio_to_wav_bytes(processed_audio, sr=sr)

                    ac1, ac2 = st.columns(2)
                    with ac1:
                        st.markdown("**Сигнал до предобработки (после приведения формата)**")
                        st.audio(raw_wav_bytes, format="audio/wav")
                    with ac2:
                        st.markdown("**Сигнал после предобработки**")
                        st.audio(proc_wav_bytes, format="audio/wav")

                    st.download_button(
                        label="Скачать предобработанный WAV",
                        data=proc_wav_bytes,
                        file_name="preprocessed_audio.wav",
                        mime="audio/wav",
                        use_container_width=False,
                    )
                except Exception as exc:
                    st.warning(f"Не удалось подготовить аудио для прослушивания: {exc}")

                band_low_hz = float(preprocessing_cfg.get("low_freq", 80.0))
                band_high_hz = float(preprocessing_cfg.get("high_freq", 8000.0))

                st.subheader("Диагностика волновой формы")
                _render_waveform_plot(raw_audio=raw_audio, processed_audio=processed_audio, sr=sr, result=result)

                st.subheader("Метрики разницы до/после предобработки")
                _render_preprocess_metrics(
                    raw_audio=raw_audio,
                    processed_audio=processed_audio,
                    sr=sr,
                    band_low_hz=band_low_hz,
                    band_high_hz=band_high_hz,
                )

                st.subheader("Сравнение спектра до/после предобработки")
                _render_spectrum_comparison(
                    raw_audio=raw_audio,
                    processed_audio=processed_audio,
                    sr=sr,
                    band_low_hz=band_low_hz,
                    band_high_hz=band_high_hz,
                )

                st.subheader("Временно-частотный анализ (STFT и MFCC)")
                _render_time_frequency_analysis(
                    processed_audio=processed_audio,
                    sr=sr,
                    result=result,
                )

                st.subheader("Признаки выбранного окна")
                _render_selected_window_features(
                    processed_audio=processed_audio,
                    sr=sr,
                    result=result,
                    features_cfg=dict(cfg.get("features", {})),
                    heuristic_cfg=dict(cfg.get("heuristic", {})),
                )
            _render_result_json(result)
        else:
            st.info("Пока нет итогового результата. Сначала запусти анализ в реальном времени.")


if __name__ == "__main__":
    main()

