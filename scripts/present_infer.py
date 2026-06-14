#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.app.pipeline import UAVClassificationPipeline
from src.config_utils import ensure_dir, load_yaml, save_json
from src.data.audio_loader import load_audio
from src.signal.preprocessing import preprocess_audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate presentation-ready inference report with visualizations"
    )
    parser.add_argument("--audio", type=str, required=True, help="Path to input .wav file")
    parser.add_argument("--model", type=str, default="models/svm_baseline.pkl", help="Path to model artifact")
    parser.add_argument("--scaler", type=str, default="models/scaler.pkl", help="Path to scaler artifact")
    parser.add_argument("--label-map", type=str, default="models/label_map.json", help="Path to label map artifact")
    parser.add_argument("--config", type=str, default="configs/inference.yaml", help="Path to inference config")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/presentation",
        help="Directory where the presentation report is created",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional custom run folder name",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="DPI for generated plots",
    )
    return parser.parse_args()


def _resolve_run_dir(base_dir: str | Path, audio_path: Path, run_name: str | None) -> Path:
    base_dir = Path(base_dir)
    if run_name:
        name = run_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{timestamp}_{audio_path.stem}"
    return ensure_dir(base_dir / name)


def _extract_window_arrays(window_details: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    if not window_details:
        return {
            "index": np.array([], dtype=np.int64),
            "time_start": np.array([], dtype=np.float64),
            "time_end": np.array([], dtype=np.float64),
            "time_center": np.array([], dtype=np.float64),
            "energy": np.array([], dtype=np.float64),
            "band_energy": np.array([], dtype=np.float64),
            "num_spectral_peaks": np.array([], dtype=np.float64),
            "score": np.array([], dtype=np.float64),
            "passed_thresholds": np.array([], dtype=np.bool_),
            "is_selected": np.array([], dtype=np.bool_),
        }

    idx = np.asarray([int(x["index"]) for x in window_details], dtype=np.int64)
    time_start = np.asarray([float(x["time_start"]) for x in window_details], dtype=np.float64)
    time_end = np.asarray([float(x["time_end"]) for x in window_details], dtype=np.float64)
    return {
        "index": idx,
        "time_start": time_start,
        "time_end": time_end,
        "time_center": (time_start + time_end) * 0.5,
        "energy": np.asarray([float(x["energy"]) for x in window_details], dtype=np.float64),
        "band_energy": np.asarray([float(x["band_energy"]) for x in window_details], dtype=np.float64),
        "num_spectral_peaks": np.asarray(
            [float(x["num_spectral_peaks"]) for x in window_details], dtype=np.float64
        ),
        "score": np.asarray([float(x["score"]) for x in window_details], dtype=np.float64),
        "passed_thresholds": np.asarray([bool(x["passed_thresholds"]) for x in window_details], dtype=np.bool_),
        "is_selected": np.asarray([bool(x["is_selected"]) for x in window_details], dtype=np.bool_),
    }


def _selected_time_bounds(result: Dict[str, Any]) -> Tuple[float, float]:
    return float(result.get("time_start", 0.0)), float(result.get("time_end", 0.0))


def _plot_waveform(
    raw_audio: np.ndarray,
    processed_audio: np.ndarray,
    sr: int,
    selected_bounds: Tuple[float, float],
    output_path: Path,
    dpi: int,
) -> None:
    t_raw = np.arange(len(raw_audio), dtype=np.float64) / float(sr)
    t_processed = np.arange(len(processed_audio), dtype=np.float64) / float(sr)
    start_t, end_t = selected_bounds

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 7), sharex=True)
    axes[0].plot(t_raw, raw_audio, color="#2b6cb0", linewidth=0.8)
    axes[0].axvspan(start_t, end_t, color="#f6ad55", alpha=0.25, label="Selected window")
    axes[0].set_title("Raw waveform")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.2)

    axes[1].plot(t_processed, processed_audio, color="#2f855a", linewidth=0.8)
    axes[1].axvspan(start_t, end_t, color="#f6ad55", alpha=0.25, label="Selected window")
    axes[1].set_title("Preprocessed waveform")
    axes[1].set_xlabel("Time, sec")
    axes[1].set_ylabel("Amplitude")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _plot_spectrogram(
    audio: np.ndarray,
    sr: int,
    selected_bounds: Tuple[float, float],
    output_path: Path,
    dpi: int,
) -> None:
    n_fft = 1024 if len(audio) >= 1024 else max(64, len(audio))
    hop_length = max(64, n_fft // 4)
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    spec_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    start_t, end_t = selected_bounds
    fig, ax = plt.subplots(figsize=(12, 5))
    img = librosa.display.specshow(spec_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="hz", ax=ax)
    ax.axvspan(start_t, end_t, color="#f6ad55", alpha=0.25, label="Selected window")
    ax.set_title("Spectrogram (dB)")
    ax.legend(loc="upper right")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _plot_class_probabilities(
    class_probabilities: Dict[str, float],
    predicted_label: str,
    output_path: Path,
    dpi: int,
) -> None:
    labels = list(class_probabilities.keys())
    values = [float(class_probabilities[k]) for k in labels]

    if not labels:
        labels = [predicted_label]
        values = [1.0]

    colors = ["#2b6cb0" if label != predicted_label else "#dd6b20" for label in labels]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Class probabilities")
    ax.set_ylabel("Probability")
    ax.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _plot_heuristics(
    window_arrays: Dict[str, np.ndarray],
    energy_threshold: float,
    band_threshold: float,
    output_path: Path,
    dpi: int,
) -> None:
    if window_arrays["time_center"].size == 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No windows available", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)
        return

    t = window_arrays["time_center"]
    energy = window_arrays["energy"]
    band_energy = window_arrays["band_energy"]
    peaks = window_arrays["num_spectral_peaks"]
    score = window_arrays["score"]
    selected_mask = window_arrays["is_selected"]
    passed_mask = window_arrays["passed_thresholds"]

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)

    axes[0].plot(t, energy, label="Energy", color="#2b6cb0", linewidth=1.2)
    axes[0].plot(t, band_energy, label="Band energy", color="#2f855a", linewidth=1.2)
    axes[0].axhline(energy_threshold, linestyle="--", color="#2b6cb0", alpha=0.6, label="Energy threshold")
    axes[0].axhline(band_threshold, linestyle="--", color="#2f855a", alpha=0.6, label="Band threshold")
    if np.any(passed_mask):
        axes[0].scatter(
            t[passed_mask],
            energy[passed_mask],
            color="#dd6b20",
            s=20,
            label="Passed thresholds",
            zorder=3,
        )
    axes[0].set_title("Heuristic energies")
    axes[0].set_ylabel("Value")
    axes[0].grid(alpha=0.2)
    axes[0].legend(loc="upper right")

    axes[1].plot(t, peaks, label="Spectral peaks", color="#6b46c1", linewidth=1.1)
    axes[1].plot(t, score, label="Window score", color="#d69e2e", linewidth=1.1)
    if np.any(selected_mask):
        axes[1].scatter(
            t[selected_mask],
            score[selected_mask],
            color="#dd6b20",
            s=35,
            label="Selected window",
            zorder=4,
        )
    axes[1].set_title("Peaks and score")
    axes[1].set_xlabel("Time, sec")
    axes[1].set_ylabel("Value")
    axes[1].grid(alpha=0.2)
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _write_markdown_report(
    output_path: Path,
    result: Dict[str, Any],
    class_probs: Dict[str, float],
    figure_paths: Dict[str, str],
) -> None:
    decision = str(result.get("decision", "unknown"))
    diagnostics = dict(result.get("diagnostics", {}))
    guidance_json = json.dumps(result.get("guidance_message", {}), ensure_ascii=False, indent=2)
    localization_json = json.dumps(result.get("localization_message", {}), ensure_ascii=False, indent=2)

    lines: List[str] = []
    lines.append("# Презентационный отчет по инференсу")
    lines.append("")
    lines.append(f"- Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Входной файл: `{result.get('audio_path', '')}`")
    lines.append("")
    lines.append("## Итог классификации")
    lines.append("")
    lines.append(f"- Решение: `{decision}`")
    lines.append(f"- Тип цели: `{result.get('target_type', 'unknown')}`")
    lines.append(f"- Сырой класс: `{result.get('raw_label_name', 'unknown')}` (id={result.get('raw_label_id', -1)})")
    lines.append(f"- Confidence: `{float(result.get('confidence', 0.0)):.4f}`")
    lines.append(f"- Signal level (RMS): `{float(result.get('signal_level', 0.0)):.6f}`")
    lines.append(
        f"- Выбранный интервал: `{float(result.get('time_start', 0.0)):.3f} .. {float(result.get('time_end', 0.0)):.3f} sec`"
    )
    lines.append(f"- Количество информативных окон: `{int(result.get('selected_windows', 0))}`")
    if diagnostics:
        lines.append(
            f"- Эвристический guess (без ML): `{diagnostics.get('heuristic_guess_label', 'unknown')}` "
            f"({diagnostics.get('heuristic_guess_reason', 'unknown')})"
        )
    lines.append("")

    lines.append("## Вероятности классов")
    lines.append("")
    if class_probs:
        lines.append("| Класс | Вероятность |")
        lines.append("|---|---:|")
        for name, prob in class_probs.items():
            lines.append(f"| `{name}` | `{prob:.4f}` |")
    else:
        lines.append("- Вероятности классов недоступны для текущей модели.")
    lines.append("")

    lines.append("## Сообщение для системы наведения")
    lines.append("")
    lines.append("```json")
    lines.append(guidance_json)
    lines.append("```")
    lines.append("")

    lines.append("## Сообщение для подсистемы локализации")
    lines.append("")
    lines.append("```json")
    lines.append(localization_json)
    lines.append("```")
    lines.append("")

    lines.append("## Визуализации")
    lines.append("")
    lines.append(f"### 1) Форма сигнала и выбранное окно\n![Waveform]({figure_paths['waveform']})")
    lines.append("")
    lines.append(f"### 2) Спектрограмма и выбранное окно\n![Spectrogram]({figure_paths['spectrogram']})")
    lines.append("")
    lines.append(f"### 3) Вероятности классов\n![Probabilities]({figure_paths['probabilities']})")
    lines.append("")
    lines.append(f"### 4) Эвристики по окнам\n![Heuristics]({figure_paths['heuristics']})")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    audio_path = Path(args.audio).resolve()
    cfg = load_yaml(args.config)

    pipeline = UAVClassificationPipeline.from_paths(
        model_path=args.model,
        scaler_path=args.scaler,
        label_map_path=args.label_map,
        config=cfg,
    )
    result = pipeline.infer_file(audio_path, include_diagnostics=True)
    diagnostics = dict(result.get("diagnostics", {}))
    heuristics = dict(diagnostics.get("heuristic", {}))

    run_dir = _resolve_run_dir(args.output_dir, audio_path=audio_path, run_name=args.run_name)
    figures_dir = ensure_dir(run_dir / "figures")

    preprocessing_cfg = dict(cfg.get("preprocessing", {}))
    target_sr = preprocessing_cfg.get("target_sample_rate", 22050)
    mono = bool(preprocessing_cfg.get("mono", True))
    normalize = bool(preprocessing_cfg.get("normalize", True))

    raw_audio, sr = load_audio(
        path=audio_path,
        target_sr=int(target_sr) if target_sr is not None else None,
        mono=mono,
        normalize=normalize,
    )
    processed_audio = preprocess_audio(raw_audio, sr=sr, cfg=preprocessing_cfg)
    selected_bounds = _selected_time_bounds(result)

    waveform_path = figures_dir / "waveform_selected_window.png"
    spectrogram_path = figures_dir / "spectrogram_selected_window.png"
    probabilities_path = figures_dir / "class_probabilities.png"
    heuristics_path = figures_dir / "heuristics_timeline.png"

    _plot_waveform(
        raw_audio=raw_audio,
        processed_audio=processed_audio,
        sr=sr,
        selected_bounds=selected_bounds,
        output_path=waveform_path,
        dpi=args.dpi,
    )
    _plot_spectrogram(
        audio=processed_audio,
        sr=sr,
        selected_bounds=selected_bounds,
        output_path=spectrogram_path,
        dpi=args.dpi,
    )
    class_probs = dict(diagnostics.get("class_probabilities", {}))
    _plot_class_probabilities(
        class_probabilities=class_probs,
        predicted_label=str(result.get("raw_label_name", "unknown")),
        output_path=probabilities_path,
        dpi=args.dpi,
    )

    window_arrays = _extract_window_arrays(list(heuristics.get("window_details", [])))
    _plot_heuristics(
        window_arrays=window_arrays,
        energy_threshold=float(heuristics.get("energy_threshold", 0.0)),
        band_threshold=float(heuristics.get("band_threshold", 0.0)),
        output_path=heuristics_path,
        dpi=args.dpi,
    )

    result_path = run_dir / "inference_result.json"
    save_json(result_path, result)

    report_path = run_dir / "presentation_report.md"
    _write_markdown_report(
        output_path=report_path,
        result=result,
        class_probs=class_probs,
        figure_paths={
            "waveform": "figures/waveform_selected_window.png",
            "spectrogram": "figures/spectrogram_selected_window.png",
            "probabilities": "figures/class_probabilities.png",
            "heuristics": "figures/heuristics_timeline.png",
        },
    )

    print("Presentation inference completed.")
    print(f"Run directory: {run_dir}")
    print(f"JSON result: {result_path}")
    print(f"Markdown report: {report_path}")
    print(f"Figures: {figures_dir}")


if __name__ == "__main__":
    main()
