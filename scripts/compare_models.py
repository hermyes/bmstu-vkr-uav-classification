#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_utils import ensure_dir, load_yaml, save_json
from src.data.feature_dataset import build_feature_matrix_from_manifest
from src.data.manifest import load_label_map
from src.evaluation.metrics import compute_classification_metrics


@dataclass
class ModelSpec:
    name: str
    needs_scaling: bool
    param_grid: Dict[str, List[Any]]

    def build(self, params: Dict[str, Any], random_seed: int, n_jobs: int):
        if self.name == "logreg":
            return LogisticRegression(
                C=float(params["C"]),
                class_weight="balanced",
                max_iter=2500,
                multi_class="multinomial",
                solver="lbfgs",
                random_state=random_seed,
            )
        if self.name == "svm_linear":
            return SVC(
                C=float(params["C"]),
                kernel="linear",
                class_weight="balanced",
                probability=True,
                random_state=random_seed,
            )
        if self.name == "svm_rbf":
            return SVC(
                C=float(params["C"]),
                gamma=params["gamma"],
                kernel="rbf",
                class_weight="balanced",
                probability=True,
                random_state=random_seed,
            )
        if self.name == "random_forest":
            return RandomForestClassifier(
                n_estimators=int(params["n_estimators"]),
                max_depth=None if params["max_depth"] is None else int(params["max_depth"]),
                min_samples_leaf=int(params["min_samples_leaf"]),
                class_weight="balanced",
                random_state=random_seed,
                n_jobs=n_jobs,
            )
        if self.name == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=int(params["n_estimators"]),
                learning_rate=float(params["learning_rate"]),
                max_depth=int(params["max_depth"]),
                random_state=random_seed,
            )
        if self.name == "knn":
            return KNeighborsClassifier(
                n_neighbors=int(params["n_neighbors"]),
                weights=params["weights"],
                p=int(params["p"]),
            )
        raise ValueError(f"Unknown model spec: {self.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiple classical models for UAV audio features")
    parser.add_argument("--config", type=str, default="configs/svm_baseline.yaml", help="Config path")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset root")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit per split")
    parser.add_argument("--n-jobs", type=int, default=-1, help="n_jobs for models that support it")
    parser.add_argument(
        "--output-json",
        type=str,
        default="reports/metrics/model_comparison.json",
        help="Output path for comparison JSON",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="reports/experiments/model_comparison.md",
        help="Output path for comparison markdown report",
    )
    return parser.parse_args()


def resolve_dataset(cfg: Dict[str, Any], dataset_override: str | None) -> Tuple[Path, Path, Path, Path]:
    dataset_cfg = dict(cfg.get("dataset", {}))
    root_dir = Path(dataset_override or dataset_cfg.get("root_dir", "train_sounds/dataset_out"))
    train_manifest = root_dir / str(dataset_cfg.get("train_manifest", "train/manifest.json"))
    val_manifest = root_dir / str(dataset_cfg.get("val_manifest", "val/manifest.json"))
    test_manifest = root_dir / str(dataset_cfg.get("test_manifest", "test/manifest.json"))
    return root_dir, train_manifest, val_manifest, test_manifest


def iter_param_dicts(grid: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    keys = list(grid.keys())
    for values in itertools.product(*[grid[k] for k in keys]):
        yield {k: v for k, v in zip(keys, values)}


def train_select_model(
    spec: ModelSpec,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    random_seed: int,
    n_jobs: int,
) -> Dict[str, Any]:
    best: Dict[str, Any] = {"val_f1_macro": -1.0}
    candidate_results: List[Dict[str, Any]] = []

    for params in iter_param_dicts(spec.param_grid):
        scaler = StandardScaler() if spec.needs_scaling else None
        if scaler is not None:
            x_train_model = scaler.fit_transform(x_train)
            x_val_model = scaler.transform(x_val)
        else:
            x_train_model = x_train
            x_val_model = x_val

        model = spec.build(params=params, random_seed=random_seed, n_jobs=n_jobs)
        model.fit(x_train_model, y_train)
        val_pred = model.predict(x_val_model)
        val_f1_macro = float(f1_score(y_val, val_pred, average="macro"))

        record = {
            "params": params,
            "val_f1_macro": val_f1_macro,
        }
        candidate_results.append(record)

        if val_f1_macro > float(best["val_f1_macro"]):
            best = {
                "model": model,
                "scaler": scaler,
                "params": params,
                "val_f1_macro": val_f1_macro,
            }

    best["candidate_results"] = candidate_results
    return best


def evaluate_model(
    model,
    scaler,
    x: np.ndarray,
    y: np.ndarray,
    label_ids: List[int],
    label_names: List[str],
) -> Dict[str, Any]:
    x_eval = scaler.transform(x) if scaler is not None else x
    y_pred = model.predict(x_eval)
    metrics = compute_classification_metrics(
        y_true=y,
        y_pred=y_pred,
        label_ids=label_ids,
        label_names=label_names,
    )
    return metrics


def write_markdown_report(
    output_md: Path,
    dataset_root: Path,
    results: List[Dict[str, Any]],
    best_by_val: Dict[str, Any],
) -> None:
    lines: List[str] = []
    lines.append("# Сравнение моделей классификации")
    lines.append("")
    lines.append(f"Дата: 2026-05-11")
    lines.append(f"Датасет: `{dataset_root}`")
    lines.append("")
    lines.append("## Итоговая таблица")
    lines.append("")
    lines.append("| Модель | Val F1-macro | Test F1-macro | Test Accuracy | Время (с) | Параметры |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for row in sorted(results, key=lambda x: x["val"]["f1_macro"], reverse=True):
        lines.append(
            f"| `{row['name']}` | {row['val']['f1_macro']:.4f} | {row['test']['f1_macro']:.4f} | "
            f"{row['test']['accuracy']:.4f} | {row['fit_seconds']:.1f} | `{json.dumps(row['best_params'], ensure_ascii=False)}` |"
        )
    lines.append("")
    lines.append("## Лучшая модель по val")
    lines.append("")
    lines.append(
        f"- Модель: `{best_by_val['name']}`\n"
        f"- Val F1-macro: `{best_by_val['val']['f1_macro']:.4f}`\n"
        f"- Test F1-macro: `{best_by_val['test']['f1_macro']:.4f}`\n"
        f"- Test Accuracy: `{best_by_val['test']['accuracy']:.4f}`\n"
        f"- Путь к артефакту: `{best_by_val['model_path']}`"
    )
    lines.append("")
    lines.append("## Примечание")
    lines.append("")
    lines.append("- Модели сохраняются, чтобы не переобучать их повторно.")
    lines.append("- Отбор гиперпараметров выполнен по `val` (метрика `F1-macro`).")

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    preprocessing_cfg = dict(cfg.get("preprocessing", {}))
    features_cfg = dict(cfg.get("features", {}))
    training_cfg = dict(cfg.get("training", {}))
    random_seed = int(training_cfg.get("random_seed", 42))

    dataset_root, train_manifest, val_manifest, test_manifest = resolve_dataset(cfg, args.dataset)
    label_map = load_label_map(dataset_root / "label_map.json")
    labels_sorted = sorted(label_map.items(), key=lambda x: x[1])
    label_names = [name for name, _ in labels_sorted]
    label_ids = [idx for _, idx in labels_sorted]

    print("Extracting train features...")
    x_train, y_train, _, feature_names = build_feature_matrix_from_manifest(
        manifest_path=train_manifest,
        preprocessing_cfg=preprocessing_cfg,
        features_cfg=features_cfg,
        limit=args.limit,
    )
    print("Extracting val features...")
    x_val, y_val, _, _ = build_feature_matrix_from_manifest(
        manifest_path=val_manifest,
        preprocessing_cfg=preprocessing_cfg,
        features_cfg=features_cfg,
        limit=args.limit,
    )
    print("Extracting test features...")
    x_test, y_test, _, _ = build_feature_matrix_from_manifest(
        manifest_path=test_manifest,
        preprocessing_cfg=preprocessing_cfg,
        features_cfg=features_cfg,
        limit=args.limit,
    )

    model_specs: List[ModelSpec] = [
        ModelSpec(
            name="logreg",
            needs_scaling=True,
            param_grid={
                "C": [0.5, 1.0, 2.0, 5.0],
            },
        ),
        ModelSpec(
            name="svm_linear",
            needs_scaling=True,
            param_grid={
                "C": [0.5, 1.0, 2.0, 5.0],
            },
        ),
        ModelSpec(
            name="svm_rbf",
            needs_scaling=True,
            param_grid={
                "C": [1.0, 5.0, 10.0],
                "gamma": ["scale", 0.1, 0.01],
            },
        ),
        ModelSpec(
            name="random_forest",
            needs_scaling=False,
            param_grid={
                "n_estimators": [300],
                "max_depth": [None, 20],
                "min_samples_leaf": [1, 2],
            },
        ),
        ModelSpec(
            name="gradient_boosting",
            needs_scaling=False,
            param_grid={
                "n_estimators": [150, 250],
                "learning_rate": [0.05, 0.1],
                "max_depth": [2, 3],
            },
        ),
        ModelSpec(
            name="knn",
            needs_scaling=True,
            param_grid={
                "n_neighbors": [3, 5, 9],
                "weights": ["distance"],
                "p": [1, 2],
            },
        ),
    ]

    results: List[Dict[str, Any]] = []
    model_dir = ensure_dir("models/comparison")

    for spec in model_specs:
        print(f"\n=== Training model: {spec.name} ===")
        t0 = time.perf_counter()
        selected = train_select_model(
            spec=spec,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            random_seed=random_seed,
            n_jobs=args.n_jobs,
        )
        fit_seconds = time.perf_counter() - t0

        model = selected["model"]
        scaler = selected["scaler"]
        best_params = selected["params"]

        val_metrics = evaluate_model(
            model=model,
            scaler=scaler,
            x=x_val,
            y=y_val,
            label_ids=label_ids,
            label_names=label_names,
        )
        test_metrics = evaluate_model(
            model=model,
            scaler=scaler,
            x=x_test,
            y=y_test,
            label_ids=label_ids,
            label_names=label_names,
        )

        model_path = model_dir / f"{spec.name}.pkl"
        scaler_path = model_dir / f"{spec.name}_scaler.pkl"
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        result = {
            "name": spec.name,
            "best_params": best_params,
            "fit_seconds": fit_seconds,
            "val": {
                "accuracy": val_metrics["accuracy"],
                "f1_macro": val_metrics["f1_macro"],
                "f1_weighted": val_metrics["f1_weighted"],
            },
            "test": {
                "accuracy": test_metrics["accuracy"],
                "f1_macro": test_metrics["f1_macro"],
                "f1_weighted": test_metrics["f1_weighted"],
            },
            "candidate_results": selected["candidate_results"],
            "model_path": str(model_path),
            "scaler_path": str(scaler_path),
            "test_metrics_full": test_metrics,
            "val_metrics_full": val_metrics,
        }
        results.append(result)

        print(
            f"{spec.name}: val_f1_macro={val_metrics['f1_macro']:.4f}, "
            f"test_f1_macro={test_metrics['f1_macro']:.4f}, "
            f"test_acc={test_metrics['accuracy']:.4f}"
        )

    best_by_val = max(results, key=lambda x: x["val"]["f1_macro"])

    output_json = Path(args.output_json)
    payload = {
        "date": "2026-05-11",
        "dataset_root": str(dataset_root),
        "n_samples": {
            "train": int(len(y_train)),
            "val": int(len(y_val)),
            "test": int(len(y_test)),
        },
        "n_features": int(x_train.shape[1]),
        "feature_names_path": "models/feature_names.json",
        "results": results,
        "best_by_val": best_by_val["name"],
    }
    save_json(output_json, payload)
    write_markdown_report(
        output_md=Path(args.output_md),
        dataset_root=dataset_root,
        results=results,
        best_by_val=best_by_val,
    )

    print("\nComparison completed.")
    print(f"JSON report: {output_json}")
    print(f"Markdown report: {args.output_md}")
    print(f"Best by val: {best_by_val['name']} (val_f1_macro={best_by_val['val']['f1_macro']:.4f})")


if __name__ == "__main__":
    main()
