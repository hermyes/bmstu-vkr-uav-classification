from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.config_utils import save_json


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    label_ids: Sequence[int],
    label_names: Sequence[str],
) -> Dict[str, object]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    accuracy = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(label_ids))
    report = classification_report(
        y_true,
        y_pred,
        labels=list(label_ids),
        target_names=list(label_names),
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(p_weighted),
        "recall_weighted": float(r_weighted),
        "f1_weighted": float(f1_weighted),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "label_ids": [int(x) for x in label_ids],
        "label_names": list(label_names),
    }


def save_metrics(metrics: Dict[str, object], output_path: str | Path) -> None:
    save_json(output_path, metrics)
