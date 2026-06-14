from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _fit_best_svc(
    x_train: np.ndarray,
    y_train: np.ndarray,
    model_cfg: Dict[str, Any],
    random_seed: int,
    n_jobs: int,
) -> Tuple[SVC, Dict[str, Any]]:
    class_weight = model_cfg.get("class_weight", "balanced")
    gs_cfg = dict(model_cfg.get("grid_search", {}))
    cv = int(gs_cfg.get("cv", 3))
    scoring = str(gs_cfg.get("scoring", "f1_macro"))
    param_grid = dict(gs_cfg.get("param_grid", {}))
    if not param_grid:
        param_grid = {"C": [1.0], "gamma": ["scale"], "kernel": ["rbf"]}

    svc = SVC(
        class_weight=class_weight,
        probability=False,
        random_state=random_seed,
    )
    grid = GridSearchCV(
        estimator=svc,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=1,
        refit=True,
    )
    grid.fit(x_train, y_train)

    return grid.best_estimator_, {
        "best_params": grid.best_params_,
        "best_score": float(grid.best_score_),
        "cv": cv,
        "scoring": scoring,
    }


def train_svm_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    model_cfg: Dict[str, Any],
    random_seed: int = 42,
    n_jobs: int = -1,
):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    best_svc, grid_summary = _fit_best_svc(
        x_train=x_train_scaled,
        y_train=y_train,
        model_cfg=model_cfg,
        random_seed=random_seed,
        n_jobs=n_jobs,
    )

    calibrate = bool(model_cfg.get("calibrate_probabilities", True))
    if calibrate:
        calibration_cv = int(model_cfg.get("calibration_cv", 3))
        classifier = CalibratedClassifierCV(best_svc, cv=calibration_cv)
        classifier.fit(x_train_scaled, y_train)
    else:
        svc_params = best_svc.get_params()
        svc_params["probability"] = True
        svc_params["random_state"] = random_seed
        svc_with_proba = SVC(**svc_params)
        svc_with_proba.fit(x_train_scaled, y_train)
        classifier = svc_with_proba

    training_summary = {
        **grid_summary,
        "calibrate_probabilities": calibrate,
    }
    return classifier, scaler, training_summary


def predict_with_probabilities(classifier, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_pred = classifier.predict(x)

    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba(x)
        return y_pred, probs

    # Fallback for models without calibrated probabilities.
    if hasattr(classifier, "decision_function"):
        scores = classifier.decision_function(x)
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return y_pred, probs

    classes = getattr(classifier, "classes_", None)
    if classes is None:
        classes = np.unique(y_pred)
    classes = np.asarray(classes)
    probs = np.zeros((len(y_pred), len(classes)), dtype=np.float32)
    class_to_idx = {int(c): i for i, c in enumerate(classes)}
    for row, pred in enumerate(y_pred):
        idx = class_to_idx.get(int(pred))
        if idx is not None:
            probs[row, idx] = 1.0
    return y_pred, probs


def predict_with_confidence(classifier, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_pred, probs = predict_with_probabilities(classifier, x)
    confidence = np.max(probs, axis=1) if probs.size else np.ones(len(y_pred), dtype=np.float32)
    return y_pred, confidence
