"""Metrics with a fixed, complete class list and explicit zero-division handling."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)


def classification_metrics(y_true, y_pred, n_classes: int) -> dict:
    labels = list(range(n_classes))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = np.nan_to_num(cm / cm.sum(1, keepdims=True))
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class": [{"class_id": c, "precision": float(p[c]), "recall": float(r[c]), "f1": float(f[c]),
                       "support": int(s[c])} for c in labels],
        "confusion": cm.tolist(), "confusion_row_normalized": cm_norm.round(4).tolist(),
    }
