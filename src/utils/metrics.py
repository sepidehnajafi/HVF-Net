"""Evaluation metrics.

The paper reports balanced accuracy throughout (mean of sensitivity and
specificity), which is invariant to class imbalance -- important given the
25 HC vs. 12/13 PD split used in each task. See Section 3.9 / Table 2.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return 0.5 * (sensitivity + specificity)


def full_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Args:
        y_true: (N,) ground-truth binary labels.
        y_pred: (N,) predicted binary labels.
        y_prob: (N,) predicted probability of the positive (PD) class.

    Returns:
        Dict with accuracy, sensitivity, specificity, f1, auc.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity) > 0
        else 0.0
    )
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")  # undefined if only one class present in y_true

    return {
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1_score": f1,
        "auc": auc,
    }


def class_weights_inverse_frequency(labels: np.ndarray, num_classes: int = 2) -> np.ndarray:
    """Inverse class-frequency weights for a weighted cross-entropy loss."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0  # avoid division by zero for absent classes
    weights = counts.sum() / (num_classes * counts)
    return weights.astype(np.float32)
