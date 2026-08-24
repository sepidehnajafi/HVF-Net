"""Subject-independent K-fold cross-validation driver.

Runs :func:`src.training.train.train_one_fold` once per fold and aggregates
mean +/- std across folds, matching the reporting convention used throughout
the paper (Tables 3-4).
"""

from __future__ import annotations

import numpy as np
import torch

from src.data.dataset import EEGEpochDataset, subject_independent_kfold_indices
from src.training.train import TrainConfig, train_one_fold


def run_kfold_cv(
    dataset: EEGEpochDataset,
    propagation: torch.Tensor,
    model_kwargs: dict,
    train_cfg: TrainConfig,
    num_folds: int = 10,
    seed: int = 42,
) -> dict:
    """
    Returns:
        Dict with per-fold metric lists and their mean/std summary, e.g.
        ``{"balanced_accuracy": {"mean": ..., "std": ..., "per_fold": [...]}}``.
    """
    folds = subject_independent_kfold_indices(dataset.samples, num_folds=num_folds, seed=seed)

    per_fold_metrics: list[dict] = []
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        fold_seed = seed + fold_idx
        result = train_one_fold(
            dataset, train_idx, test_idx, propagation, model_kwargs, train_cfg, fold_seed
        )
        per_fold_metrics.append(result["metrics"])
        print(
            f"[fold {fold_idx + 1}/{num_folds}] "
            f"balanced_accuracy={result['metrics']['balanced_accuracy']:.4f} "
            f"(best_epoch={result['best_epoch']})"
        )

    return _aggregate(per_fold_metrics)


def _aggregate(per_fold_metrics: list[dict]) -> dict:
    keys = per_fold_metrics[0].keys()
    summary = {}
    for k in keys:
        values = np.array([m[k] for m in per_fold_metrics])
        summary[k] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)),
            "per_fold": values.tolist(),
        }
    return summary
