"""Label-permutation test (Section 3.9).

For each of ``num_permutations`` iterations, subject-level labels are
randomly shuffled (stratified to preserve the original class ratio) and the
*entire* subject-independent 10-fold cross-validation protocol is re-run
from randomly initialized weights -- i.e. this calls exactly the same
:func:`src.training.evaluate.run_kfold_cv` used for the real result, just on
shuffled labels. This is intentionally expensive (``num_permutations x
num_folds`` full training runs) because it is the only way to guarantee the
resulting null distribution reflects the real training protocol rather than
a cheaper approximation.
"""

from __future__ import annotations

import numpy as np
import torch

from src.data.dataset import EEGEpochDataset, shuffle_labels_stratified
from src.training.evaluate import run_kfold_cv
from src.training.train import TrainConfig


def run_permutation_test(
    dataset: EEGEpochDataset,
    propagation: torch.Tensor,
    model_kwargs: dict,
    train_cfg: TrainConfig,
    real_accuracy: float,
    num_permutations: int = 200,
    num_folds: int = 10,
    base_seed: int = 42,
) -> dict:
    """
    Args:
        real_accuracy: the true (unpermuted) cross-validated balanced
            accuracy for this task, obtained beforehand via
            :func:`run_kfold_cv` on the un-shuffled dataset.
        num_permutations: N in the p-value formula below.

    Returns:
        Dict with ``null_accuracies`` (list, one mean-CV-accuracy per
        permutation), ``mean``, ``std``, ``min``, ``max``, and ``p_value``,
        where::

            p_value = (count(null_accuracies >= real_accuracy) + 1) / (N + 1)
    """
    null_accuracies = []
    for perm_idx in range(num_permutations):
        perm_seed = base_seed + 1000 + perm_idx  # disjoint from CV fold seeds
        shuffled_samples = shuffle_labels_stratified(dataset.samples, seed=perm_seed)
        shuffled_dataset = EEGEpochDataset(shuffled_samples)

        cv_result = run_kfold_cv(
            shuffled_dataset,
            propagation,
            model_kwargs,
            train_cfg,
            num_folds=num_folds,
            seed=perm_seed,
        )
        null_accuracies.append(cv_result["balanced_accuracy"]["mean"])
        print(f"[permutation {perm_idx + 1}/{num_permutations}] "
              f"null_accuracy={null_accuracies[-1]:.4f}")

    null_accuracies = np.array(null_accuracies)
    count_ge_real = int(np.sum(null_accuracies >= real_accuracy))
    p_value = (count_ge_real + 1) / (num_permutations + 1)

    return {
        "null_accuracies": null_accuracies.tolist(),
        "mean": float(null_accuracies.mean()),
        "std": float(null_accuracies.std()),
        "min": float(null_accuracies.min()),
        "max": float(null_accuracies.max()),
        "real_accuracy": real_accuracy,
        "permutations_ge_real": count_ge_real,
        "num_permutations": num_permutations,
        "p_value": p_value,
    }
