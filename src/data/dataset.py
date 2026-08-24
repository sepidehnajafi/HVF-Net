"""PyTorch Dataset and subject-independent cross-validation utilities.

Ensures no subject's epochs ever appear in both the train and test partition
of any fold (Section 3.2), which is the central methodological safeguard
against data leakage in this work.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset


@dataclass
class Sample:
    connectivity: np.ndarray   # (C, C, B) volumetric PLV tensor
    spectral: np.ndarray       # (C, B) log band-power matrix
    label: int                 # 1 = PD, 0 = HC
    subject_id: str


class EEGEpochDataset(Dataset):
    """Wraps a flat list of precomputed epoch-level features.

    Precomputing connectivity/spectral features offline (see
    ``scripts/precompute_features.py``) is strongly recommended: PLV
    extraction is the dominant cost in the pipeline and should not be
    repeated every epoch during training.
    """

    def __init__(self, samples: list[Sample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return {
            "connectivity": torch.from_numpy(s.connectivity).float(),
            "spectral": torch.from_numpy(s.spectral).float(),
            "label": torch.tensor(s.label, dtype=torch.long),
            "subject_id": s.subject_id,
        }


def subject_level_labels(samples: list[Sample]) -> tuple[list[str], list[int]]:
    """One (subject_id, majority label) pair per unique subject, used to
    stratify the subject-level K-fold split.
    """
    by_subject: dict[str, list[int]] = {}
    for s in samples:
        by_subject.setdefault(s.subject_id, []).append(s.label)
    subject_ids = sorted(by_subject)
    labels = [int(np.round(np.mean(by_subject[sid]))) for sid in subject_ids]
    return subject_ids, labels


def subject_independent_kfold_indices(
    samples: list[Sample],
    num_folds: int = 10,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate subject-independent, stratified K-fold splits.

    The split is performed on *subjects*, then expanded back to epoch-level
    indices, guaranteeing every epoch from a given subject falls entirely
    within either the train or the test partition of a fold.

    Args:
        samples: flat list of :class:`Sample`.
        num_folds: K.
        seed: random seed for the fold assignment.

    Returns:
        List of ``(train_idx, test_idx)`` epoch-level index arrays, one tuple
        per fold.
    """
    subject_ids, subject_labels = subject_level_labels(samples)
    subject_ids = np.array(subject_ids)
    subject_labels = np.array(subject_labels)

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

    subject_to_epoch_idx: dict[str, list[int]] = {}
    for i, s in enumerate(samples):
        subject_to_epoch_idx.setdefault(s.subject_id, []).append(i)

    folds = []
    for train_subj_idx, test_subj_idx in skf.split(subject_ids, subject_labels):
        train_subjects = subject_ids[train_subj_idx]
        test_subjects = subject_ids[test_subj_idx]

        train_epochs = np.concatenate(
            [subject_to_epoch_idx[sid] for sid in train_subjects]
        )
        test_epochs = np.concatenate(
            [subject_to_epoch_idx[sid] for sid in test_subjects]
        )
        folds.append((train_epochs, test_epochs))

    return folds


def shuffle_labels_stratified(
    samples: list[Sample],
    seed: int,
) -> list[Sample]:
    """Return a copy of ``samples`` with subject-level labels randomly
    permuted, preserving the original class ratio (used by the
    label-permutation test, Section 3.9).
    """
    rng = np.random.default_rng(seed)
    subject_ids, subject_labels = subject_level_labels(samples)
    subject_labels = np.array(subject_labels)
    permuted = rng.permutation(subject_labels)
    label_map = dict(zip(subject_ids, permuted))

    shuffled = []
    for s in samples:
        shuffled.append(
            Sample(
                connectivity=s.connectivity,
                spectral=s.spectral,
                label=int(label_map[s.subject_id]),
                subject_id=s.subject_id,
            )
        )
    return shuffled
