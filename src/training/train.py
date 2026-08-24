"""Single-fold training loop with early stopping.

Used both by the main 10-fold cross-validation run (``scripts/run_cv.py``)
and by the label-permutation test (``scripts/run_permutation_test.py``),
which calls this exact same function on shuffled labels to guarantee an
apples-to-apples null distribution (Section 3.9).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data.dataset import EEGEpochDataset
from src.models.losses import TotalLoss
from src.models.hvf_net import HVFNet
from src.utils.metrics import class_weights_inverse_frequency, full_metrics


@dataclass
class TrainConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 30
    early_stopping_patience: int = 8
    class_weighting: bool = True
    validation_split: float = 0.2
    temperature: float = 0.07
    lambda_max: float = 4.0
    focusing_gamma: float = 2.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _split_train_val(train_idx: np.ndarray, labels: np.ndarray, val_frac: float, seed: int):
    """Stratified train/validation split *within* the training partition of
    a fold, so the test partition remains fully untouched during model
    selection / early stopping.
    """
    rng = np.random.default_rng(seed)
    val_idx, tr_idx = [], []
    for c in np.unique(labels[train_idx]):
        class_idx = train_idx[labels[train_idx] == c]
        rng.shuffle(class_idx)
        n_val = max(1, int(round(len(class_idx) * val_frac)))
        val_idx.extend(class_idx[:n_val])
        tr_idx.extend(class_idx[n_val:])
    return np.array(tr_idx), np.array(val_idx)


def train_one_fold(
    dataset: EEGEpochDataset,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    propagation: torch.Tensor,
    model_kwargs: dict,
    train_cfg: TrainConfig,
    seed: int,
) -> dict:
    """Train HVF-Net from randomly initialized weights on one fold and
    evaluate on the held-out test partition.

    Args:
        dataset: full :class:`EEGEpochDataset`.
        train_idx, test_idx: epoch-level indices for this fold (subject-
            independent; see ``src/data/dataset.py``).
        propagation: precomputed (C, C) hypergraph propagation matrix.
        model_kwargs: kwargs forwarded to ``HVFNet``.
        train_cfg: :class:`TrainConfig`.
        seed: seed for this fold's training run (weight init, val split,
            data loader shuffling).

    Returns:
        Dict with ``metrics`` (see :func:`src.utils.metrics.full_metrics`)
        and the ``best_epoch`` at which early stopping selected the model.
    """
    torch.manual_seed(seed)
    device = torch.device(train_cfg.device)
    propagation = propagation.to(device)

    all_labels = np.array([dataset.samples[i].label for i in range(len(dataset))])
    inner_train_idx, val_idx = _split_train_val(
        train_idx, all_labels, train_cfg.validation_split, seed
    )

    train_loader = DataLoader(
        Subset(dataset, inner_train_idx),
        batch_size=train_cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=train_cfg.batch_size)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=train_cfg.batch_size)

    model = HVFNet(**model_kwargs).to(device)

    class_weights = None
    if train_cfg.class_weighting:
        class_weights = torch.tensor(
            class_weights_inverse_frequency(all_labels[inner_train_idx]),
            device=device,
        )

    criterion = TotalLoss(
        temperature=train_cfg.temperature,
        lambda_max=train_cfg.lambda_max,
        gamma=train_cfg.focusing_gamma,
        class_weights=class_weights,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg.max_epochs)

    best_val_bacc = -1.0
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(train_cfg.max_epochs):
        model.train()
        for batch in train_loader:
            volumetric = batch["connectivity"].to(device)
            spectral = batch["spectral"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits, embedding = model(volumetric, spectral, propagation, return_embedding=True)
            loss_dict = criterion(logits, embedding, labels)
            loss_dict["total"].backward()
            optimizer.step()
        scheduler.step()

        val_bacc = _evaluate(model, val_loader, propagation, device)["balanced_accuracy"]
        if val_bacc > best_val_bacc:
            best_val_bacc = val_bacc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= train_cfg.early_stopping_patience:
                break

    model.load_state_dict(best_state)
    test_metrics = _evaluate(model, test_loader, propagation, device)
    return {"metrics": test_metrics, "best_epoch": best_epoch}


@torch.no_grad()
def _evaluate(model, loader, propagation, device) -> dict:
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    for batch in loader:
        volumetric = batch["connectivity"].to(device)
        spectral = batch["spectral"].to(device)
        labels = batch["label"]

        logits = model(volumetric, spectral, propagation)
        probs = torch.softmax(logits, dim=-1)[:, 1]
        preds = logits.argmax(dim=-1)

        all_labels.append(labels.numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    return full_metrics(y_true, y_pred, y_prob)
