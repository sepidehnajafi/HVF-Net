#!/usr/bin/env python
"""Run subject-independent 10-fold cross-validation for one task
(e.g. PD-OFF vs. HC) and report mean +/- std accuracy, matching Tables 3-4.

Usage:
    python scripts/run_cv.py \
        --features data/features_off_vs_hc.npz \
        --electrode-coords data/electrode_coords.npy \
        --config configs/default_config.yaml \
        --out outputs/cv_results_off.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.dataset import EEGEpochDataset, Sample
from src.data.hypergraph import build_incidence_matrix, hyperedge_weights, hypergraph_laplacian
from src.training.evaluate import run_kfold_cv
from src.training.train import TrainConfig
from src.utils.seed import set_seed


def load_dataset(features_path: Path) -> EEGEpochDataset:
    npz = np.load(features_path, allow_pickle=True)
    samples = [
        Sample(
            connectivity=npz["connectivity"][i],
            spectral=npz["spectral"][i],
            label=int(npz["labels"][i]),
            subject_id=str(npz["subject_ids"][i]),
        )
        for i in range(len(npz["labels"]))
    ]
    return EEGEpochDataset(samples)


def build_propagation_matrix(electrode_coords_path: Path, k_neighbors: int) -> torch.Tensor:
    coords = np.load(electrode_coords_path)
    H = build_incidence_matrix(coords, k_neighbors=k_neighbors)
    w = hyperedge_weights(H)
    d_v = H @ w
    d_e = H.sum(axis=0)
    D_v_inv_sqrt = np.diag(1.0 / np.sqrt(d_v + 1e-8))
    D_e_inv = np.diag(1.0 / (d_e + 1e-8))
    W = np.diag(w)
    propagation = D_v_inv_sqrt @ H @ W @ D_e_inv @ H.T @ D_v_inv_sqrt
    return torch.from_numpy(propagation.astype(np.float32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--electrode-coords", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/default_config.yaml"))
    parser.add_argument("--out", type=Path, default=Path("outputs/cv_results.json"))
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    set_seed(cfg["experiment"]["seed"])

    dataset = load_dataset(args.features)
    propagation = build_propagation_matrix(
        args.electrode_coords, cfg["hypergraph"]["k_neighbors"]
    )

    model_kwargs = dict(
        num_electrodes=cfg["data"]["num_channels"],
        num_bands=len(cfg["data"]["bands"]),
        hypergraph_hidden_dim=cfg["hypergraph"]["node_embedding_dim"],
        hypergraph_layers=cfg["hypergraph"]["num_conv_layers"],
        cnn_channel_progression=tuple(cfg["cnn3d"]["channel_progression"]),
        spectral_token_dim=cfg["spectral_stream"]["token_dim"],
        spectral_projection_dim=cfg["spectral_stream"]["projection_dim"],
        transformer_d_model=cfg["transformer"]["d_model"],
        transformer_heads=cfg["transformer"]["num_heads"],
        transformer_layers=cfg["transformer"]["num_layers"],
        transformer_ffn_dim=cfg["transformer"]["ffn_dim"],
        transformer_dropout=cfg["transformer"]["dropout"],
        bcmi_d_k=cfg["bcmi"]["d_k"],
        classifier_dropout=cfg["regularization"]["dropout_head"],
    )

    train_cfg = TrainConfig(
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        batch_size=cfg["training"]["batch_size"],
        max_epochs=cfg["training"]["max_epochs"],
        early_stopping_patience=cfg["training"]["early_stopping_patience"],
        class_weighting=cfg["training"]["class_weighting"],
        validation_split=cfg["training"]["validation_split"],
        temperature=cfg["contrastive"]["temperature"],
        lambda_max=cfg["contrastive"]["lambda_max"],
        focusing_gamma=cfg["contrastive"]["focusing_exponent"],
    )

    results = run_kfold_cv(
        dataset,
        propagation,
        model_kwargs,
        train_cfg,
        num_folds=cfg["evaluation"]["num_folds"],
        seed=cfg["experiment"]["seed"],
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    bacc = results["balanced_accuracy"]
    print(f"\nBalanced accuracy: {bacc['mean']*100:.2f}% +/- {bacc['std']*100:.2f}%")
    print(f"Full results written to {args.out}")


if __name__ == "__main__":
    main()
