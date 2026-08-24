#!/usr/bin/env python
"""Run the label-permutation test for one task (Section 3.9, Table 8).

Reuses the exact model construction and cross-validation code path as
``run_cv.py`` -- only the labels differ (randomly shuffled per permutation).

Usage:
    python scripts/run_permutation_test.py \
        --features data/features_off_vs_hc.npz \
        --electrode-coords data/electrode_coords.npy \
        --config configs/default_config.yaml \
        --real-accuracy 0.9725 \
        --num-permutations 200 \
        --out outputs/permutation_test_off.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from scripts.run_cv import build_propagation_matrix, load_dataset
from src.training.permutation_test import run_permutation_test
from src.training.train import TrainConfig
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--electrode-coords", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/default_config.yaml"))
    parser.add_argument(
        "--real-accuracy",
        type=float,
        required=True,
        help="Cross-validated balanced accuracy on the true (unpermuted) labels, "
        "obtained beforehand via run_cv.py.",
    )
    parser.add_argument("--num-permutations", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path("outputs/permutation_test.json"))
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
        # NOTE: patience is intentionally set to 10 here (vs. 8 for the main
        # model) to reduce sensitivity to noisy validation curves under
        # permuted labels -- see Section 3.9 of the paper.
        early_stopping_patience=10,
        class_weighting=cfg["training"]["class_weighting"],
        validation_split=cfg["training"]["validation_split"],
        temperature=cfg["contrastive"]["temperature"],
        lambda_max=cfg["contrastive"]["lambda_max"],
        focusing_gamma=cfg["contrastive"]["focusing_exponent"],
    )

    results = run_permutation_test(
        dataset,
        propagation,
        model_kwargs,
        train_cfg,
        real_accuracy=args.real_accuracy,
        num_permutations=args.num_permutations,
        num_folds=cfg["evaluation"]["num_folds"],
        base_seed=cfg["experiment"]["seed"],
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nReal accuracy: {results['real_accuracy']*100:.2f}%")
    print(f"Null distribution: {results['mean']*100:.2f}% +/- {results['std']*100:.2f}% "
          f"(range {results['min']*100:.2f}-{results['max']*100:.2f}%)")
    print(f"p-value: {results['p_value']:.6f}")


if __name__ == "__main__":
    main()
