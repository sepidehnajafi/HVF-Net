#!/usr/bin/env python
"""Precompute per-epoch connectivity + spectral features from raw EEG.

Reads preprocessed, artifact-rejected epochs (one .npy/.fif set per subject)
and writes a single features file consumed by ``run_cv.py`` and
``run_permutation_test.py``. Precomputing avoids repeating the relatively
expensive PLV extraction on every training epoch.

Expected input layout:

    data/raw/<subject_id>/epochs.npy   # (num_epochs, num_channels, samples)
    data/raw/<subject_id>/label.txt    # single line: "0" (HC) or "1" (PD)

Usage:
    python scripts/precompute_features.py \
        --raw-dir data/raw --out data/features.npz --config configs/default_config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from src.data.connectivity import batch_volumetric_connectivity
from src.data.spectral_features import spectral_feature_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/default_config.yaml"))
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    bands = {name: tuple(edges) for name, edges in cfg["data"]["bands"].items()}
    fs = cfg["data"]["sampling_rate_hz"]

    connectivity_list, spectral_list, labels, subject_ids = [], [], [], []

    subject_dirs = sorted(p for p in args.raw_dir.iterdir() if p.is_dir())
    for subject_dir in tqdm(subject_dirs, desc="subjects"):
        epochs = np.load(subject_dir / "epochs.npy")  # (num_epochs, C, T)
        label = int((subject_dir / "label.txt").read_text().strip())

        conn = batch_volumetric_connectivity(epochs, fs, bands)  # (N, C, C, B)
        spec = np.stack(
            [spectral_feature_matrix(e, fs, bands) for e in epochs], axis=0
        )  # (N, C, B)

        connectivity_list.append(conn)
        spectral_list.append(spec)
        labels.extend([label] * len(epochs))
        subject_ids.extend([subject_dir.name] * len(epochs))

    np.savez_compressed(
        args.out,
        connectivity=np.concatenate(connectivity_list, axis=0),
        spectral=np.concatenate(spectral_list, axis=0),
        labels=np.array(labels),
        subject_ids=np.array(subject_ids),
    )
    print(f"Wrote {len(labels)} epochs from {len(subject_dirs)} subjects to {args.out}")


if __name__ == "__main__":
    main()
