# HVF-Net

**HVF-Net: A Hypergraph-Based Volumetric Fusion Network for EEG-Based Parkinson's Disease Classification**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0.1-ee4c2c)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository provides the official PyTorch implementation of HVF-Net, a unified architecture that
jointly models volumetric dynamic functional connectivity, neuro-inspired
hypergraph representations of multi-electrode interactions, 3D convolutional
feature extraction, and bidirectional cross-modal fusion for task-evoked EEG
classification of Parkinson's disease.

**Author:** Sepideh Najafi ([sepidenajafi.irfa@gmail.com](mailto:sepidenajafi.irfa@gmail.com))

> 📄 Paper: *Manuscript submitted for publication (2026). DOI/link to be added upon acceptance.*

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
  - [1. Precompute features](#1-precompute-features)
  - [2. Run cross-validation](#2-run-cross-validation)
  - [3. Run the label-permutation test](#3-run-the-label-permutation-test)
- [Results](#results)
- [Reproducibility Notes](#reproducibility-notes)
- [Citation](#citation)
- [License](#license)

## Overview

HVF-Net processes each EEG epoch along two parallel pathways that are
fused by a learnable, bidirectional gated cross-attention module:

<p align="center">
  <img src="assets/architecture.png" alt="HVF-Net architecture" width="850">
</p>

See Section 3.7 of the paper for the full mathematical formulation
(Equations 1-23).

## Repository Structure

```
hvf-net/
├── configs/
│   └── default_config.yaml       # all hyperparameters (mirrors Table 2)
├── src/
│   ├── data/
│   │   ├── preprocessing.py      # filtering, epoching
│   │   ├── connectivity.py       # PLV -> volumetric dFC tensor
│   │   ├── spectral_features.py  # per-electrode log band power
│   │   ├── hypergraph.py         # K-NN hyperedges, incidence matrix, Laplacian
│   │   ├── nisp.py               # Neuro-Inspired Spectral Perturbation aug.
│   │   └── dataset.py            # subject-independent K-fold splitting
│   ├── models/
│   │   ├── hypergraph_conv.py    # hypergraph convolution layers
│   │   ├── cnn3d.py              # 3D CNN backbone
│   │   ├── spectral_transformer.py
│   │   ├── bcmi.py               # bidirectional cross-modal interaction
│   │   ├── losses.py             # adaptive supervised contrastive loss
│   │   └── hvf_net.py        # full model assembly
│   ├── training/
│   │   ├── train.py              # single-fold training loop
│   │   ├── evaluate.py           # 10-fold CV driver
│   │   └── permutation_test.py   # label-permutation significance test
│   └── utils/
│       ├── seed.py
│       └── metrics.py            # balanced accuracy, sensitivity, etc.
├── scripts/
│   ├── precompute_features.py    # raw EEG -> connectivity/spectral features
│   ├── run_cv.py                 # entry point: reproduce Tables 3-4
│   └── run_permutation_test.py   # entry point: reproduce Table 8
├── tests/
│   └── test_shapes.py            # shape/gradient-flow smoke tests
├── requirements.txt
└── LICENSE
```

## Installation

```bash
git clone https://github.com/sepidehnajafi/HVF-Net.git
cd hvf-net
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Verify the installation:

```bash
pytest tests/ -v
```

## Data Preparation

This work uses a subset of the **PRED+CT** data warehouse (auditory oddball
paradigm), comprising 50 participants (25 idiopathic PD, 25 age- and
sex-matched healthy controls), with PD recordings collected in both
medicated (ON) and unmedicated (OFF) states. The dataset itself is not
included in this repository (see PRED+CT's own access terms). Users should
prepare their own data following the preprocessing protocol described in
the paper (Section 3.3). Once available, organize raw, artifact-rejected
epochs as:

```
data/raw/<subject_id>/epochs.npy   # (num_epochs, num_channels, samples), float32
data/raw/<subject_id>/label.txt    # single line: "0" (HC) or "1" (PD)
data/electrode_coords.npy          # (num_channels, 3) scalp 3-D coordinates
```

## Usage

### 1. Precompute features

```bash
python scripts/precompute_features.py \
    --raw-dir data/raw \
    --out data/features_off_vs_hc.npz \
    --config configs/default_config.yaml
```

### 2. Run cross-validation

```bash
python scripts/run_cv.py \
    --features data/features_off_vs_hc.npz \
    --electrode-coords data/electrode_coords.npy \
    --config configs/default_config.yaml \
    --out outputs/cv_results_off.json
```

### 3. Run the label-permutation test

```bash
python scripts/run_permutation_test.py \
    --features data/features_off_vs_hc.npz \
    --electrode-coords data/electrode_coords.npy \
    --config configs/default_config.yaml \
    --real-accuracy 0.9725 \
    --num-permutations 200 \
    --out outputs/permutation_test_off.json
```

⚠️ This re-runs the **entire** 10-fold cross-validation protocol from
randomly initialized weights, once per permutation (`num_permutations x
num_folds` full training runs in total). Budget compute time accordingly.

## Results

| Task          | Balanced Accuracy | AUC    |
|---------------|:------------------:|:------:|
| PD-OFF vs. HC | 97.25% ± 0.97%     | 0.9900 |
| PD-ON  vs. HC | 94.22% ± 1.12%     | 0.9650 |

See the paper for the full results, ablation studies, and Integrated
Gradients attribution analysis.

## Reproducibility Notes

- All cross-validation is **subject-independent**: no subject's epochs ever
  appear in both the train and test partition of a fold (`src/data/dataset.py`).
- `Accuracy` is reported as **balanced accuracy** (mean of sensitivity and
  specificity) throughout, with inverse class-frequency weighting applied
  during training, so that the class imbalance between HC and PD subjects
  does not bias the reported metric.
- Random seeds, hardware, and library versions used to produce the paper's
  reported numbers are listed in `configs/default_config.yaml` and the
  paper's hyperparameter table (Table 2), summarized here:

  | Category   | Parameter                     | Value                              |
  |------------|--------------------------------|-------------------------------------|
  | Model      | Optimizer                     | AdamW                               |
  |            | Learning rate                 | 1e-3                                |
  |            | Weight decay                  | 1e-4                                |
  |            | Batch size                    | 64                                  |
  |            | Epochs                        | 30                                  |
  |            | Early stopping patience       | 8                                   |
  |            | LR scheduler                  | Cosine annealing                    |
  |            | Random seed                   | 42                                  |
  | Hardware   | GPU                           | NVIDIA RTX 3090 (24GB) / RTX 4080 (16GB) |
  |            | CUDA version                  | 11.7                                |
  |            | PyTorch version                | 2.0.1                               |
  | Model Size | Total trainable parameters    | ~5.2M                               |
  | Time       | Total training time (10 folds) | ~2–2.5 hours                        |

## Citation

If you use this code, please cite:

```bibtex
@article{najafi2026hvfnet,
  title   = {HVF-Net: A Hypergraph-Based Volumetric Fusion Network for
             EEG-Based Parkinson's Disease Classification},
  author  = {Najafi, Sepideh},
  year    = {2026},
  note    = {Manuscript submitted for publication}
}
```

## License

This project is released under the [MIT License](LICENSE).
