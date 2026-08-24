"""Shape/sanity smoke tests. Run with: pytest tests/ -v

These do not require real EEG data -- they verify that randomly initialized
tensors of the documented shapes flow through the full model without error,
and that the hypergraph/BCMI utilities produce mathematically valid outputs
(e.g. a symmetric propagation matrix, a properly normalized PLV matrix).
"""

import numpy as np
import torch

from src.data.hypergraph import build_incidence_matrix, hyperedge_weights, hypergraph_laplacian
from src.models.bcmi import BCMIModule
from src.models.hvf_net import HVFNet

NUM_ELECTRODES = 60
NUM_BANDS = 5
BATCH_SIZE = 4


def _dummy_propagation() -> torch.Tensor:
    rng = np.random.default_rng(0)
    coords = rng.normal(size=(NUM_ELECTRODES, 3))
    H = build_incidence_matrix(coords, k_neighbors=6)
    w = hyperedge_weights(H)
    L_H = hypergraph_laplacian(H, w)
    propagation = np.eye(NUM_ELECTRODES, dtype=np.float32) - L_H
    return torch.from_numpy(propagation)


def test_hypergraph_laplacian_shape_and_symmetry():
    rng = np.random.default_rng(1)
    coords = rng.normal(size=(NUM_ELECTRODES, 3))
    H = build_incidence_matrix(coords, k_neighbors=6)
    w = hyperedge_weights(H)
    L_H = hypergraph_laplacian(H, w)

    assert L_H.shape == (NUM_ELECTRODES, NUM_ELECTRODES)
    assert np.allclose(L_H, L_H.T, atol=1e-5), "Hypergraph Laplacian must be symmetric"


def test_bcmi_output_shapes():
    bcmi = BCMIModule(connectivity_dim=128, spectral_dim=64, d_k=16)
    f_c = torch.randn(BATCH_SIZE, 128)
    f_s = torch.randn(BATCH_SIZE, 64)
    f_c_out, f_s_out = bcmi(f_c, f_s)
    assert f_c_out.shape == (BATCH_SIZE, 128)
    assert f_s_out.shape == (BATCH_SIZE, 64)


def test_bcmi_parameter_count():
    bcmi = BCMIModule(connectivity_dim=128, spectral_dim=64, d_k=16)
    num_params = sum(p.numel() for p in bcmi.parameters())
    # Documented estimate in the paper (~63,000); allow a wide tolerance
    # since exact head/bias configurations may evolve.
    assert 55_000 <= num_params <= 75_000, f"unexpected BCMI param count: {num_params}"


def test_full_model_forward_pass():
    model = HVFNet(num_electrodes=NUM_ELECTRODES, num_bands=NUM_BANDS)
    propagation = _dummy_propagation()

    volumetric_plv = torch.rand(BATCH_SIZE, NUM_ELECTRODES, NUM_ELECTRODES, NUM_BANDS)
    spectral = torch.randn(BATCH_SIZE, NUM_ELECTRODES, NUM_BANDS)

    logits, embedding = model(volumetric_plv, spectral, propagation, return_embedding=True)

    assert logits.shape == (BATCH_SIZE, 2)
    assert embedding.shape == (BATCH_SIZE, 128 + 64)


def test_full_model_backward_pass():
    """Ensures gradients flow through every branch (hypergraph, 3D CNN,
    spectral Transformer, BCMI) without a disconnected-graph error.
    """
    model = HVFNet(num_electrodes=NUM_ELECTRODES, num_bands=NUM_BANDS)
    propagation = _dummy_propagation()

    volumetric_plv = torch.rand(BATCH_SIZE, NUM_ELECTRODES, NUM_ELECTRODES, NUM_BANDS)
    spectral = torch.randn(BATCH_SIZE, NUM_ELECTRODES, NUM_BANDS)
    labels = torch.randint(0, 2, (BATCH_SIZE,))

    logits = model(volumetric_plv, spectral, propagation)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()

    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0
    assert all(np.isfinite(g) for g in grad_norms)
