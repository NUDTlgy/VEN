from __future__ import annotations

import torch

from ven import (
    GaussianVEN,
    posterior_cholesky,
    posterior_schur,
    posterior_woodbury_cached,
    prepare_woodbury_cache,
    selector,
)


def test_three_backends_return_same_posterior() -> None:
    dtype = torch.float64
    precision = torch.tensor(
        [[2.5, -0.4, 0.0, 0.0], [-0.4, 2.7, -0.5, 0.0], [0.0, -0.5, 2.8, -0.3], [0.0, 0.0, -0.3, 2.2]],
        dtype=dtype,
    )
    information = torch.tensor([0.2, -0.4, 0.7, 0.1], dtype=dtype)
    observation_map = selector([0, 3], 4, dtype=dtype)
    observed_values = torch.tensor([0.5, -0.1], dtype=dtype)
    noise = torch.tensor([0.08, 0.12], dtype=dtype)

    cholesky = posterior_cholesky(precision, information, observation_map, observed_values, noise)
    schur = posterior_schur(precision, information, observation_map, observed_values, noise)
    woodbury = posterior_woodbury_cached(prepare_woodbury_cache(precision), information, observation_map, observed_values, noise)

    assert torch.allclose(cholesky.mean, schur.mean, atol=1e-10, rtol=1e-10)
    assert torch.allclose(cholesky.mean, woodbury.mean, atol=1e-10, rtol=1e-10)
    assert torch.allclose(cholesky.covariance, schur.covariance, atol=1e-10, rtol=1e-10)
    assert torch.allclose(cholesky.covariance, woodbury.covariance, atol=1e-10, rtol=1e-10)


def test_model_builds_scalar_and_block_precisions() -> None:
    torch.manual_seed(7)
    for state_dim in (1, 2):
        model = GaussianVEN(node_attr_dim=2, edge_attr_dim=1, sample_dim=3, state_dim=state_dim, hidden_dim=12).double()
        node_attr = torch.randn(4, 2, dtype=torch.float64)
        sample = torch.randn(4, 3, dtype=torch.float64)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        edge_attr = torch.randn(3, 1, dtype=torch.float64)
        factors = model(node_attr, edge_index, sample, edge_attr)
        assert factors.precision.shape == (4 * state_dim, 4 * state_dim)
        torch.linalg.cholesky(factors.precision)
