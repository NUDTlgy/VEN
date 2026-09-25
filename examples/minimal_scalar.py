"""Small end-to-end standard continuous Gaussian VEN example."""

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


def main() -> None:
    torch.manual_seed(7)
    dtype = torch.float64
    node_attr = torch.randn(4, 2, dtype=dtype)
    sample = torch.randn(4, 3, dtype=dtype)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    edge_attr = torch.randn(3, 1, dtype=dtype)

    model = GaussianVEN(2, 1, 3, state_dim=1, hidden_dim=16).to(dtype=dtype)
    factors = model(node_attr, edge_index, sample, edge_attr)

    observed = selector([0, 3], 4, dtype=dtype)
    values = torch.tensor([0.4, -0.2], dtype=dtype)
    noise_variance = torch.tensor([0.05, 0.05], dtype=dtype)

    cholesky = posterior_cholesky(factors.precision, factors.information, observed, values, noise_variance)
    schur = posterior_schur(factors.precision, factors.information, observed, values, noise_variance)
    cache = prepare_woodbury_cache(factors.precision)
    woodbury = posterior_woodbury_cached(cache, factors.information, observed, values, noise_variance)

    print("precision shape:", tuple(factors.precision.shape))
    print("posterior mean:", cholesky.mean.tolist())
    print("max backend mean difference:", max(
        (cholesky.mean - schur.mean).abs().max().item(),
        (cholesky.mean - woodbury.mean).abs().max().item(),
    ))
    print("max backend covariance difference:", max(
        (cholesky.covariance - schur.covariance).abs().max().item(),
        (cholesky.covariance - woodbury.covariance).abs().max().item(),
    ))


if __name__ == "__main__":
    main()
