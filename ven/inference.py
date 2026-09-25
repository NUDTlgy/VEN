"""Exact Gaussian conditioning and interchangeable VEN inference backends.

The functions in this module deliberately operate on ordinary dense tensors so
that the mathematical contract is easy to inspect.  Sparse or block-sparse
implementations can use the same equations without changing the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class GaussianPosterior:
    """Posterior moments and the conditioned natural parameters."""

    mean: Tensor
    covariance: Tensor
    precision: Tensor
    information: Tensor


@dataclass(frozen=True)
class WoodburyCache:
    """Base precision quantities reused across evidence queries."""

    precision: Tensor
    covariance: Tensor


def selector(indices: Tensor | list[int], dimension: int, *, dtype=None, device=None) -> Tensor:
    """Return a linear selector matrix ``S`` for observed state coordinates."""

    indices = torch.as_tensor(indices, dtype=torch.long, device=device)
    if indices.ndim != 1 or bool((indices < 0).any()) or bool((indices >= dimension).any()):
        raise ValueError("indices must be a one-dimensional vector in [0, dimension)")
    result = torch.zeros((indices.numel(), dimension), dtype=dtype or torch.get_default_dtype(), device=indices.device)
    if indices.numel():
        result[torch.arange(indices.numel(), device=indices.device), indices] = 1
    return result


def _observation_covariance(noise: Tensor, count: int, *, dtype, device) -> Tensor:
    noise = torch.as_tensor(noise, dtype=dtype, device=device)
    if noise.ndim == 0:
        if bool((noise <= 0).any()):
            raise ValueError("observation variance must be positive")
        return torch.eye(count, dtype=dtype, device=device) * noise
    if noise.ndim == 1:
        if noise.numel() != count or bool((noise <= 0).any()):
            raise ValueError("diagonal observation variance must have length k and be positive")
        return torch.diag(noise)
    if noise.shape != (count, count) or bool((noise != noise.T).any()) or bool((torch.linalg.eigvalsh(noise) <= 0).any()):
        raise ValueError("observation covariance must be symmetric positive definite")
    return noise


def conditioned_system(
    precision: Tensor,
    information: Tensor,
    observation_map: Tensor,
    observed_values: Tensor,
    observation_variance: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Apply ``y = S z + eps`` to a natural-parameter Gaussian."""

    if precision.ndim != 2 or precision.shape[0] != precision.shape[1]:
        raise ValueError("precision must be square")
    dimension = precision.shape[0]
    if information.shape != (dimension,):
        raise ValueError("information must have shape [dimension]")
    if observation_map.ndim != 2 or observation_map.shape[1] != dimension:
        raise ValueError("observation_map must have shape [k, dimension]")
    if observed_values.shape != (observation_map.shape[0],):
        raise ValueError("observed_values must have shape [k]")
    R = _observation_covariance(
        observation_variance,
        observation_map.shape[0],
        dtype=precision.dtype,
        device=precision.device,
    )
    noise_precision = torch.linalg.inv(R)
    q_post = precision + observation_map.T @ noise_precision @ observation_map
    b_post = information + observation_map.T @ noise_precision @ observed_values
    return q_post, b_post, R


def _validate_spd(matrix: Tensor) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    torch.linalg.cholesky(matrix)


def cholesky_solve(matrix: Tensor, rhs: Tensor) -> Tensor:
    """Solve an SPD system with Cholesky factorization."""

    factor = torch.linalg.cholesky(matrix)
    if rhs.ndim == 1:
        return torch.cholesky_solve(rhs.unsqueeze(-1), factor).squeeze(-1)
    return torch.cholesky_solve(rhs, factor)


def schur_solve(matrix: Tensor, rhs: Tensor, elimination_order: Tensor | list[int] | None = None) -> Tensor:
    """Solve an SPD system by explicit scalar Schur elimination.

    This is the transparent reference implementation of variable elimination.
    ``rhs`` may be a vector ``[D]`` or a matrix ``[D, K]``.  A block-state VEN
    can use the same routine after flattening its state coordinates.
    """

    _validate_spd(matrix)
    dimension = matrix.shape[0]
    if rhs.shape[0] != dimension:
        raise ValueError("rhs and matrix dimensions do not match")
    if elimination_order is None:
        order = torch.arange(dimension, device=matrix.device)
    else:
        order = torch.as_tensor(elimination_order, dtype=torch.long, device=matrix.device)
        if order.ndim != 1 or order.numel() != dimension or torch.unique(order).numel() != dimension:
            raise ValueError("elimination_order must be a permutation of all coordinates")
    permuted = matrix.index_select(0, order).index_select(1, order)
    reduced_rhs = rhs.index_select(0, order)
    records: list[tuple[Tensor, Tensor, Tensor]] = []
    while permuted.shape[0] > 1:
        pivot = permuted[0, 0]
        row = permuted[0, 1:].clone()
        rhs0 = reduced_rhs[0].clone()
        permuted = permuted[1:, 1:] - torch.outer(row, row) / pivot
        if reduced_rhs.ndim == 1:
            reduced_rhs = reduced_rhs[1:] - row * rhs0 / pivot
        else:
            reduced_rhs = reduced_rhs[1:] - row.unsqueeze(-1) * rhs0.unsqueeze(0) / pivot
        records.append((pivot, row, rhs0))
    solution = reduced_rhs / permuted[0, 0]
    for pivot, row, rhs0 in reversed(records):
        if solution.ndim == 1:
            recovered = (rhs0 - torch.dot(row, solution)) / pivot
        else:
            recovered = (rhs0 - row @ solution) / pivot
        solution = torch.cat((recovered.unsqueeze(0), solution), dim=0)
    restored = torch.empty_like(solution)
    restored.index_copy_(0, order, solution)
    return restored


def posterior_cholesky(
    precision: Tensor,
    information: Tensor,
    observation_map: Tensor,
    observed_values: Tensor,
    observation_variance: Tensor,
) -> GaussianPosterior:
    """Condition and solve the posterior with a Cholesky backend."""

    q_post, b_post, _ = conditioned_system(precision, information, observation_map, observed_values, observation_variance)
    factor = torch.linalg.cholesky(q_post)
    mean = torch.cholesky_solve(b_post[:, None], factor).squeeze(-1)
    covariance = torch.cholesky_inverse(factor)
    return GaussianPosterior(mean, covariance, q_post, b_post)


def posterior_schur(
    precision: Tensor,
    information: Tensor,
    observation_map: Tensor,
    observed_values: Tensor,
    observation_variance: Tensor,
    elimination_order: Tensor | list[int] | None = None,
) -> GaussianPosterior:
    """Condition and solve the posterior by direct Schur elimination."""

    q_post, b_post, _ = conditioned_system(precision, information, observation_map, observed_values, observation_variance)
    mean = schur_solve(q_post, b_post, elimination_order)
    covariance = schur_solve(q_post, torch.eye(q_post.shape[0], dtype=q_post.dtype, device=q_post.device), elimination_order)
    return GaussianPosterior(mean, covariance, q_post, b_post)


def posterior_woodbury(
    precision: Tensor,
    information: Tensor,
    observation_map: Tensor,
    observed_values: Tensor,
    observation_variance: Tensor,
) -> GaussianPosterior:
    """Condition using a base covariance and a small evidence solve.

    Use :func:`prepare_woodbury_cache` and
    :func:`posterior_woodbury_cached` when serving repeated queries.
    """

    return posterior_woodbury_cached(
        prepare_woodbury_cache(precision),
        information,
        observation_map,
        observed_values,
        observation_variance,
    )


def prepare_woodbury_cache(precision: Tensor) -> WoodburyCache:
    """Factor a fixed base precision once for repeated Woodbury queries."""

    _validate_spd(precision)
    factor = torch.linalg.cholesky(precision)
    return WoodburyCache(precision=precision.detach(), covariance=torch.cholesky_inverse(factor).detach())


def posterior_woodbury_cached(
    cache: WoodburyCache,
    information: Tensor,
    observation_map: Tensor,
    observed_values: Tensor,
    observation_variance: Tensor,
) -> GaussianPosterior:
    """Run a Woodbury query using a previously prepared base cache."""

    precision = cache.precision
    if information.ndim != 1 or information.shape[0] != precision.shape[0]:
        raise ValueError("information must have shape [dimension]")
    R = _observation_covariance(
        observation_variance,
        observation_map.shape[0],
        dtype=precision.dtype,
        device=precision.device,
    )
    base_covariance = cache.covariance
    base_mean = base_covariance @ information
    evidence_covariance = R + observation_map @ base_covariance @ observation_map.T
    gain = base_covariance @ observation_map.T @ torch.linalg.inv(evidence_covariance)
    mean = base_mean + gain @ (observed_values - observation_map @ base_mean)
    covariance = base_covariance - gain @ observation_map @ base_covariance
    q_post = precision + observation_map.T @ torch.linalg.inv(R) @ observation_map
    b_post = information + observation_map.T @ torch.linalg.inv(R) @ observed_values
    return GaussianPosterior(mean, covariance, q_post, b_post)
