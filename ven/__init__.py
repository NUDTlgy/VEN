"""Public reference implementation of the standard continuous Gaussian VEN."""

from .inference import (
    GaussianPosterior,
    WoodburyCache,
    cholesky_solve,
    conditioned_system,
    posterior_cholesky,
    posterior_schur,
    posterior_woodbury,
    posterior_woodbury_cached,
    prepare_woodbury_cache,
    schur_solve,
    selector,
)
from .model import GaussianVEN, VENFactors

__all__ = [
    "GaussianPosterior",
    "WoodburyCache",
    "GaussianVEN",
    "VENFactors",
    "cholesky_solve",
    "conditioned_system",
    "posterior_cholesky",
    "posterior_schur",
    "posterior_woodbury",
    "posterior_woodbury_cached",
    "prepare_woodbury_cache",
    "schur_solve",
    "selector",
]
