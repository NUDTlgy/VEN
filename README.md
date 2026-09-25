# Variable Elimination Networks (VEN)

This directory contains the reviewer-facing reference implementation of the standard continuous Gaussian VEN described in the paper.

VEN learns local Gaussian potentials and composes them into one global precision matrix. For a graph `G=(V,E)`, static node/edge attributes `a`, and current sample features `x`, the model constructs

```text
Q_theta(G,a) = blkdiag(D_1,...,D_n) + sum_e B_e^T W_e B_e
b_theta(x,G) = stack_i f_theta,b(x_i,a_i)
```

The node and edge blocks are positive by construction:

```text
D_i = L_i L_i^T + epsilon I
W_e = A_e A_e^T
```

Optional linear observations `y = S z + epsilon`, with `epsilon ~ N(0,R)`, update the same Gaussian system:

```text
Q_plus = Q + S^T R^(-1) S
b_plus = b + S^T R^(-1) y
```

The package exposes three numerically equivalent inference backends:

- `posterior_cholesky`: direct dense Cholesky factorization;
- `posterior_schur`: explicit scalar Schur elimination and back-substitution;
- `prepare_woodbury_cache` + `posterior_woodbury_cached`: base covariance cache plus a small evidence-space solve.

All three return the posterior mean and covariance. The backend changes the numerical route, not the learned model or posterior semantics.

## Layout

```text
ven_public_release/
|-- ven/
|   |-- model.py       # GaussianVEN: local maps, Q assembly, b source map
|   |-- inference.py   # conditioning, Cholesky, Schur, Woodbury
|   `-- __init__.py
|-- examples/
|   `-- minimal_scalar.py
|-- tests/
|   `-- test_backends.py
|-- requirements.txt
`-- pyproject.toml
```

The implementation uses plain PyTorch tensors and does not require PyTorch Geometric. A graph is passed as `edge_index` with shape `[2, E]`, using one undirected entry per edge. `state_dim=1` is the scalar special case; `state_dim>1` gives block-valued node states.

## Quick start

```bash
python -m pip install -r requirements.txt
python -m examples.minimal_scalar
pytest -q
```

The example constructs a four-node graph, adds two noisy observations, and verifies that the three backends agree to numerical precision.

## Scope

This release is intentionally small and auditable. It contains the standard continuous Gaussian model with fixed `Q_theta(G,a)` for a fixed graph and static attributes, together with the exact conditional inference interface. Dataset downloaders, experiment queues, cached checkpoints, and task-specific adapters belong in a separate reproducibility archive rather than the core reviewer repository.

## Reproducibility note

The paper's public experiments use the same mathematical interface with task-specific data loaders and evaluation scripts. This clean reference package is the minimal implementation of the method; it is suitable for inspecting the equations, running the backend equivalence test, and adapting the model to a new attributed graph.
