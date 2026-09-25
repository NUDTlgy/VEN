"""Standard continuous Gaussian VEN model.

For fixed graph topology and static attributes, the node and edge maps produce
one shared precision ``Q``.  Current sample features produce only ``b``.
Observations are supplied separately to the exact inference functions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


class _MLP(nn.Sequential):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__(nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, output_dim))


@dataclass(frozen=True)
class VENFactors:
    node_blocks: Tensor
    edge_blocks: Tensor
    precision: Tensor
    information: Tensor


class GaussianVEN(nn.Module):
    """Learn local Gaussian potentials and assemble one global precision.

    Parameters
    ----------
    node_attr_dim:
        Dimension of static node attributes ``a_i``.
    edge_attr_dim:
        Dimension of static edge attributes ``a_e``.  Use zero for no edge
        features; the graph still contributes through endpoint attributes.
    sample_dim:
        Dimension of current sample features ``x_i``.
    state_dim:
        Joint state dimension per node.  ``state_dim=1`` is the scalar case.
    hidden_dim:
        Width of the local MLPs.
    jitter:
        Positive diagonal jitter used in ``D_i``.
    """

    def __init__(self, node_attr_dim: int, edge_attr_dim: int, sample_dim: int, state_dim: int = 1, hidden_dim: int = 64, jitter: float = 1e-4) -> None:
        super().__init__()
        if min(node_attr_dim, edge_attr_dim, sample_dim) < 0 or state_dim < 1 or hidden_dim < 1:
            raise ValueError("dimensions must be non-negative and state_dim/hidden_dim must be positive")
        self.node_attr_dim = node_attr_dim
        self.edge_attr_dim = edge_attr_dim
        self.sample_dim = sample_dim
        self.state_dim = state_dim
        self.jitter = float(jitter)
        block_size = state_dim * state_dim
        self.node_map = _MLP(node_attr_dim, hidden_dim, block_size)
        self.edge_map = _MLP(2 * node_attr_dim + edge_attr_dim, hidden_dim, block_size)
        self.source_map = _MLP(sample_dim + node_attr_dim, hidden_dim, state_dim)

    def _positive_block(self, raw: Tensor) -> Tensor:
        d = self.state_dim
        lower = raw.reshape(-1, d, d).tril()
        diagonal = torch.diagonal(lower, dim1=-2, dim2=-1)
        lower = lower - torch.diag_embed(diagonal) + torch.diag_embed(torch.nn.functional.softplus(diagonal) + self.jitter)
        return lower @ lower.transpose(-1, -2)

    def local_factors(self, node_attr: Tensor, edge_index: Tensor, edge_attr: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Return positive node blocks ``D_i`` and edge blocks ``W_e``."""

        if node_attr.ndim != 2 or node_attr.shape[1] != self.node_attr_dim:
            raise ValueError("node_attr must have shape [N, node_attr_dim]")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if edge_attr is None:
            edge_attr = node_attr.new_zeros((edge_index.shape[1], self.edge_attr_dim))
        if edge_attr.shape != (edge_index.shape[1], self.edge_attr_dim):
            raise ValueError("edge_attr must have shape [E, edge_attr_dim]")
        source, target = edge_index.long()
        if bool((source < 0).any()) or bool((target < 0).any()) or bool((source >= node_attr.shape[0]).any()) or bool((target >= node_attr.shape[0]).any()):
            raise ValueError("edge_index contains an invalid node id")
        node_blocks = self._positive_block(self.node_map(node_attr))
        edge_input = torch.cat((node_attr[source], node_attr[target], edge_attr), dim=-1)
        edge_blocks = self._positive_block(self.edge_map(edge_input))
        return node_blocks, edge_blocks

    def assemble_precision(self, node_blocks: Tensor, edge_blocks: Tensor, edge_index: Tensor) -> Tensor:
        """Assemble ``blkdiag(D_i) + sum_e B_e^T W_e B_e``."""

        nodes, d, _ = node_blocks.shape
        if edge_blocks.shape[0] != edge_index.shape[1] or edge_blocks.shape[1:] != (d, d):
            raise ValueError("edge_blocks and edge_index are inconsistent")
        dimension = nodes * d
        precision = node_blocks.new_zeros((dimension, dimension))
        for i in range(nodes):
            sl = slice(i * d, (i + 1) * d)
            precision[sl, sl] += node_blocks[i]
        for edge, (left, right) in enumerate(edge_index.T.tolist()):
            li = slice(left * d, (left + 1) * d)
            ri = slice(right * d, (right + 1) * d)
            weight = edge_blocks[edge]
            precision[li, li] += weight
            precision[ri, ri] += weight
            precision[li, ri] -= weight
            precision[ri, li] -= weight
        return precision

    def forward(self, node_attr: Tensor, edge_index: Tensor, sample: Tensor, edge_attr: Tensor | None = None) -> VENFactors:
        """Construct ``Q`` and ``b`` for one graph/sample pair."""

        if sample.ndim != 2 or sample.shape != (node_attr.shape[0], self.sample_dim):
            raise ValueError("sample must have shape [N, sample_dim]")
        node_blocks, edge_blocks = self.local_factors(node_attr, edge_index, edge_attr)
        information = self.source_map(torch.cat((sample, node_attr), dim=-1)).reshape(-1)
        precision = self.assemble_precision(node_blocks, edge_blocks, edge_index)
        return VENFactors(node_blocks=node_blocks, edge_blocks=edge_blocks, precision=precision, information=information)
