from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv


class WeightedSAGEConv(SAGEConv):
    """Native mean SAGEConv with optional nonnegative COO edge weights."""

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            return super().forward(x, edge_index)

        source, target = edge_index
        if edge_weight.ndim != 1 or edge_weight.shape != source.shape:
            raise ValueError("edge_weight must have one entry per edge")
        weights = edge_weight.to(x)
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("edge_weight must be finite and nonnegative")

        total = x.new_zeros(x.size(0)).index_add_(0, target, weights)
        # A node with no positive incoming weight receives a zero neighbor vector.
        denominator = torch.where(total > 0, total, 1)
        normalized = weights / denominator[target]
        neighbors = torch.sparse_coo_tensor(
            edge_index.flip(0), normalized, (x.size(0), x.size(0))
        )
        return self.lin_l(torch.sparse.mm(neighbors, x)) + self.lin_r(x)


from ._elliptic import load_elliptic


class GraphSAGE(nn.Module):
    """Full-graph node classifier returning raw logits."""

    @classmethod
    def from_elliptic(cls, root, **model_options):
        """Return a backbone and its splits, using options from elliptic.yaml."""
        splits = load_elliptic(root)
        model = cls(splits["train"][0].num_features, out_channels=2, **model_options)
        return model, splits

    def __init__(
        self, in_channels: int, hidden_channels: int, out_channels: int,
        num_layers: int = 2, dropout: float = 0.5, batch_norm: bool = False,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least one")
        if not 0 <= dropout <= 1:
            raise ValueError("dropout must be in [0, 1]")
        self.dropout = dropout
        channels = [in_channels, *([hidden_channels] * (num_layers - 1)), out_channels]
        self.convs = nn.ModuleList(
            WeightedSAGEConv(source, target)
            for source, target in zip(channels, channels[1:])
        )
        self.norms = nn.ModuleList(
            nn.BatchNorm1d(hidden_channels) if batch_norm else nn.Identity()
            for _ in range(num_layers - 1)
        )

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            if isinstance(norm, nn.BatchNorm1d):
                norm.reset_parameters()

    def get_embeddings(self, x, edge_index, edge_weight=None):
        """Return the input to the classifier, including training-mode dropout.

        For a one-layer model this is the original input.
        """
        for conv, norm in zip(self.convs[:-1], self.norms):
            x = conv(x, edge_index, edge_weight)
            x = F.dropout(norm(x).relu(), p=self.dropout, training=self.training)
        return x

    def forward_with_embeddings(self, x, edge_index, edge_weight=None):
        embeddings = self.get_embeddings(x, edge_index, edge_weight)
        logits = self.convs[-1](embeddings, edge_index, edge_weight)
        return logits, embeddings

    def forward(self, x, edge_index, edge_weight=None):
        return self.forward_with_embeddings(x, edge_index, edge_weight)[0]
