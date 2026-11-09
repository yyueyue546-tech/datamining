from __future__ import annotations

from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """Full-graph node classifier returning raw logits."""

    def __init__(
        self, in_channels: int, hidden_channels: int, out_channels: int,
        num_layers: int = 2, dropout: float = 0.5, batch_norm: bool = False,
        *, normalize: bool = True, add_self_loops: bool | None = None,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least one")
        if not 0 <= dropout <= 1:
            raise ValueError("dropout must be in [0, 1]")
        self.dropout = dropout
        channels = [in_channels, *([hidden_channels] * (num_layers - 1)), out_channels]
        self.convs = nn.ModuleList(
            GCNConv(source, target, cached=False, normalize=normalize, add_self_loops=add_self_loops)
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
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.dropout(norm(x).relu(), p=self.dropout, training=self.training)
        return x

    def forward_with_embeddings(self, x, edge_index, edge_weight=None):
        embeddings = self.get_embeddings(x, edge_index, edge_weight)
        logits = self.convs[-1](embeddings, edge_index, edge_weight=edge_weight)
        return logits, embeddings

    def forward(self, x, edge_index, edge_weight=None):
        return self.forward_with_embeddings(x, edge_index, edge_weight)[0]
