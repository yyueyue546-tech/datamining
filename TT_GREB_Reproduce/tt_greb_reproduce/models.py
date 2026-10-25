from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None, return_embedding: bool = False):
        hidden = self.conv1(x, edge_index, edge_weight=edge_weight).relu()
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        logits = self.conv2(hidden, edge_index, edge_weight=edge_weight)
        return (logits, hidden) if return_embedding else logits
