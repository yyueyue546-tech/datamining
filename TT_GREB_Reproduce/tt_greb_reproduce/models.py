from __future__ import annotations


import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, SAGEConv



def build_model(backbone: str, in_dim: int, hidden_dim: int, num_classes: int,
                dropout: float = 0.5, **kwargs) -> nn.Module:
    """兼容旧调用；新代码请用 backbones.build_backbone（支持外部模型）。"""
    from .backbones import build_backbone
    return build_backbone(backbone, in_dim, hidden_dim, num_classes, dropout, **kwargs)


class _BaseBackbone(nn.Module):
    """统一接口：out_channels / forward_with_embeddings / reset_parameters。"""

    def __init__(self, num_classes: int, dropout: float):
        super().__init__()
        self.out_channels = num_classes
        self.num_classes = num_classes
        self.dropout = dropout

    def forward_with_embeddings(self, x, edge_index, edge_weight=None):
        logits, embeddings = self(x, edge_index, edge_weight=edge_weight,
                                  return_embedding=True)
        return logits, embeddings

    def reset_parameters(self):
        for module in self.modules():
            reset = getattr(module, "reset_parameters", None)
            if reset is not None and module is not self:
                reset()


class GCN(_BaseBackbone):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.5):
        super().__init__(num_classes, dropout)
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, x, edge_index, edge_weight=None, return_embedding: bool = False):
        hidden = self.conv1(x, edge_index, edge_weight=edge_weight).relu()
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        logits = self.conv2(hidden, edge_index, edge_weight=edge_weight)
        return (logits, hidden) if return_embedding else logits


class GraphSAGE(_BaseBackbone):
    """GraphSAGE 骨干，接口与 GCN 保持一致（edge_weight 被忽略）。"""

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.5):
        super().__init__(num_classes, dropout)
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, num_classes)

    def forward(self, x, edge_index, edge_weight=None, return_embedding: bool = False):
        hidden = self.conv1(x, edge_index).relu()
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        logits = self.conv2(hidden, edge_index)
        return (logits, hidden) if return_embedding else logits

