from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data


class _ReweightMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, out_dim))

    def forward(self, value):
        return torch.sigmoid(self.net(value))


class TTGREBAdapter(nn.Module):
    """Test-time graph rebirth modules; the supplied GNN remains frozen."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.prototype_node = _ReweightMLP(feature_dim, feature_dim)
        self.environment_node = _ReweightMLP(feature_dim, feature_dim)
        self.prototype_edge = _ReweightMLP(2 * feature_dim, 1)
        self.environment_edge = _ReweightMLP(2 * feature_dim, 1)

    def weights(self, graph: Data):
        x = graph.x
        src, dst = graph.edge_index
        edge_input = torch.cat([x[src], x[dst]], dim=-1)
        return (
            self.prototype_node(x),
            self.environment_node(x),
            self.prototype_edge(edge_input).squeeze(-1),
            self.environment_edge(edge_input).squeeze(-1),
        )

    def rebirth(self, graph: Data) -> tuple[Data, tuple[torch.Tensor, ...]]:
        p_node, e_node, p_edge, e_edge = self.weights(graph)
        node_weight = (1.0 - p_node) * e_node + p_node
        edge_weight = (1.0 - p_edge) * e_edge + p_edge
        reborn = Data(
            x=graph.x * node_weight,
            edge_index=graph.edge_index,
            y=graph.y,
            edge_weight=edge_weight,
        )
        return reborn, (p_node, e_node, p_edge, e_edge)


@dataclass
class AdaptationResult:
    graph: Data
    steps: int
    final_loss: float


def _ratio_regularizer(weight: torch.Tensor, target: float) -> torch.Tensor:
    return (weight.mean() - target).square()


def adapt_test_graph(
    model: nn.Module,
    graph: Data,
    *,
    steps: int = 20,
    lr: float = 1e-2,
    alpha: float = 0.1,
    beta: float = 0.1,
    prototype_ratio: float = 0.5,
    environment_ratio: float = 0.5,
) -> AdaptationResult:
    """Optimize TT-GREB using only the graph and frozen model outputs."""
    was_training = model.training
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    adapter = TTGREBAdapter(graph.x.size(1)).to(graph.x.device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=lr)
    with torch.no_grad():
        original_logits, original_embedding = model(
            graph.x, graph.edge_index, return_embedding=True
        )

    final_loss = 0.0
    for _ in range(steps):
        reborn, weights = adapter.rebirth(graph)
        p_node, e_node, p_edge, e_edge = weights
        _, prototype_embedding = model(
            reborn.x, reborn.edge_index, edge_weight=reborn.edge_weight, return_embedding=True
        )
        # Prototype view should retain the frozen model's local predictive pattern.
        prototype_graph = Data(x=graph.x * p_node, edge_index=graph.edge_index, y=graph.y)
        _, prototype_view = model(
            prototype_graph.x, prototype_graph.edge_index, return_embedding=True
        )
        _, environment_embedding = model(
            graph.x * e_node,
            graph.edge_index,
            edge_weight=e_edge,
            return_embedding=True,
        )
        loss_pro = 1.0 - F.cosine_similarity(prototype_embedding, prototype_view, dim=-1).mean()
        # Maximize the environment discrepancy, as in the paper's environment objective.
        loss_env = -((original_embedding.detach() - environment_embedding) ** 2).mean()
        regularizer = (
            _ratio_regularizer(p_node, prototype_ratio)
            + _ratio_regularizer(p_edge, prototype_ratio)
            + _ratio_regularizer(e_node, environment_ratio)
            + _ratio_regularizer(e_edge, environment_ratio)
        )
        loss = loss_pro + alpha * loss_env + beta * regularizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())

    reborn, _ = adapter.rebirth(graph)
    if was_training:
        model.train()
    return AdaptationResult(reborn.detach(), steps, final_loss)
