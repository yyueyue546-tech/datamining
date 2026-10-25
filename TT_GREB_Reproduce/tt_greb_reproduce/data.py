from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import EdgeStorage, GlobalStorage, NodeStorage


def register_torch26_safe_globals() -> None:
    """Allow trusted PyG cache objects to load under torch 2.6."""
    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe_globals is not None:
        add_safe_globals([
            Data,
            DataEdgeAttr,
            DataTensorAttr,
            GlobalStorage,
            NodeStorage,
            EdgeStorage,
        ])


@dataclass
class GraphSplits:
    train: Data
    valid: Data
    test: list[Data]


def _as_graph(value: Any, name: str) -> Data:
    if not isinstance(value, Data):
        raise TypeError(f"{name} must be torch_geometric.data.Data, got {type(value)!r}")
    for field in ("x", "edge_index", "y"):
        if getattr(value, field, None) is None:
            raise ValueError(f"{name} is missing required field {field!r}")
    return value


def validate_splits(splits: GraphSplits, expected_test_graphs: int = 8) -> None:
    graphs = [splits.train, splits.valid, *splits.test]
    for index, graph in enumerate(graphs):
        name = "train" if index == 0 else "valid" if index == 1 else f"test[{index - 2}]"
        _as_graph(graph, name)
    if expected_test_graphs is not None and len(splits.test) != expected_test_graphs:
        raise ValueError(
            f"expected {expected_test_graphs} test graphs for Cora, got {len(splits.test)}"
        )
    feature_dim = splits.train.x.size(1)
    labels = torch.cat([g.y.reshape(-1) for g in graphs])
    if labels.numel() == 0:
        raise ValueError("the split contains no node labels")
    num_classes = int(labels.max().item()) + 1
    for index, graph in enumerate(graphs):
        name = "train" if index == 0 else "valid" if index == 1 else f"test[{index - 2}]"
        if graph.x.ndim != 2 or graph.x.size(1) != feature_dim:
            raise ValueError(f"{name} has incompatible feature shape {tuple(graph.x.shape)}")
        if graph.edge_index.ndim != 2 or graph.edge_index.size(0) != 2:
            raise ValueError(f"{name}.edge_index must have shape [2, E]")
        if graph.edge_index.numel():
            if int(graph.edge_index.min()) < 0 or int(graph.edge_index.max()) >= graph.num_nodes:
                raise ValueError(f"{name}.edge_index contains an out-of-range node index")
        if graph.y.numel() and (int(graph.y.min()) < 0 or int(graph.y.max()) >= num_classes):
            raise ValueError(f"{name}.y contains an invalid class index")
        if graph.x.size(0) != graph.y.reshape(-1).size(0):
            raise ValueError(f"{name}: x and y have different node counts")


def load_split(path: str | Path, expected_test_graphs: int = 8) -> GraphSplits:
    register_torch26_safe_globals()
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("split file must contain a dict with train, valid, and test")
    valid = payload.get("valid", payload.get("val"))
    test = payload.get("test")
    if valid is None or test is None:
        raise KeyError("split file requires keys 'train', 'valid' (or 'val'), and 'test'")
    splits = GraphSplits(
        train=_as_graph(payload["train"], "train"),
        valid=_as_graph(valid, "valid"),
        test=[_as_graph(graph, f"test[{i}]") for i, graph in enumerate(test)],
    )
    validate_splits(splits, expected_test_graphs)
    return splits


def _induced_subgraph(graph: Data, nodes: torch.Tensor) -> Data:
    nodes = nodes.sort().values
    remap = torch.full((graph.num_nodes,), -1, dtype=torch.long)
    remap[nodes] = torch.arange(nodes.numel())
    src, dst = graph.edge_index
    keep = (remap[src] >= 0) & (remap[dst] >= 0)
    edge_index = torch.stack([remap[src[keep]], remap[dst[keep]]])
    result = Data(x=graph.x[nodes], edge_index=edge_index, y=graph.y[nodes])
    return result


def standard_cora_smoke(root: str | Path = "TT_GREB_Reproduce/data") -> GraphSplits:
    """Create a deterministic 1/1/8 smoke split from standard Planetoid Cora.

    This is deliberately marked as a smoke path: it is not the paper's Cora
    preprocessing and must not be used for a reported reproduction result.
    """
    register_torch26_safe_globals()
    dataset = Planetoid(root=str(root), name="Cora")
    graph = dataset[0].cpu()
    chunks = torch.tensor_split(torch.arange(graph.num_nodes), 10)
    splits = GraphSplits(
        train=_induced_subgraph(graph, chunks[0]),
        valid=_induced_subgraph(graph, chunks[1]),
        test=[_induced_subgraph(graph, chunk) for chunk in chunks[2:]],
    )
    validate_splits(splits, expected_test_graphs=8)
    return splits


def load_masked_cora(processed_file: str | Path) -> Data:
    """Load a standard Planetoid Cora graph with train/val/test node masks."""
    processed_file = Path(processed_file)
    if processed_file.name != "data.pt":
        raise ValueError("--processed-cora must point to a processed Planetoid data.pt file")
    root = processed_file.parent.parent.parent
    register_torch26_safe_globals()
    dataset = Planetoid(root=str(root), name="Cora")
    graph = dataset[0].cpu()
    for field in ("train_mask", "val_mask", "test_mask"):
        mask = getattr(graph, field, None)
        if mask is None or mask.dtype != torch.bool or mask.numel() != graph.num_nodes:
            raise ValueError(f"Cora is missing a valid {field}")
    if int(graph.train_mask.sum()) != 140 or int(graph.val_mask.sum()) != 500 or int(graph.test_mask.sum()) != 1000:
        raise ValueError("the supplied file does not contain the expected Planetoid Cora masks")
    return graph


def move_graph(graph: Data, device: torch.device) -> Data:
    return copy.copy(graph).to(device)
