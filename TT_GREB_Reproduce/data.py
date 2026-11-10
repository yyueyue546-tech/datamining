"""四数据集加载与 1:1:8 完整图切分。

协议（论文原意）：
  - train: 1 张干净完整图
  - valid: 1 张干净完整图
  - test : 8 张完整图，同结构、不同噪声种子
评估时只使用 test_mask 指定的节点。
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import EdgeStorage, GlobalStorage, NodeStorage

from .paths import DATASETS, DATASET_META


def register_safe_globals() -> None:
    add = getattr(torch.serialization, "add_safe_globals", None)
    if add is not None:
        add([Data, DataEdgeAttr, DataTensorAttr,
             GlobalStorage, NodeStorage, EdgeStorage])


def _ensure_masks(graph: Data, num_classes: int, seed: int,
                  train_ratio=0.1, valid_ratio=0.1) -> Data:
    """没有官方 mask 时按类别分层生成。"""
    if hasattr(graph, "train_mask") and graph.train_mask is not None:
        return graph
    labels = graph.y.reshape(-1)
    gen = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for cls in range(num_classes):
        idx = torch.nonzero(labels == cls, as_tuple=False).view(-1).numpy()
        gen.shuffle(idx)
        n = len(idx)
        n_tr = max(1, int(round(n * train_ratio)))
        n_va = max(1, int(round(n * valid_ratio)))
        tr.append(idx[:n_tr])
        va.append(idx[n_tr:n_tr + n_va])
        te.append(idx[n_tr + n_va:])
    tr = torch.from_numpy(np.concatenate(tr)).long()
    va = torch.from_numpy(np.concatenate(va)).long()
    te = torch.from_numpy(np.concatenate(te)).long()
    graph.train_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
    graph.val_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
    graph.test_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
    graph.train_mask[tr] = True
    graph.val_mask[va] = True
    graph.test_mask[te] = True
    return graph


def load_clean(name: str, seed: int = 0) -> Data:
    """加载原始完整图，保证有 train/val/test mask。"""
    register_safe_globals()
    meta = DATASET_META[name]

    if name == "cora":
        ds = Planetoid(root=str(DATASETS["cora"].parent), name="Cora", split="public")
        graph = ds[0].cpu()
    elif name == "amazon":
        ds = Amazon(root=str(DATASETS["amazon"].parent), name="Photo")
        graph = ds[0].cpu()
        graph = _ensure_masks(graph, meta["num_classes"], seed)
    elif name == "twitch":
        graph = _load_twitch(DATASETS["twitch"])
        graph = _ensure_masks(graph, meta["num_classes"], seed)
    elif name == "ogb":
        graph = _load_ogb_arxiv(DATASETS["ogb"] / "raw")
        graph = _ensure_masks(graph, meta["num_classes"], seed)
    else:
        raise NotImplementedError(f"dataset {name!r} not wired yet")

    graph.train_mask = graph.train_mask.bool()
    graph.val_mask = graph.val_mask.bool()
    graph.test_mask = graph.test_mask.bool()
    return graph


def split_118(graph: Data, seed: int = 0, n_test_graphs: int = 8):
    """返回 (train, valid, test_list)。

    按论文 1:1:8 协议重新划分全图节点：train:val:test = 270:270:2168。
    8 张 test 图与训练图节点/边结构完全相同，噪声由 shift.py 注入。
    """
    graph = graph.cpu()
    n = graph.num_nodes
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)

    n_train = int(n * 0.50)
    n_val = int(n * 0.25)
    n_test = n - n_train - n_val

    idx_train = perm[:n_train]
    idx_val = perm[n_train:n_train + n_val]
    idx_test = perm[n_train + n_val:]

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True

    base = graph.clone()
    base.train_mask = train_mask
    base.val_mask = val_mask
    base.test_mask = test_mask

    test_graphs = [base.clone() for _ in range(n_test_graphs)]

    print(f"[split_50_25_25] train={int(train_mask.sum())} val={int(val_mask.sum())} test={int(test_mask.sum())}")
    return base, base.clone(), test_graphs


def describe(name: str, graph: Data) -> str:
    tr = int(graph.train_mask.sum())
    va = int(graph.val_mask.sum())
    te = int(graph.test_mask.sum())
    return (f"{name}: nodes={graph.num_nodes} edges={graph.edge_index.size(1)} "
            f"feat={graph.x.size(1)} classes={DATASET_META[name]['num_classes']} "
            f"train={tr} val={va} test={te}")

def _load_twitch(root) -> Data:
    """加载 Musae Twitch-DE。raw/musae_DE_{edges.csv,target.csv,features.json}
    节点 9498，边 153138(单向->对称 306276)，特征 3170 维多热，二分类(mature)。
    """
    import json
    import pandas as pd

    root = Path(root)
    raw = root / "raw" if (root / "raw").exists() else root

    edges = pd.read_csv(raw / "musae_DE_edges.csv")
    tgt = pd.read_csv(raw / "musae_DE_target.csv")
    with open(raw / "musae_DE_features.json", encoding="utf-8") as f:
        feats = json.load(f)

    n = len(tgt)
    dim = 3170

    y = torch.tensor(tgt["mature"].astype(int).to_numpy(), dtype=torch.long)

    x = torch.zeros(n, dim)
    for nid_str, idxs in feats.items():
        if idxs:
            x[int(nid_str), torch.as_tensor(idxs, dtype=torch.long)] = 1.0

    src = torch.from_numpy(edges["from"].to_numpy().copy()).long()
    dst = torch.from_numpy(edges["to"].to_numpy().copy()).long()
    edge_index = torch.stack(
        [torch.cat([src, dst]), torch.cat([dst, src])], dim=0
    )

    graph = Data(x=x, edge_index=edge_index, y=y)
    graph.num_nodes = n
    return graph

def _load_ogb_arxiv(raw) -> Data:
    """手读 ogbn-arxiv raw 目录下的 csv.gz，构造 PyG Data。"""
    import pandas as pd
    from torch_geometric.utils import to_undirected, remove_self_loops

    x = pd.read_csv(raw / "node-feat.csv.gz", header=None).values.astype("float32")
    y = pd.read_csv(raw / "node-label.csv.gz", header=None).values.ravel().astype("int64")
    ei = pd.read_csv(raw / "edge.csv.gz", header=None).values.T.astype("int64")

    print(f"[OGB] loading arxiv: nodes={x.shape[0]} feats={x.shape[1]} edges={ei.shape[1]}")

    edge_index = torch.from_numpy(ei)
    edge_index = to_undirected(edge_index)
    edge_index, _ = remove_self_loops(edge_index)

    return Data(
        x=torch.from_numpy(x),
        y=torch.from_numpy(y),
        edge_index=edge_index,
    )
