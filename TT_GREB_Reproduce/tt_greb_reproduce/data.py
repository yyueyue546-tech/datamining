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
        ds = Planetoid(root=str(DATASETS["cora"].parent),
                       name=DATASETS["cora"].name, split="public")
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
    elif name == "elliptic":
        graph = _load_elliptic(DATASETS["elliptic"])
        # 未标注节点（y=-1）保留在图里用于消息传递，但不进入任何 mask。
        # mask 由 split_118 生成，这里不调用 _ensure_masks。
    else:
        raise NotImplementedError(f"dataset {name!r} not wired yet")

    if getattr(graph, "train_mask", None) is not None:
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

    # 只有带标签的节点参与划分（Elliptic 有大量 y=-1 的未标注节点，
    # 它们仍留在图里做消息传递，但不进入 train/val/test）。
    labels = graph.y.reshape(-1)
    eligible = torch.nonzero(labels >= 0, as_tuple=False).view(-1)
    perm = eligible[torch.randperm(eligible.numel(), generator=g)]

    n_train = int(eligible.numel() * 0.50)
    n_val = int(eligible.numel() * 0.25)
    n_test = eligible.numel() - n_train - n_val

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

    # 注意：features.json 与 edges.csv 都用 new_id（0..n-1）编号，
    # 而 target.csv 的行顺序不是 new_id 顺序，必须按 new_id 重排后再取标签。
    tgt = tgt.sort_values("new_id").reset_index(drop=True)
    if tgt["new_id"].tolist() != list(range(n)):
        raise RuntimeError("twitch new_id 不是 0..n-1 的置换")
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

def _load_elliptic(root) -> Data:
    """加载 Elliptic 比特币交易图。

    raw 目录内容：
      - 1.pkl ... 48.pkl : 每个时间步一个块，(邻接, 标签, 特征)
      - elliptic_txs_classes.csv : txId, class(1/2/unknown)
      - elliptic_txs_edgelist.csv: txId1, txId2

    pkl 的行顺序与 classes.csv 一致（块首尾相接，覆盖第 7880..203768 行，
    即第一个时间步没有特征文件），因此可以还原成全图 txId 编号：
      - 特征/标签按块顺序拼接
      - 边表按 txId 映射到本地下标
      - 邻接矩阵非对称，转成无向图
    标签映射：1(illicit)->1, 2(licit)->0, unknown->-1（不参与划分）。
    """
    import pickle
    import pandas as pd
    from torch_geometric.utils import to_undirected, remove_self_loops

    root = Path(root)
    raw = root / "raw" if (root / "raw").exists() else root

    pkl_files = sorted(raw.glob("*.pkl"), key=lambda p: int(p.stem))
    if not pkl_files:
        raise FileNotFoundError(f"no elliptic pkl found in {raw}")

    cls = pd.read_csv(raw / "elliptic_txs_classes.csv")
    label_map = {"1": 1, "2": 0, "unknown": -1}
    csv_seq = cls["class"].astype(str).map(label_map).to_numpy()

    def _pattern(arr):
        return "".join("u" if v < 0 else str(int(v)) for v in arr)

    haystack = _pattern(csv_seq)
    feats, labels = [], []
    offset = None
    for pkl_path in pkl_files:
        with open(pkl_path, "rb") as f:
            _adj, y, x = pickle.load(f)
        y = np.asarray(y).astype(np.int64)
        if offset is None:
            offset = haystack.find(_pattern(y[:300]))
            if offset < 0:
                raise RuntimeError("elliptic 数据块与 classes.csv 无法对齐")
        seg = csv_seq[offset:offset + len(y)]
        if len(seg) != len(y) or not np.array_equal(seg, y):
            raise RuntimeError(
                f"{pkl_path.name} 与 classes.csv 对不齐 (offset={offset})")
        feats.append(np.asarray(x, dtype=np.float32))
        labels.append(y)
        offset += len(y)

    x_np = np.concatenate(feats, axis=0)
    y_np = np.concatenate(labels, axis=0)
    num_nodes = x_np.shape[0]
    start = haystack.find(_pattern(labels[0][:300]))
    tx_ids = cls["txId"].to_numpy()[start:start + num_nodes]
    if len(tx_ids) != num_nodes:
        raise RuntimeError("txId 数量与节点数不一致")

    id2loc = {int(t): i for i, t in enumerate(tx_ids)}
    edges = pd.read_csv(raw / "elliptic_txs_edgelist.csv")
    src = edges["txId1"].to_numpy()
    dst = edges["txId2"].to_numpy()
    keep = np.array([(int(a) in id2loc and int(b) in id2loc)
                     for a, b in zip(src, dst)], dtype=bool)
    src_keep = np.array([id2loc[int(a)] for a in src[keep]], dtype=np.int64)
    dst_keep = np.array([id2loc[int(b)] for b in dst[keep]], dtype=np.int64)
    edge_index = torch.from_numpy(np.stack([src_keep, dst_keep], axis=0))
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index, _ = remove_self_loops(edge_index)

    print(f"[elliptic] blocks={len(pkl_files)} nodes={num_nodes} "
          f"edges={edge_index.size(1)} feats={x_np.shape[1]} "
          f"labeled={int((y_np >= 0).sum())}")

    return Data(
        x=torch.from_numpy(x_np),
        y=torch.from_numpy(y_np),
        edge_index=edge_index,
    )


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
