"""给 8 张测试图注入分布偏移。

设计原则：
  - 每张图用不同随机种子，模拟 8 个独立偏移环境
  - 特征噪声 + 边扰动可组合
  - 噪声强度按数据集缩放（Cora 的 sigma=0.3 不能直接套到 arxiv）
"""
from __future__ import annotations

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


# 各数据集的基准噪声强度（经验起点，后续网格搜）
SHIFT_PRESETS = {
    "cora":   {"sigma": 0.20, "drop": 0.12, "add": 0.0},
    "amazon": {"sigma": 0.3,  "drop": 0.2, "add": 0.0},
    "twitch": {"sigma": 0.15, "drop": 0.15, "add": 0.0},
    "ogb":    {"sigma": 0.2,  "drop": 0.15, "add": 0.0},
    # Elliptic 特征是标准化后的连续值，不能按 0/1 翻转，改用高斯噪声
    "elliptic": {"sigma": 0.20, "drop": 0.10, "add": 0.0, "mode": "gaussian"},
}


def add_feature_noise(graph: Data, sigma: float, seed: int,
                      mode: str = "binary") -> Data:
    gen = torch.Generator().manual_seed(seed)
    x = graph.x
    if sigma <= 0:
        out = Data(x=x.clone(), edge_index=graph.edge_index.clone())
        for k, v in graph:
            if k in ("x", "edge_index"):
                continue
            setattr(out, k, v)
        return out

    if mode == "gaussian":
        # 连续特征：加性高斯噪声
        x_new = x + sigma * torch.randn(x.shape, generator=gen)
        out = Data(x=x_new, edge_index=graph.edge_index.clone())
        for k, v in graph:
            if k in ("x", "edge_index"):
                continue
            setattr(out, k, v)
        return out

    # 0/1 稀疏词袋的语义保持型扰动：翻转词的存在性
    rand = torch.rand(x.shape, generator=gen)

    # 原本为 1 的位置：以 sigma 概率翻成 0（丢词）
    mask_one = (x.abs() > 1e-8)
    flip_one = mask_one & (rand < sigma)

    # 原本为 0 的位置：以 sigma/20 概率翻成 1（加噪词，保守）
    mask_zero = ~mask_one
    flip_zero = mask_zero & (rand < sigma / 20.0)

    x_new = x.clone()
    x_new[flip_one] = 0.0
    x_new[flip_zero] = 1.0

    out = Data(x=x_new, edge_index=graph.edge_index.clone())
    for k, v in graph:
        if k in ("x", "edge_index"):
            continue
        setattr(out, k, v)
    return out


def _rebuild(x, edge_index, old: Data) -> Data:
    """用新的 x / edge_index 重建图，并原样带上可选 mask。"""
    out = Data(x=x, edge_index=edge_index, y=old.y.clone(),
               num_nodes=old.num_nodes)
    for key in ("train_mask", "val_mask", "test_mask"):
        mask = getattr(old, key, None)
        if mask is not None:
            out[key] = mask.clone()
    return out


def drop_edges(graph: Data, ratio: float, seed: int) -> Data:
    if ratio <= 0:
        return graph.clone() if hasattr(graph, "clone") else graph
    gen = torch.Generator().manual_seed(seed + 10_000)
    ei = graph.edge_index
    num_edges = ei.size(1)

    # 对称丢边：无向边 (min,max) 只在规范形式上掷一次骰子，保证对称性不破坏
    u_list = ei[0].tolist()
    v_list = ei[1].tolist()
    key_mask = {}
    keep = torch.zeros(num_edges, dtype=torch.bool)
    for k in range(num_edges):
        a = u_list[k]
        b = v_list[k]
        key = (a, b) if a <= b else (b, a)
        if key not in key_mask:
            key_mask[key] = bool(torch.rand(1, generator=gen).item() >= ratio)
        keep[k] = key_mask[key]
    return _rebuild(graph.x.clone(), graph.edge_index[:, keep], graph)


def add_random_edges(graph: Data, ratio: float, seed: int) -> Data:
    if ratio <= 0:
        return graph
    gen = torch.Generator().manual_seed(seed + 20_000)
    num_edges = graph.edge_index.size(1)
    num_add = int(num_edges * ratio)
    n = graph.num_nodes
    src = torch.randint(0, n, (num_add,), generator=gen)
    dst = torch.randint(0, n, (num_add,), generator=gen)
    extra = torch.stack([src, dst], dim=0)
    edge_index = torch.cat([graph.edge_index, extra], dim=1)
    return _rebuild(graph.x.clone(), edge_index, graph)


def make_test_graph(base: Data, dataset: str, index: int, seed: int,
                    sigma=None, drop=None, add=None) -> Data:
    """生成第 index 张测试图（0..7），每张用独立种子。"""
    preset = SHIFT_PRESETS[dataset]
    sigma = preset["sigma"] if sigma is None else sigma
    drop = preset["drop"] if drop is None else drop
    add = preset["add"] if add is None else add
    mode = preset.get("mode", "binary")

    s = seed * 1000 + index          # 保证不同 (seed, index) 组合互不重复
    g = add_feature_noise(base, sigma, seed=s, mode=mode)
    g = drop_edges(g, drop, seed=s)
    g = add_random_edges(g, add, seed=s)
    return g


def make_test_set(base: Data, dataset: str, n_graphs: int = 8,
                  seed: int = 0, **kw) -> list[Data]:
    """生成 n_graphs 个强度递增的测试图，模拟 '多个偏移环境'。

    强度 scale 从 0.2 线性升到 1.8 乘以 preset 值。
    第 0 张 mild，最后一张 strong。
    """
    preset = SHIFT_PRESETS[dataset]
    base_sigma = kw.pop("sigma", preset["sigma"])
    base_drop = kw.pop("drop", preset["drop"])
    base_add = kw.pop("add", preset["add"])

    scales = [0.2 + 1.6 * i / max(1, n_graphs - 1) for i in range(n_graphs)]
    graphs = []
    for i, sc in enumerate(scales):
        g = make_test_graph(base, dataset, i, seed,
                            sigma=base_sigma * sc,
                            drop=base_drop * sc,
                            add=base_add * sc, **kw)
        graphs.append(g)
    return graphs
