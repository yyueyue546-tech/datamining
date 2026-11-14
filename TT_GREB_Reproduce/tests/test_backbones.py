"""验证 TT-GREB 主函数可以承接任意满足统一接口的骨干模型。

不需要第三部分的真实代码：这里用几个"假模型"覆盖所有可能的签名差异：
  1. 只接受两个参数 forward(x, edge_index)
  2. 标准三参数 forward(x, edge_index, edge_weight=None)
  3. forward 永远返回 (logits, embeddings)
  4. 提供 forward_with_embeddings / reset_parameters，构造参数用 in_channels 命名

直接运行：python tests/test_backbones.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from tt_greb_reproduce.backbones import (
    build_backbone,
    forward_embeddings,
    forward_logits,
    infer_num_classes,
    freeze_model,
)
from tt_greb_reproduce.tt_greb import adapt_test_graph


def toy_graph(n=90, d=24, c=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(n) % c
    x = torch.zeros(n, d)
    # 每个类有自己的一组词，保证 0/1 presence 特征与可学的信号
    for cls in range(c):
        rows = torch.nonzero(y == cls, as_tuple=False).view(-1)
        words = torch.arange(cls * 6, cls * 6 + 6)
        x[rows.unsqueeze(1), words.unsqueeze(0)] = 1.0
    # 同类节点连成环，保证是正同质性图（否则两层 GCN 学不到信号）
    src, dst = [], []
    for cls in range(c):
        idx = torch.nonzero(y == cls, as_tuple=False).view(-1).tolist()
        for a, b in zip(idx, idx[1:] + idx[:1]):
            src += [a, b]
            dst += [b, a]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)


class TwoArgGCN(nn.Module):
    """最简签名：不接受 edge_weight。"""

    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        return self.conv2(self.conv1(x, edge_index).relu(), edge_index)


class WeightedGCN(nn.Module):
    """标准签名：接受 edge_weight。"""

    def __init__(self, in_dim, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        h = F.dropout(self.conv1(x, edge_index, edge_weight=edge_weight).relu(),
                      p=self.dropout, training=self.training)
        return self.conv2(h, edge_index, edge_weight=edge_weight)


class TupleGCN(nn.Module):
    """forward 永远返回元组。"""

    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, x, edge_index, edge_weight=None):
        h = self.conv1(x, edge_index).relu()
        return self.conv2(h, edge_index), h


class ExternalStyle(nn.Module):
    """仿第三部分模型：in_channels 命名 + forward_with_embeddings + reset_parameters。"""

    def __init__(self, in_channels, hidden_channels, num_classes, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        return self.conv2(self.conv1(x, edge_index).relu(), edge_index)

    def forward_with_embeddings(self, x, edge_index, edge_weight=None):
        h = self.conv1(x, edge_index).relu()
        return self.conv2(h, edge_index), h

    def reset_parameters(self):
        for m in self.modules():
            reset = getattr(m, "reset_parameters", None)
            if reset is not None and m is not self:
                reset()


def check(name, cls, expect_embeddings):
    graph = toy_graph()
    model = build_backbone("dummy", graph.x.size(1), 16, 3, 0.2,
                           module=f"__main__:{cls.__name__}")
    model.eval()
    logits = forward_logits(model, graph.x, graph.edge_index)
    assert logits.shape == (graph.num_nodes, 3), f"{name}: logits shape {logits.shape}"
    assert infer_num_classes(model, graph) == 3, f"{name}: num_classes 推断错误"

    if expect_embeddings:
        emb = forward_embeddings(model, graph.x, graph.edge_index)
        assert emb.shape[0] == graph.num_nodes, f"{name}: embeddings 形状错误"
    else:
        try:
            forward_embeddings(model, graph.x, graph.edge_index)
            raise AssertionError(f"{name}: 没有嵌入接口却没报错")
        except ValueError:
            pass

    # 先训几轮，让伪标签有意义，门控才会真正保留特征
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    model.train()
    for _ in range(80):
        opt.zero_grad()
        loss = F.cross_entropy(forward_logits(model, graph.x, graph.edge_index), graph.y)
        loss.backward()
        opt.step()
    model.eval()

    freeze_model(model)
    assert all(not p.requires_grad for p in model.parameters()), f"{name}: 冻结失败"

    reborn = adapt_test_graph(model, graph, steps=2)
    assert reborn.x.shape == graph.x.shape
    assert bool(((reborn.x == 0) | (reborn.x == 1)).all())
    assert float(reborn.x.mean()) > 0.0, f"{name}: 门控把所有特征都删掉了"
    print(f"  [ok] {name}: 类别数推断 / 前向 / 冻结 / 重生流程通过")


def main():
    print("1) 模型接口适配")
    check("两参数 forward", TwoArgGCN, expect_embeddings=False)
    check("带 edge_weight", WeightedGCN, expect_embeddings=False)
    check("元组输出", TupleGCN, expect_embeddings=False)
    check("外部风格模型", ExternalStyle, expect_embeddings=True)

    print("2) 不支持的 edge_weight 必须给出明确报错")
    graph = toy_graph()
    model = TwoArgGCN(graph.x.size(1), 16, 3)
    try:
        forward_logits(model, graph.x, graph.edge_index, edge_weight=torch.ones(graph.edge_index.size(1)))
        raise AssertionError("应抛 TypeError")
    except TypeError as exc:
        print(f"  [ok] {str(exc)[:60]}...")

    print("3) 未知骨干 / 缺失模块必须有提示")
    for name, module in (("unknown", None), ("gat", None)):
        try:
            build_backbone(name, 24, 16, 3, module=module)
            print(f"  [!!] {name} 竟然构建成功")
        except (ValueError, ModuleNotFoundError) as exc:
            print(f"  [ok] {name}: {str(exc)[:70]}...")

    print("\n全部通过：TT-GREB 主函数可以承接 GCN / GraphSAGE / GAT / GPR-GNN。")


if __name__ == "__main__":
    main()
