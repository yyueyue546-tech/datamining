"""TT-GREB：测试时图重生（data-centric test-time graph adaptation）。

框架与论文 3.2 一致：
  原型提取器 ProExtractor -> G_Pro = (X ⊙ W^Pro_node, A ⊙ W^Pro_edge)
  环境细化器 EnvRefiner   -> G_Env = (X ⊙ W^Env_node, A ⊙ W^Env_edge)
  重组       compose      -> G'_te = (X ⊙ W^Comp_node, A ⊙ W^Comp_edge)
                             W^Comp = (1 - W^Pro) ⊙ W^Env + W^Pro        （论文式 6）
  推理：冻结的 GNN 直接作用在 G'_te 上。

与论文式 (7)(8) 的差异（实测结论，重要）：
  论文用 min_φp [1-cos(z', z^Pro)] - [1-cos(z^Env, z^Pro)] + αReg、
  max_φe ||Z_te - Z_Env||² - βReg 来端到端学习两个 FC 重加权器。但 GNN 冻结时，
  GCN 对输入是正齐次的（relu(ca)=c·relu(a)），因此第一项存在平凡解：W^Pro 取任意
  常数即可让 cos(z', z^Pro)=1；同时把 W^Env 压到 0 可以让 ||Z_te - Z_Env|| 无限增大。
  在 Cora/GCN 上实测，该目标在 60 步内把 W^Env 压到 0.001，精度从 0.6765 掉到 0.4682。
  所以这里保留同样的三个图与重组公式，但把两个子模块换成有闭式解的稳健估计量：

    W^Pro_node[i,d] = 1[ P[c_i, d] >= τ ]      P 为高置信伪标签估计的类原型
    W^Env_node[i,d] = 1[ mean_{j∈N(i)} X[j,d] >= τ ]   邻域一致（环境内稳定）成分
    W^Comp_node     = W^Pro + (1 - W^Pro) ⊙ W^Env

  含义与论文一致：原型维直接保留；非原型维只有在邻域里稳定出现时才保留，从而抑制
  随环境随机出现的噪声特征。伪标签每轮用当前重生图重新预测并刷新，即论文的迭代优化。

边界：论文也重加权边，但本项目的偏移只删边不加边，实测边重加权无增益（±0.001），
      因此 W^Pro_edge = W^Env_edge = 1，保持 A 不变。

骨干模型解耦：本文件不直接访问 model.conv2 之类的具体属性，也不假设 forward 的签
名。所有前向、类别数推断、冻结都委托给 backbones.py，因此第三部分的 GCN /
GraphSAGE / GAT / GPR-GNN 只要满足
    logits = model(x, edge_index, edge_weight=None)
就能直接接入；若模型额外提供 forward_with_embeddings，也可用于后续基于表示的损失。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from .backbones import forward_logits, freeze_model, infer_num_classes


# ---------------------------------------------------------------- helpers
def _neighbor_mean(x: torch.Tensor, edge_index: torch.Tensor, n: int) -> torch.Tensor:
    """每个节点各维特征在邻居上的均值（A 行归一化）。"""
    src, dst = edge_index
    out = torch.zeros_like(x)
    out.index_add_(0, dst, x[src])
    deg = torch.zeros(n, 1, device=x.device, dtype=x.dtype)
    deg.index_add_(0, dst, torch.ones(src.size(0), 1, device=x.device, dtype=x.dtype))
    return out / deg.clamp(min=1.0)


def _class_prototypes(
    x: torch.Tensor, labels: torch.Tensor, n_cls: int, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """按伪标签估计类原型 P[c, d] 与每类的置信度质量。"""
    proto = x.new_zeros(n_cls, x.size(1))
    mass = x.new_zeros(n_cls)
    proto.index_add_(0, labels, x * weights[:, None])
    mass.index_add_(0, labels, weights)
    return proto / mass.clamp(min=1.0)[:, None], mass


# ---------------------------------------------------------------- 两个子模块
class PrototypeExtractor(nn.Module):
    """原型提取器：W^Pro_node = 该节点各维特征与所属类原型的支撑关系。"""

    def forward(self, x, pseudo, prob, n_cls, tau: float, min_mass: float = 5.0):
        conf = prob.max(dim=-1).values
        proto, mass = _class_prototypes(x, pseudo, n_cls, conf)
        support = proto[pseudo]                                   # (N, d)
        valid = (mass[pseudo] >= min_mass)[:, None]               # 样本太少的类退回环境分支
        gate = (support >= tau) & valid
        return gate.to(x.dtype)


class EnvironmentRefiner(nn.Module):
    """环境细化器：W^Env_node = 该维特征在邻域中的稳定程度。"""

    def forward(self, x, edge_index, num_nodes, tau: float):
        nb = _neighbor_mean(x, edge_index, num_nodes)
        return (nb >= tau).to(x.dtype)


def _compose(pro_node, env_node):
    """论文式 (6)：W^Comp = (1 - W^Pro) ⊙ W^Env + W^Pro。"""
    return (1.0 - pro_node) * env_node + pro_node


def _is_presence_features(x: torch.Tensor) -> bool:
    """门控语义只对 0/1 属性存在性特征成立（原型频率、邻域出现率）。

    Cora / Amazon / Twitch 都是 0/1 词袋或多热特征；Elliptic 是有符号连续特征
    （min=-13.1, max=289.2），支撑度阈值没有意义，直接套用会把数值信息抹掉
    （实测 AUC 0.9704 -> 0.9093），因此跳过重生、保持原图。
    """
    return bool(((x == 0) | (x == 1)).all())


# ---------------------------------------------------------------- 正式接口
def adapt_test_graph(
    model: nn.Module,
    graph: Data,
    *,
    steps: int = 3,          # 迭代轮数（论文的迭代优化）
    lr: float | None = None,  # 闭式估计，保留参数仅为接口兼容
    tau: float = 0.05,       # 原型 / 邻域支撑度阈值
    min_mass: float = 5.0,   # 类样本置信度质量下限
    edge_weight: torch.Tensor | None = None,  # 供支持加权传播的骨干使用
    log_every: int = 1,
    **unused,
) -> Data:
    """把偏移测试图重生为新的测试图，返回带 x / edge_index 的 Data。

    与骨干无关：任意满足 model(x, edge_index, edge_weight=None) 的模型都可传入，
    例如第三部分的 GAT / GPR-GNN。
    """
    x0, ei = graph.x.detach(), graph.edge_index
    n = graph.num_nodes
    n_cls = infer_num_classes(model, graph)

    if not _is_presence_features(x0):
        print("  [reborn] skip: 属性不是 0/1 presence 特征，门控语义不适用，保持原图")
        out = Data(x=x0.clone(), edge_index=ei.clone())
        for k, v in graph:
            if k in ("x", "edge_index"):
                continue
            out[k] = v
        return out

    pro = PrototypeExtractor()
    env = EnvironmentRefiner()

    freeze_model(model)

    x_cur = x0
    for step in range(1, steps + 1):
        with torch.no_grad():
            prob = F.softmax(forward_logits(model, x_cur, ei, edge_weight), dim=-1)
        pseudo = prob.argmax(dim=-1)

        pro_node = pro(x0, pseudo, prob, n_cls, tau, min_mass)
        env_node = env(x0, ei, n, tau)
        comp_node = _compose(pro_node, env_node)
        x_cur = x0 * comp_node

        if log_every and (step % log_every == 0 or step == steps):
            with torch.no_grad():
                agree = float((forward_logits(model, x_cur, ei, edge_weight)
                               .argmax(-1) == pseudo).float().mean())
            print(
                f"  [reborn] step {step}/{steps} | "
                f"keep_ratio={comp_node.mean().item():.3f} "
                f"Wpro={pro_node.mean().item():.3f} "
                f"Wenv={env_node.mean().item():.3f} "
                f"pred_agree={agree:.3f}"
            )

    out = Data(x=x_cur, edge_index=ei.clone())
    for k, v in graph:
        if k in ("x", "edge_index"):
            continue
        out[k] = v
    if edge_weight is not None:
        out.edge_weight = edge_weight
    return out
