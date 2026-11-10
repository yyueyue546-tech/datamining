"""
TT-GREB 简化复现版（TTA 语义修正版）

核心改动（相对于之前的精简版）：
  1) L_pro 不再锚定 z_te0（"别改预测"），改为伪标签软锚 + 高置信保持
  2) 加入熵最小化，让适应朝"更自信"的方向走
  3) 加入 embedding drift 正则，防止图塌陷
  4) per-node 重加权（fc_node 输出 (N, d)），保留已验证的修复
  5) edge_weight 真正透传到返回的 Data，供 model 使用（若 model 支持）

接口与 run_cora_gcn.py 保持兼容：
  adapt_test_graph(model, graph, *, steps, lr, ...) -> Data
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data


# ---------------------------------------------------------------- adapters
class _Reweighter(nn.Module):
    """输出 per-node (N,d) 与 per-edge (E,) 的权重。"""

    def __init__(self, d: int):
        super().__init__()
        self.fc_node = nn.Sequential(nn.Linear(d, d), nn.Sigmoid())
        self.fc_edge = nn.Sequential(nn.Linear(2 * d, 1), nn.Sigmoid())

    def forward(self, x, edge_index):
        W_node = self.fc_node(x)                             # (N, d)
        src, dst = edge_index
        W_edge = self.fc_edge(
            torch.cat([x[src], x[dst]], dim=-1)).squeeze(-1)  # (E,)
        return W_node, W_edge


class PrototypeExtractor(_Reweighter):
    pass


class EnvironmentRefiner(_Reweighter):
    pass


# ---------------------------------------------------------------- helpers
def _reg(W, lam: float = 0.5):
    return ((W - lam) ** 2).mean()


def _mse(a, b):
    return ((a - b) ** 2).mean()


def _entropy(logits, eps: float = 1e-8):
    p = F.softmax(logits, dim=-1)
    return -(p * (p + eps).log()).sum(-1)


def _call_model(model, x, edge_index, edge_weight=None):
    """
    兼容两种 model 签名：
      model(x, edge_index)
      model(x, edge_index, edge_weight=...)
    返回 logits（不取 embedding）。
    """
    if edge_weight is None:
        return model(x, edge_index)
    try:
        return model(x, edge_index, edge_weight=edge_weight)
    except TypeError:
        return model(x, edge_index)


def _compose(pro_node, pro_edge, env_node, env_edge):
    """GREB 组合规则：pro 为主，env 补足。"""
    node_w = pro_node + (1.0 - pro_node) * env_node      # (N, d)
    edge_w = pro_edge + (1.0 - pro_edge) * env_edge      # (E,)
    return node_w, edge_w


# ---------------------------------------------------------------- 正式接口
def adapt_test_graph(
    model: nn.Module,
    graph: Data,
    *,
    steps: int = 60,
    lr: float = 5e-3,
    # 损失权重
    w_pl: float = 0.3,          # 伪标签（main）
    w_ent: float = 0.5,         # 熵最小化
    target_mode: str = "self",  # 伪标签 target: 'self'=当前模型软预测, 'erm'=初始干净图预测
    w_drift: float = 0.02,      # embedding 漂移正则
    w_reg: float = 0.03,        # 权重正则
    w_env: float = 0.4,         # 环境分支辅助（dist 拉近）
    # 其余超参
    lam: float = 0.7,
    conf_thresh: float = 0.9,
    update_every: int = 20,      # 锚点/伪标签动态刷新间隔
    drift_thresh: float = 0.3,
    log_every: int = 20,
    **unused,
):
    device = graph.x.device
    d = graph.x.size(-1)

    pro = PrototypeExtractor(d).to(device)
    env = EnvironmentRefiner(d).to(device)

    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    opt = torch.optim.Adam([
        {"params": list(pro.parameters()), "lr": lr},
        {"params": list(env.parameters()), "lr": lr * 5.0},
    ])

    x0, ei = graph.x, graph.edge_index

    # ---- 初始 embedding + 伪标签 ----
    with torch.no_grad():
        z_te0 = _call_model(model, x0, ei)
        prob0 = F.softmax(z_te0, dim=-1)
        conf0, pseudo0 = prob0.max(dim=-1)

        # 只取置信度前 30% 作为锚点（避免锚点过多导致 TTA 无空间）
        # 锚点/探索节点只在测试节点内选取（避免训练节点占满 top-k）
        if hasattr(graph, "test_mask") and graph.test_mask is not None:
            pool = graph.test_mask.bool()
        else:
            pool = torch.ones_like(conf0, dtype=torch.bool)
        n_pool = int(pool.sum().item())
        k = max(10, int(0.20 * n_pool))
        thr = torch.topk(conf0[pool], k).values.min()
        keep = pool & (conf0 >= thr)
        explore = pool & (~keep)
        print(f"DEBUG: total_node={conf0.numel()}, test_mask_sum={n_pool}, k={k}, keep={int(keep.sum())}")

    z_te0_det = z_te0.detach()

    # ---- 动态刷新锚点与伪标签 ----
    def _refresh_anchor(logits_cur, pool_mask):
        with torch.no_grad():
            conf_cur = torch.softmax(logits_cur.detach(), dim=-1).max(dim=-1).values
            pseudo_cur = logits_cur.detach().argmax(dim=-1)
            kk = max(10, int(0.20 * int(pool_mask.sum().item())))
            thr_cur = torch.topk(conf_cur[pool_mask], kk).values.min()
            keep_cur = pool_mask & (conf_cur >= thr_cur)
            explore_cur = pool_mask & (~keep_cur)
        return conf_cur, pseudo_cur, keep_cur, explore_cur

    # ---- 训练循环 ----
    for step in range(1, steps + 1):
        opt.zero_grad()

        # ---- 每 update_every 步刷新锚点（修复：之前从未调用）----
        if step > 1 and (step - 1) % update_every == 0:
            with torch.no_grad():
                logits_cur_dbg = _call_model(model, x0 * node_w if 'node_w' in dir() else x0, ei)
            conf_cur, pseudo_cur, keep_new, explore_new = _refresh_anchor(logits_cur_dbg, pool)
            keep, explore = keep_new, explore_new
            if step % log_every == 0:
                print(f"  [anchor-refresh] step {step}: keep={int(keep.sum())}/{int(pool.sum())}")

        pro_node, pro_edge = pro(x0, ei)
        env_node, env_edge = env(x0, ei)

        node_w, edge_w = _compose(pro_node, pro_edge, env_node, env_edge)

        # 重加权后的图
        x_w = x0 * node_w
        logits_new = _call_model(model, x_w, ei, edge_weight=edge_w)

        # 环境分支（辅助）
        x_env = x0 * env_node
        logits_env = _call_model(model, x_env, ei)

        # ---------- 1) 伪标签损失（仅 keep 区，不再污染 explore 区） ----------
        if keep.sum() > 0:
            with torch.no_grad():
                if target_mode == 'self':
                    soft0 = torch.softmax(logits_new.detach(), dim=-1)
                    w_node = (soft0.max(dim=-1).values[keep] ** 2).clamp(min=1e-6)
                else:
                    soft0 = torch.softmax(z_te0_det, dim=-1)
                    w_node = (conf0[keep] ** 2).clamp(min=1e-6)
            logp = F.log_softmax(logits_new[keep], dim=-1)
            L_pl = -(w_node * (soft0[keep] * logp).sum(-1)).sum() / (w_node.sum() + 1e-8)
        else:
            L_pl = torch.zeros((), device=device)

        # ---------- 2) 熵最小化（仅 keep 区，不对不确定节点压熵） ----------
        if keep.sum() > 0:
            ent = _entropy(logits_new)
            L_ent = ent[keep].mean()
        else:
            L_ent = torch.zeros((), device=device)

        # ---------- 3) embedding drift 正则 ----------
        drift = ((logits_new - z_te0_det) ** 2).mean()
        L_drift = torch.relu(drift - drift_thresh)

        # ---------- 4) 环境辅助：拉近 env 输出与原始 ----------
        dist_env = torch.norm(z_te0_det - logits_env, p=2, dim=-1).mean()
        L_env = dist_env

        # ---------- 5) 权重正则 ----------
        L_reg = _reg(pro_node, lam) + _reg(pro_edge, lam) \
              + _reg(env_node, 0.35) + _reg(env_edge, 0.35)

        loss = (
            w_pl * L_pl
            + w_ent * L_ent
            + w_drift * L_drift
            + w_env * L_env
            + w_reg * L_reg
        )

        loss.backward()
        opt.step()

        if log_every and (step % log_every == 0 or step == steps):
            with torch.no_grad():
                ent_all = _entropy(logits_new).mean().item()
                drift_val = ((logits_new - z_te0_det) ** 2).mean().item()
                L_pl_val = L_pl.item() if torch.is_tensor(L_pl) else 0.0
                L_ent_val = L_ent.item() if torch.is_tensor(L_ent) else 0.0
                ent_each = _entropy(logits_new)
                ent_anchor = ent_each[keep].mean().item() if keep.sum() > 0 else 0.0
                ent_explore = ent_each[explore].mean().item() if explore.sum() > 0 else 0.0
            print(
                f"  [reborn] step {step:3d}/{steps} | "
                f"L_pl={L_pl_val:+.4f} L_ent={L_ent_val:+.4f} | "
                f"ent={ent_all:.4f}(anc={ent_anchor:.3f},exp={ent_explore:.3f}) drift={drift_val:.4f} | "
                f"|W_pro|={pro_node.mean().item():.3f} "
                f"|W_env|={env_node.mean().item():.3f} "
                f"keep={keep.sum().item()}/{keep.numel()}"
            )

    # ---- 最终 reborn 图 ----
    with torch.no_grad():
        pro_node, pro_edge = pro(x0, ei)
        env_node, env_edge = env(x0, ei)
        node_w, edge_w = _compose(pro_node, pro_edge, env_node, env_edge)

    x_reborn = x0 * node_w

    out = Data(x=x_reborn, edge_index=ei.clone())
    for k, v in graph:
        if k in ("x", "edge_index"):
            continue
        out[k] = v
    out.edge_weight = edge_w
    return out






