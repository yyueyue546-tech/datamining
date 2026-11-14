"""第四轮：伪标签质量是唯一瓶颈（oracle +0.0499 vs 实际 +0.0306）。

测两件事：
  1) 用邻居 logit 平滑得到更准的伪标签，再门控（评估仍用原始 logits）
  2) 直接对 logits 做归一化图平滑后预测（Correct&Smooth 风格），看绝对精度上限
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F

from tt_greb_reproduce.data import load_clean, split_118
from tt_greb_reproduce.models import build_model
from tt_greb_reproduce.shift import make_test_set
from tt_greb_reproduce.paths import DATASET_META
from tt_greb_reproduce.tt_greb import _neighbor_mean, _class_prototypes


def main(seed=0):
    n_cls = DATASET_META["cora"]["num_classes"]
    clean = load_clean("cora", seed=seed)
    base, _, _ = split_118(clean, seed=seed, n_test_graphs=0)
    mask, y, n = base.test_mask, base.y, base.num_nodes
    model = build_model("gcn", base.x.size(1), 64, n_cls, 0.5)
    model.load_state_dict(torch.load(f"results_v2/cora_seed{seed}/checkpoint.pt", map_location="cpu"))
    model.eval()

    def logits(x, ei):
        with torch.no_grad():
            return model(x, ei)

    def smooth(logit, ei, alpha=0.5, iters=2):
        """归一化邻接平均的 logit 平滑。"""
        src, dst = ei[0], ei[1]
        deg = torch.zeros(n).index_add_(0, dst, torch.ones(src.numel()))
        deg = deg.clamp(min=1)
        z = logit
        for _ in range(iters):
            agg = torch.zeros_like(z).index_add_(0, dst, z[src] / deg[dst].unsqueeze(-1))
            z = (1 - alpha) * z + alpha * agg
        return z

    def gate(x0, ei, tau=0.10, lp_alpha=0.0, rounds=3):
        x_cur = x0
        for _ in range(rounds):
            lg = logits(x_cur, ei)
            if lp_alpha > 0:
                lg = smooth(lg, ei, alpha=lp_alpha)
            prob = F.softmax(lg, -1)
            pseudo = prob.argmax(-1)
            P, _ = _class_prototypes(x0, pseudo, n_cls, prob.max(-1).values)
            nb = _neighbor_mean(x0, ei, n)
            x_cur = x0 * ((P[pseudo] >= tau) | (nb >= tau)).float()
        return x_cur

    variants = ["erm", "gate", "gate_lp0.3", "gate_lp0.5", "gate_lp0.7",
                "smooth0.3", "smooth0.5", "smooth0.7", "gate_then_smooth0.5"]
    res = {k: [] for k in variants}
    for g in make_test_set(clean, "cora", n_graphs=8, seed=seed):
        ei = g.edge_index
        def acc(lg):
            return float((lg[mask].argmax(-1) == y[mask]).float().mean())

        lg0 = logits(g.x, ei)
        res["erm"].append(acc(lg0))
        for a in (0.3, 0.5, 0.7):
            res[f"smooth{a}"].append(acc(smooth(lg0, ei, alpha=a, iters=2)))

        xg = gate(g.x, ei)
        res["gate"].append(acc(logits(xg, ei)))
        for a in (0.3, 0.5, 0.7):
            res[f"gate_lp{a}"].append(acc(logits(gate(g.x, ei, lp_alpha=a), ei)))
        res["gate_then_smooth0.5"].append(acc(smooth(logits(xg, ei), ei, 0.5, 2)))

    E = sum(res["erm"]) / 8
    print(f"seed{seed} ERM={E:.4f}")
    for k in variants:
        if k == "erm":
            continue
        v = res[k]
        m = sum(v) / 8
        print(f"  {k:<22} mean={m:.4f} delta={m-E:+.4f} per={[round(a,3) for a in v]}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
