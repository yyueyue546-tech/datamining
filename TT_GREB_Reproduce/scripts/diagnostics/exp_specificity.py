"""类别特异性筛选：伪噪声词在各类中频率接近全局频率。

对词 d 计算 lift = P(d | class c) / P(d)。
真实类指示词的 lift 明显 > 1；均匀撒入的噪声词 lift ≈ 1。
门控改为：(类内频率 >= tau 且 lift >= s) 或 (邻域均值 >= tau)。
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
from tt_greb_reproduce.tt_greb import _neighbor_mean


def proto_lift(x, labels, n_cls, weights):
    """返回 (类内频率 P, lift)。"""
    p = torch.zeros(n_cls, x.size(1))
    mass = torch.zeros(n_cls)
    p.index_add_(0, labels, x * weights.unsqueeze(-1))
    mass.index_add_(0, labels, weights)
    p = p / mass.clamp(min=1e-8).unsqueeze(-1)
    glob = (x * weights.unsqueeze(-1)).sum(0) / weights.sum().clamp(min=1e-8)
    lift = p / glob.clamp(min=1e-6).unsqueeze(0)
    return p, lift


def main(seeds):
    n_cls = DATASET_META["cora"]["num_classes"]
    lifts = [0.0, 1.2, 1.5, 2.0, 3.0]
    keys = [f"lift{s}" for s in lifts]
    agg = {k: [] for k in keys}
    erm_all = []

    for seed in seeds:
        clean = load_clean("cora", seed=seed)
        base, _, _ = split_118(clean, seed=seed, n_test_graphs=0)
        mask, y, n = base.test_mask, base.y, base.num_nodes
        model = build_model("gcn", base.x.size(1), 64, n_cls, 0.5)
        ck = Path(f"results_v2/cora_seed{seed}/checkpoint.pt")
        if not ck.exists():
            continue
        model.load_state_dict(torch.load(ck, map_location="cpu"))
        model.eval()

        def logits(x, ei):
            with torch.no_grad():
                return model(x, ei)

        def gate(x0, ei, s, tau=0.05, rounds=3):
            x_cur = x0
            for _ in range(rounds):
                prob = F.softmax(logits(x_cur, ei), -1)
                pseudo = prob.argmax(-1)
                P, L = proto_lift(x0, pseudo, n_cls, prob.max(-1).values)
                nb = _neighbor_mean(x0, ei, n)
                branch = (P[pseudo] >= tau)
                if s > 0:
                    branch = branch & (L[pseudo] >= s)
                x_cur = x0 * (branch | (nb >= tau)).float()
            return x_cur

        per = {k: [] for k in keys}
        e_list = []
        for g in make_test_set(clean, "cora", n_graphs=8, seed=seed):
            ei = g.edge_index
            lg0 = logits(g.x, ei)
            e_list.append(float((lg0[mask].argmax(-1) == y[mask]).float().mean()))
            for s in lifts:
                xg = gate(g.x, ei, s)
                per[f"lift{s}"].append(
                    float((logits(xg, ei)[mask].argmax(-1) == y[mask]).float().mean()))
        erm_all.append(sum(e_list) / 8)
        for k in per:
            agg[k].append(sum(per[k]) / 8)
        print(f"seed{seed} ERM={erm_all[-1]:.4f} " +
              " ".join(f"{k}={agg[k][-1]:.4f}" for k in per))

    E = sum(erm_all) / len(erm_all)
    print(f"\n=== {len(erm_all)} seeds, ERM={E:.4f} ===")
    for k in agg:
        m = sum(agg[k]) / len(agg[k])
        wins = sum(1 for a, e in zip(agg[k], erm_all) if a > e)
        print(f"  {k:<10} mean={m:.4f} delta={m-E:+.4f} wins={wins}/{len(erm_all)}")


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0, 1, 2]
    main(seeds)
