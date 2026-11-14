"""选择性重生：只对“模型自己不自信”的节点做门控。

集成实验显示弱偏移图（T0-T2）门控会掉点，强偏移图收益很大。
因此用 per-node 置信度做开关：conf >= thr 的节点保留原特征。
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


def main(seeds):
    n_cls = DATASET_META["cora"]["num_classes"]
    thrs = [1.01, 0.95, 0.90, 0.80, 0.70]   # 1.01 = 全量重生（当前做法）
    agg = {f"sel{t:.2f}": [] for t in thrs}
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

        def gate(x0, ei, tau=0.05, rounds=3):
            x_cur = x0
            for _ in range(rounds):
                prob = F.softmax(logits(x_cur, ei), -1)
                pseudo = prob.argmax(-1)
                P, _ = _class_prototypes(x0, pseudo, n_cls, prob.max(-1).values)
                nb = _neighbor_mean(x0, ei, n)
                x_cur = x0 * ((P[pseudo] >= tau) | (nb >= tau)).float()
            return x_cur

        per = {f"sel{t:.2f}": [] for t in thrs}
        e_list = []
        for g in make_test_set(clean, "cora", n_graphs=8, seed=seed):
            ei = g.edge_index
            lg0 = logits(g.x, ei)
            conf = F.softmax(lg0, -1).max(-1).values
            e_list.append(float((lg0[mask].argmax(-1) == y[mask]).float().mean()))
            xg = gate(g.x, ei)
            for t in thrs:
                sel = (conf < t).float().unsqueeze(-1)
                xs = sel * xg + (1 - sel) * g.x
                per[f"sel{t:.2f}"].append(
                    float((logits(xs, ei)[mask].argmax(-1) == y[mask]).float().mean()))
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
        print(f"  {k} mean={m:.4f} delta={m-E:+.4f} wins={wins}/{len(erm_all)}")


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else list(range(10))
    main(seeds)
