"""多 seed 验证两个便宜的提升点：
  - tau 默认值（0.05 / 0.08 / 0.10 / 0.15）
  - 门控后 logit 平滑（0 / 0.5）
  - MC-Dropout 伪标签（10 次前向平均）是否能顶上去
每个 seed 复用 results_v2/cora_seed{seed}/checkpoint.pt。
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
    taus = [0.05, 0.08, 0.10, 0.15]
    keys = (["erm"] + [f"t{t}" for t in taus] +
            ["t0.10_smooth", "mcd_t0.10", "mcd_t0.10_smooth"])
    agg = {k: [] for k in keys}

    for seed in seeds:
        clean = load_clean("cora", seed=seed)
        base, _, _ = split_118(clean, seed=seed, n_test_graphs=0)
        mask, y, n = base.test_mask, base.y, base.num_nodes
        model = build_model("gcn", base.x.size(1), 64, n_cls, 0.5)
        ckpt = Path(f"results_v2/cora_seed{seed}/checkpoint.pt")
        if not ckpt.exists():
            print(f"skip seed{seed}: no checkpoint")
            continue
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.eval()

        def logits(x, ei, mc=1):
            with torch.no_grad():
                if mc == 1:
                    return model(x, ei)
                model.train()
                acc = 0.0
                for _ in range(mc):
                    acc = acc + F.log_softmax(model(x, ei), -1)
                model.eval()
                return acc / mc

        def smooth(lg, ei, alpha=0.5, iters=2):
            src, dst = ei[0], ei[1]
            deg = torch.zeros(n).index_add_(0, dst, torch.ones(src.numel())).clamp(min=1)
            z = lg
            for _ in range(iters):
                z = (1 - alpha) * z + alpha * torch.zeros_like(z).index_add_(
                    0, dst, z[src] / deg[dst].unsqueeze(-1))
            return z

        def gate(x0, ei, tau, mc=1, rounds=3):
            x_cur = x0
            for _ in range(rounds):
                lg = logits(x_cur, ei, mc)
                prob = F.softmax(lg, -1)
                pseudo = prob.argmax(-1)
                P, _ = _class_prototypes(x0, pseudo, n_cls, prob.max(-1).values)
                nb = _neighbor_mean(x0, ei, n)
                x_cur = x0 * ((P[pseudo] >= tau) | (nb >= tau)).float()
            return x_cur

        per = {k: [] for k in keys}
        for g in make_test_set(clean, "cora", n_graphs=8, seed=seed):
            ei = g.edge_index
            def acc(lg):
                return float((lg[mask].argmax(-1) == y[mask]).float().mean())

            lg0 = logits(g.x, ei)
            per["erm"].append(acc(lg0))
            for t in taus:
                per[f"t{t}"].append(acc(logits(gate(g.x, ei, t), ei)))
            xs = gate(g.x, ei, 0.10)
            per["t0.10_smooth"].append(acc(smooth(logits(xs, ei), ei)))
            xm = gate(g.x, ei, 0.10, mc=10)
            per["mcd_t0.10"].append(acc(logits(xm, ei)))
            per["mcd_t0.10_smooth"].append(acc(smooth(logits(xm, ei), ei)))

        for k in keys:
            agg[k].append(sum(per[k]) / 8)
        print(f"seed{seed} ERM={agg['erm'][-1]:.4f} " +
              " ".join(f"{k}={agg[k][-1]:.4f}" for k in keys if k != "erm"))

    print("\n=== mean over seeds ===")
    E = sum(agg["erm"]) / len(agg["erm"])
    print(f"ERM mean={E:.4f}")
    for k in keys:
        if k == "erm":
            continue
        m = sum(agg[k]) / len(agg[k])
        wins = sum(1 for a, e in zip(agg[k], agg["erm"]) if a > e)
        print(f"  {k:<16} mean={m:.4f} delta={m-E:+.4f} wins={wins}/{len(agg[k])}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        seeds = [int(s) for s in sys.argv[1].split(",")]
    else:
        seeds = list(range(10))
    main(seeds)
