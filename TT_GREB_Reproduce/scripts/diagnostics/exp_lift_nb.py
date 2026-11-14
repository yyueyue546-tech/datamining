"""10 seed 对比：类内 lift、邻域 lift、两者同时（nb_lift 变体）。"""
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

N_CLS = DATASET_META["cora"]["num_classes"]


def class_lift(x, labels, weights):
    p = torch.zeros(N_CLS, x.size(1))
    mass = torch.zeros(N_CLS)
    p.index_add_(0, labels, x * weights.unsqueeze(-1))
    mass.index_add_(0, labels, weights)
    p = p / mass.clamp(min=1e-8).unsqueeze(-1)
    g = (x * weights.unsqueeze(-1)).sum(0) / weights.sum().clamp(min=1e-8)
    return p, p / g.clamp(min=1e-6).unsqueeze(0), g


def main(seeds):
    modes = ["base", "cls1.5", "nb1.3", "nb1.5", "both1.5"]
    agg = {k: [] for k in modes}
    erm_all = []

    for seed in seeds:
        clean = load_clean("cora", seed=seed)
        base, _, _ = split_118(clean, seed=seed, n_test_graphs=0)
        mask, y, n = base.test_mask, base.y, base.num_nodes
        model = build_model("gcn", base.x.size(1), 64, N_CLS, 0.5)
        ck = Path(f"results_v2/cora_seed{seed}/checkpoint.pt")
        if not ck.exists():
            continue
        model.load_state_dict(torch.load(ck, map_location="cpu"))
        model.eval()

        def logits(x, ei):
            with torch.no_grad():
                return model(x, ei)

        def gate(x0, ei, mode, tau=0.05, rounds=3):
            xc = x0
            for _ in range(rounds):
                pr = F.softmax(logits(xc, ei), -1)
                ps = pr.argmax(-1)
                P, L, g = class_lift(x0, ps, pr.max(-1).values)
                nb = _neighbor_mean(x0, ei, n)
                br = P[ps] >= tau
                if mode in ("cls1.5", "both1.5"):
                    br = br & (L[ps] >= 1.5)
                bnb = nb >= tau
                if mode in ("nb1.3", "nb1.5"):
                    s = 1.3 if mode == "nb1.3" else 1.5
                    bnb = bnb & (nb / g.clamp(min=1e-6).unsqueeze(0) >= s)
                if mode == "both1.5":
                    bnb = bnb & (nb / g.clamp(min=1e-6).unsqueeze(0) >= 1.5)
                xc = x0 * (br | bnb).float()
            return xc

        per = {k: [] for k in modes}
        e_list = []
        for gl in make_test_set(clean, "cora", n_graphs=8, seed=seed):
            ei = gl.edge_index
            lg0 = logits(gl.x, ei)
            e_list.append(float((lg0[mask].argmax(-1) == y[mask]).float().mean()))
            for m in modes:
                xg = gl.x if m == "base" else gate(gl.x, ei, m)
                per[m].append(float((logits(xg, ei)[mask].argmax(-1) == y[mask]).float().mean()))
        erm_all.append(sum(e_list) / 8)
        for k in modes:
            agg[k].append(sum(per[k]) / 8)
        print(f"seed{seed} ERM={erm_all[-1]:.4f} " +
              " ".join(f"{k}={agg[k][-1]:.4f}" for k in modes))

    E = sum(erm_all) / len(erm_all)
    print(f"\n=== {len(erm_all)} seeds ERM={E:.4f} ===")
    for k in modes:
        m = sum(agg[k]) / len(agg[k])
        wins = sum(1 for a, e in zip(agg[k], erm_all) if a > e)
        print(f"  {k:<8} mean={m:.4f} delta={m-E:+.4f} wins={wins}/{len(erm_all)}")


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else list(range(10))
    main(seeds)
