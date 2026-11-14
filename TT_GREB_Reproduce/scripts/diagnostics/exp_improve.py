"""在 Cora 上比较几种在现有重生门控之上的改进方案。"""
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

    def new_ei(x_new, ei, k=5, thr=0.5):
        """按特征余弦相似度补边（只补高相似且原本不相连的对）。"""
        xn = F.normalize(x_new, dim=-1)
        sim = xn @ xn.t()
        sim.fill_diagonal_(-1)
        vals, idx = sim.topk(k, dim=-1)
        keep = vals >= thr
        rows, cols = torch.nonzero(keep, as_tuple=True)
        src = idx[rows, cols]
        dst = rows
        exist = set(zip(ei[0].tolist(), ei[1].tolist()))
        add = [(int(a), int(b)) for a, b in zip(src.tolist(), dst.tolist())
               if (a, b) not in exist and (b, a) not in exist]
        if not add:
            return ei
        add = torch.tensor(add).t()
        return torch.cat([ei, add], dim=1)

    def reborn(x0, ei, tau=0.05, rounds=3, update_from="x0", fill=False):
        x_cur = x0
        comp = None
        for _ in range(rounds):
            prob = F.softmax(logits(x_cur, ei), -1)
            pseudo = prob.argmax(-1)
            src_x = x0 if update_from == "x0" else x_cur
            P, _ = _class_prototypes(src_x, pseudo, n_cls, prob.max(-1).values)
            nb = _neighbor_mean(src_x, ei, n)
            comp = ((P[pseudo] >= tau) | (nb >= tau)).float()
            x_cur = x0 * comp
        if fill:
            prob = F.softmax(logits(x_cur, ei), -1)
            pseudo = prob.argmax(-1)
            P, _ = _class_prototypes(x0, pseudo, n_cls, prob.max(-1).values)
            x_cur = torch.maximum(x_cur, ((P[pseudo] >= 0.5) & (x0 < 1e-8)).float())
        return x_cur, comp

    graphs = make_test_set(clean, "cora", n_graphs=8, seed=seed)
    res = {k: [] for k in ("erm", "base", "iter_src", "fill", "edge", "ens50", "ens30",
                           "base+edge", "iter+edge")}
    for g in graphs:
        lg_e = logits(g.x, g.edge_index)
        res["erm"].append(float((lg_e[mask].argmax(-1) == y[mask]).float().mean()))

        xb, _ = reborn(g.x, g.edge_index)
        res["base"].append(float((logits(xb, g.edge_index)[mask].argmax(-1) == y[mask]).float().mean()))

        xi, _ = reborn(g.x, g.edge_index, update_from="x_cur")
        res["iter_src"].append(float((logits(xi, g.edge_index)[mask].argmax(-1) == y[mask]).float().mean()))

        xf, _ = reborn(g.x, g.edge_index, fill=True)
        res["fill"].append(float((logits(xf, g.edge_index)[mask].argmax(-1) == y[mask]).float().mean()))

        ei_add = new_ei(xb, g.edge_index)
        res["edge"].append(float((logits(xb, ei_add)[mask].argmax(-1) == y[mask]).float().mean()))
        res["base+edge"].append(res["edge"][-1])
        xie, _ = reborn(g.x, g.edge_index, update_from="x_cur")
        ei2 = new_ei(xie, g.edge_index)
        res["iter+edge"].append(float((logits(xie, ei2)[mask].argmax(-1) == y[mask]).float().mean()))

        lg_b = logits(xb, g.edge_index)
        for w, key in ((0.5, "ens50"), (0.3, "ens30")):
            mix = w * lg_e + (1 - w) * lg_b
            res[key].append(float((mix[mask].argmax(-1) == y[mask]).float().mean()))

    E = sum(res["erm"]) / 8
    print(f"seed{seed}  ERM mean={E:.4f}")
    for k, v in res.items():
        if k == "erm":
            continue
        m = sum(v) / 8
        print(f"  {k:<12} mean={m:.4f} delta={m-E:+.4f} per={[round(a,3) for a in v]}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
