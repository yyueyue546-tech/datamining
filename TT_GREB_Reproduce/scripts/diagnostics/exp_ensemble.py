"""稳健性检查：把 10 个 seed 的 GCN 做 logit 集成。

目的：排除“收益只是因为单个模型太弱”的质疑。
若集成后 ERM 本身变强、TT 的增量为正但变小，则说明增益来自分布偏移本身，
而不是模型容量不足。
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


def main(seed=0, members=range(10)):
    n_cls = DATASET_META["cora"]["num_classes"]
    clean = load_clean("cora", seed=seed)
    base, _, _ = split_118(clean, seed=seed, n_test_graphs=0)
    mask, y, n = base.test_mask, base.y, base.num_nodes

    models = []
    for m in members:
        ck = Path(f"results_v2/cora_seed{m}/checkpoint.pt")
        if not ck.exists():
            continue
        mdl = build_model("gcn", base.x.size(1), 64, n_cls, 0.5)
        mdl.load_state_dict(torch.load(ck, map_location="cpu"))
        mdl.eval()
        models.append(mdl)
    print(f"ensemble size = {len(models)}")

    def ens_logits(x, ei):
        with torch.no_grad():
            return torch.stack([F.log_softmax(m(x, ei), -1) for m in models], 0).mean(0)

    def gate(x0, ei, tau=0.05, rounds=3):
        x_cur = x0
        for _ in range(rounds):
            lg = ens_logits(x_cur, ei)
            prob = F.softmax(lg, -1)
            pseudo = prob.argmax(-1)
            P, _ = _class_prototypes(x0, pseudo, n_cls, prob.max(-1).values)
            nb = _neighbor_mean(x0, ei, n)
            x_cur = x0 * ((P[pseudo] >= tau) | (nb >= tau)).float()
        return x_cur

    def acc(lg):
        return float((lg[mask].argmax(-1) == y[mask]).float().mean())

    clean_acc = acc(ens_logits(clean.x, clean.edge_index))
    em, tt = [], []
    for i, g in enumerate(make_test_set(clean, "cora", n_graphs=8, seed=seed)):
        e = acc(ens_logits(g.x, g.edge_index))
        t = acc(ens_logits(gate(g.x, g.edge_index), g.edge_index))
        em.append(e); tt.append(t)
        print(f"  T{i} erm={e:.4f} tt={t:.4f} d={t-e:+.4f}")
    E = sum(em) / 8; T = sum(tt) / 8
    print(f"\nensemble clean={clean_acc:.4f} ERM={E:.4f} TT={T:.4f} delta={T-E:+.4f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
