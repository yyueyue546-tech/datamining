"""第三轮：软门控 / 融合方式 / 伪标签误差上界。

关键问题：剩下 9 个点是被“伪标签错”卡住，还是被“门控形式”卡住？
  oracle_proto : 用真实标签算原型（只用来看上界，不可用于方法）
  soft_pw      : x * P[伪标签] 软权重
  soft_norm    : 软权重并按行回到 clean 均值尺度
  and          : 原型 AND 邻域（高精度）
  vote         : 3 轮门控多数投票
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
    tgt_sum = float(clean.x.sum(1).mean())
    model = build_model("gcn", base.x.size(1), 64, n_cls, 0.5)
    model.load_state_dict(torch.load(f"results_v2/cora_seed{seed}/checkpoint.pt", map_location="cpu"))
    model.eval()

    def logits(x, ei):
        with torch.no_grad():
            return model(x, ei)

    def one_round(x0, ei, src_x, labels, tau=0.10, weight=None):
        prob = F.softmax(logits(src_x, ei), -1)
        P, _ = _class_prototypes(src_x, labels, n_cls,
                                 prob.max(-1).values if weight is None else weight)
        nb = _neighbor_mean(src_x, ei, n)
        return P[labels], nb, prob

    def run(x0, ei, variant, tau=0.10, rounds=3):
        x_cur = x0
        votes = None
        for _ in range(rounds):
            prob = F.softmax(logits(x_cur, ei), -1)
            pseudo = prob.argmax(-1)
            P, _ = _class_prototypes(x0, pseudo, n_cls, prob.max(-1).values)
            nb = _neighbor_mean(x0, ei, n)
            pw = P[pseudo]
            if variant == "or":
                g = ((pw >= tau) | (nb >= tau)).float()
            elif variant == "and":
                g = ((pw >= tau) & (nb >= tau)).float()
            else:
                raise ValueError(variant)
            votes = g if votes is None else votes + g
            x_cur = x0 * g
        if variant == "vote":
            pass
        return x_cur, votes

    def soft(x0, ei, gamma=1.0, normalize=False, rounds=3):
        x_cur = x0
        for _ in range(rounds):
            prob = F.softmax(logits(x_cur, ei), -1)
            pseudo = prob.argmax(-1)
            P, _ = _class_prototypes(x0, pseudo, n_cls, prob.max(-1).values)
            w = P[pseudo].clamp(0, 1) ** gamma
            x_cur = x0 * w
        if normalize:
            s = x_cur.sum(1, keepdim=True)
            scale = tgt_sum / s.clamp(min=1e-6)
            scale[s <= 0] = 0
            x_cur = x_cur * scale
        return x_cur

    variants = ["erm", "or_tau0.10", "and_tau0.10", "vote_or", "vote_and",
                "soft_g1", "soft_g2", "soft_g1_norm", "oracle_proto", "oracle_and"]
    res = {k: [] for k in variants}
    pseudo_acc = []

    for g in make_test_set(clean, "cora", n_graphs=8, seed=seed):
        def acc(xx, ei=g.edge_index):
            return float((logits(xx, ei)[mask].argmax(-1) == y[mask]).float().mean())

        res["erm"].append(acc(g.x))
        pseudo_acc.append(float((logits(g.x, g.edge_index)[mask].argmax(-1) == y[mask]).float().mean()))

        xo, _ = run(g.x, g.edge_index, "or")
        res["or_tau0.10"].append(acc(xo))

        xa, va = run(g.x, g.edge_index, "and")
        res["and_tau0.10"].append(acc(xa))
        res["vote_and"].append(acc(g.x * (va >= 2).float()))

        # vote over OR gates of different tau
        xv = g.x
        votes = None
        for t in (0.05, 0.10, 0.15):
            xvt, _ = run(g.x, g.edge_index, "or", tau=t)
            votes = (xvt.sum(1, keepdim=True) > 0).float() if votes is None else votes + (xvt.sum(1, keepdim=True) > 0).float()
        res["vote_or"].append(acc(g.x * (votes >= 2).float()))

        res["soft_g1"].append(acc(soft(g.x, g.edge_index, 1.0)))
        res["soft_g2"].append(acc(soft(g.x, g.edge_index, 2.0)))
        res["soft_g1_norm"].append(acc(soft(g.x, g.edge_index, 1.0, normalize=True)))

        # oracle：原型用真实标签（上界，不可用于方法）
        prob = F.softmax(logits(g.x, g.edge_index), -1)
        P, _ = _class_prototypes(g.x, y, n_cls, prob.max(-1).values)
        nb = _neighbor_mean(g.x, g.edge_index, n)
        pw = P[y]
        res["oracle_proto"].append(acc(g.x * (pw >= 0.10).float()))
        res["oracle_and"].append(acc(g.x * ((pw >= 0.10) & (nb >= 0.10)).float()))

    E = sum(res["erm"]) / 8
    print(f"seed{seed} ERM={E:.4f}  pseudo-label acc={sum(pseudo_acc)/8:.4f}")
    for k in variants:
        if k == "erm":
            continue
        v = res[k]
        m = sum(v) / 8
        print(f"  {k:<16} mean={m:.4f} delta={m-E:+.4f} per={[round(a,3) for a in v]}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
