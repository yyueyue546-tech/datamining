"""第二轮改进实验：针对“噪声是加词而非丢词”这一事实。

T7 上 clean rowsum=18.2，带噪 rowsum=37.1，即约一半词是假的。
当前门控只用概率阈值，无法删掉“高频但属于噪声”的词。
这里测：
  renorm   : 门控后按行重新缩放到训练均值（修正尺度漂移）
  topk     : 按类原型概率取 top-k（k=18）
  topk_tau : top-k 且概率 >= tau
  tau0.1/0.2/0.5 : 单纯提高阈值
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
    k_clean = int(round(tgt_sum))
    model = build_model("gcn", base.x.size(1), 64, n_cls, 0.5)
    model.load_state_dict(torch.load(f"results_v2/cora_seed{seed}/checkpoint.pt", map_location="cpu"))
    model.eval()

    def logits(x, ei):
        with torch.no_grad():
            return model(x, ei)

    def gates(x0, ei, tau=0.05, rounds=3, mode="tau", k=k_clean):
        x_cur = x0
        comp = None
        for _ in range(rounds):
            prob = F.softmax(logits(x_cur, ei), -1)
            pseudo = prob.argmax(-1)
            P, _ = _class_prototypes(x0, pseudo, n_cls, prob.max(-1).values)
            nb = _neighbor_mean(x0, ei, n)
            pw = P[pseudo]
            if mode == "tau":
                comp = ((pw >= tau) | (nb >= tau)).float()
            else:  # topk / topk_tau
                idx = pw.topk(min(k, pw.size(1)), dim=-1).indices
                comp = torch.zeros_like(pw)
                comp.scatter_(1, idx, 1.0)
                if mode == "topk_tau":
                    comp = comp * (pw >= tau).float()
                comp = ((comp > 0) | (nb >= tau)).float()
            x_cur = x0 * comp
        return x_cur

    def renorm(x):
        s = x.sum(1, keepdim=True)
        scale = tgt_sum / s.clamp(min=1e-6)
        scale[s <= 0] = 0.0
        return x * scale

    variants = [
        "erm", "tau0.05", "tau0.10", "tau0.20", "tau0.50",
        "renorm_raw", "renorm_tau0.05",
        "topk", "topk_tau", "renorm_topk",
    ]
    res = {k: [] for k in variants}
    for g in make_test_set(clean, "cora", n_graphs=8, seed=seed):
        def acc(xx, ei):
            return float((logits(xx, ei)[mask].argmax(-1) == y[mask]).float().mean())

        res["erm"].append(acc(g.x, g.edge_index))
        res["renorm_raw"].append(acc(renorm(g.x), g.edge_index))
        for t in (0.05, 0.10, 0.20, 0.50):
            res[f"tau{t:.2f}"].append(acc(gates(g.x, g.edge_index, tau=t), g.edge_index))
        xg = gates(g.x, g.edge_index, tau=0.05)
        res["renorm_tau0.05"].append(acc(renorm(xg), g.edge_index))
        xt = gates(g.x, g.edge_index, mode="topk")
        res["topk"].append(acc(xt, g.edge_index))
        res["topk_tau"].append(acc(gates(g.x, g.edge_index, mode="topk_tau"), g.edge_index))
        res["renorm_topk"].append(acc(renorm(xt), g.edge_index))

    E = sum(res["erm"]) / 8
    print(f"seed{seed}  ERM mean={E:.4f}  (clean rowsum={tgt_sum:.2f}, k={k_clean})")
    for k in variants:
        if k == "erm":
            continue
        v = res[k]
        m = sum(v) / 8
        print(f"  {k:<16} mean={m:.4f} delta={m-E:+.4f} per={[round(a,3) for a in v]}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
