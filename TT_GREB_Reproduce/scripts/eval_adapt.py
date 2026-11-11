"""用已训练好的 checkpoint 快速验证 adapt_test_graph 的效果。

用法：
  python scripts/eval_adapt.py --seed 0 --graphs 0,3,7 --steps 100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from tt_greb_reproduce.backbones import build_backbone, forward_logits
from tt_greb_reproduce.data import load_clean, split_118
from tt_greb_reproduce.shift import make_test_set
from tt_greb_reproduce.tt_greb import adapt_test_graph
from tt_greb_reproduce.paths import DATASET_META


def evaluate(model, g, mask):
    model.eval()
    with torch.no_grad():
        logits = forward_logits(model, g.x, g.edge_index,
                                getattr(g, "edge_weight", None))
    return float((logits[mask].argmax(-1) == g.y[mask]).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cora")
    ap.add_argument("--backbone", default="gcn",
                    choices=["gcn", "sage", "gat", "gprgnn"])
    ap.add_argument("--model-module", default=None)
    ap.add_argument("--model-kwargs", default=None,
                    help="JSON，如 '{\"heads\": 4}'")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--graphs", default="0,3,7")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--ckpt", default=None, help="直接指定 checkpoint 路径")
    args = ap.parse_args()

    if args.model_path:
        sys.path.insert(0, str(Path(args.model_path).resolve()))
    model_kwargs = json.loads(args.model_kwargs) if args.model_kwargs else None

    idxs = [int(s) for s in args.graphs.split(",") if s != ""]
    meta = DATASET_META[args.dataset]
    clean = load_clean(args.dataset, seed=args.seed)
    base, _, _ = split_118(clean, seed=args.seed, n_test_graphs=0)
    mask = base.test_mask

    tag = args.dataset if args.backbone == "gcn" else f"{args.dataset}_{args.backbone}"
    ckpt = Path(args.ckpt) if args.ckpt else (
        Path("results_v2") / f"{tag}_seed{args.seed}" / "checkpoint.pt")
    model = build_backbone(args.backbone, base.x.size(1), 64, meta["num_classes"], 0.5,
                           module=args.model_module, module_kwargs=model_kwargs)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    graphs = make_test_set(clean, args.dataset, n_graphs=8, seed=args.seed)
    print(f"clean test = {evaluate(model, base, mask):.4f}")
    rows = []
    for i in idxs:
        g = graphs[i]
        e = evaluate(model, g, mask)
        r = adapt_test_graph(model, g, steps=args.steps, tau=args.tau,
                             edge_weight=getattr(g, "edge_weight", None))
        t = evaluate(model, r, mask)
        rows.append((i, e, t))
        print(f"test[{i}] erm={e:.4f} tta={t:.4f} delta={t - e:+.4f}")

    print("mean erm = %.4f  mean tta = %.4f  delta = %+.4f" % (
        sum(r[1] for r in rows) / len(rows),
        sum(r[2] for r in rows) / len(rows),
        sum(r[2] - r[1] for r in rows) / len(rows),
    ))


if __name__ == "__main__":
    main()
