"""汇总 results_v2 下的所有实验，生成 REPORT.md 与 results.csv。"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results_v2")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out) if args.out else root / "REPORT.md"

    groups = defaultdict(list)
    for p in glob.glob(str(root / "*_seed*" / "summary.json")):
        s = json.load(open(p, encoding="utf-8"))
        key = f"{s['dataset']}-{s.get('backbone', 'gcn')}"
        groups[key].append(s)

    if not groups:
        print("no results found")
        return

    lines = ["# TT-GREB vs ERM 实验结果", ""]
    csv_rows = []

    for ds, runs in sorted(groups.items()):
        runs.sort(key=lambda s: s["seed"])
        metric = runs[0]["metric"]
        n_graph = runs[0]["n_test_graphs"]
        lines += [
            f"## {ds}（metric={metric}，{len(runs)} 个随机种子 × {n_graph} 张偏移测试图）",
            "",
            "| seed | clean test | ERM | TT-GREB | delta | TT std |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for s in runs:
            lines.append(
                f"| {s['seed']} | {s['clean_test']:.4f} | {s['erm']['mean']:.4f} | "
                f"{s['tta']['mean']:.4f} | {s['delta']['mean']:+.4f} | {s['tta']['std']:.4f} |"
            )
            csv_rows.append({
                "dataset": ds, "seed": s["seed"], "metric": metric,
                "clean_test": s["clean_test"], "erm": s["erm"]["mean"],
                "tt": s["tta"]["mean"], "delta": s["delta"]["mean"],
            })

        erm_m = st.mean(s["erm"]["mean"] for s in runs)
        tt_m = st.mean(s["tta"]["mean"] for s in runs)
        d_m = st.mean(s["delta"]["mean"] for s in runs)
        d_sd = st.stdev([s["delta"]["mean"] for s in runs]) if len(runs) > 1 else 0.0
        pos = sum(1 for s in runs if s["delta"]["mean"] > 0)

        # 按测试图强度（0=最轻，7=最重）取平均
        per_graph_erm = [st.mean(s["erm"]["values"][i] for s in runs) for i in range(n_graph)]
        per_graph_tt = [st.mean(s["tta"]["values"][i] for s in runs) for i in range(n_graph)]

        lines += [
            "",
            f"**均值**：ERM {erm_m:.4f} → TT-GREB {tt_m:.4f}，delta {d_m:+.4f}"
            f"（跨种子标准差 {d_sd:.4f}），正增益种子 {pos}/{len(runs)}。",
            "",
            "| 测试图 | " + " | ".join(f"T{i}" for i in range(n_graph)) + " |",
            "|---|" + "---:|" * n_graph,
            "| ERM | " + " | ".join(f"{v:.4f}" for v in per_graph_erm) + " |",
            "| TT-GREB | " + " | ".join(f"{v:.4f}" for v in per_graph_tt) + " |",
            "| delta | " + " | ".join(f"{t - e:+.4f}" for e, t in zip(per_graph_erm, per_graph_tt)) + " |",
            "",
        ]
        print(f"{ds}: ERM {erm_m:.4f} -> TT {tt_m:.4f} ({d_m:+.4f}), 正增益 {pos}/{len(runs)}")

    out.write_text("\n".join(lines), encoding="utf-8")
    with open(root / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "seed", "metric", "clean_test", "erm", "tt", "delta"])
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\nwrote {out} and {root / 'results.csv'}")


if __name__ == "__main__":
    main()
