"""TT-GREB 四数据集主实验。

协议：
  1. 干净图上训 ERM（GCN）
  2. 生成 8 张分布偏移测试图
  3. 每张图分别评估 ERM baseline 与 TT-GREB
  4. 汇总 mean/std 写入 summary.json
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import f1_score, roc_auc_score

from tt_greb_reproduce.data import load_clean, split_118, describe
from tt_greb_reproduce.shift import make_test_set, SHIFT_PRESETS
from tt_greb_reproduce.backbones import build_backbone, forward_logits
from tt_greb_reproduce.tt_greb import adapt_test_graph
from tt_greb_reproduce.paths import DATASET_META


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metric(model: nn.Module, graph, metric: str = "acc", mask_name: str = "test_mask") -> float:
    model.eval()
    with torch.no_grad():
        logits = forward_logits(
            model, graph.x, graph.edge_index,
            getattr(graph, "edge_weight", None),
        )

    mask = getattr(graph, mask_name, None)
    if mask is not None and mask.dtype == torch.bool and mask.numel() == graph.num_nodes:
        logits = logits[mask]
        y = graph.y[mask]
    else:
        y = graph.y

    if metric == "acc":
        return float((logits.argmax(-1) == y).float().mean().cpu())
    if metric == "f1":
        return float(f1_score(y.cpu(), logits.argmax(-1).cpu(), average="macro"))
    if metric == "auc":
        if logits.size(1) < 2:
            raise ValueError("ROC-AUC 需要至少两个类别")
        prob = torch.softmax(logits, dim=-1)[:, 1].cpu()
        return float(roc_auc_score(y.cpu(), prob))
    raise ValueError(f"unknown metric: {metric}")


def train_model(
    graph,
    backbone: str,
    num_classes: int,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    patience: int,
    metric: str,
    model_module: str | None = None,
    model_kwargs: dict | None = None,
) -> tuple[nn.Module, float, int]:
    device = graph.x.device
    model = build_backbone(
        backbone, graph.x.size(1), hidden_dim, num_classes, dropout,
        module=model_module, module_kwargs=model_kwargs,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state, best_val, best_epoch = None, -1.0, 0
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = forward_logits(model, graph.x, graph.edge_index,
                                getattr(graph, "edge_weight", None))
        loss = F.cross_entropy(logits[graph.train_mask], graph.y[graph.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        val_score = compute_metric(model, graph, metric, mask_name="val_mask")

        if val_score > best_val + 1e-6:
            best_val = val_score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if patience > 0 and stale >= patience:
                print(f"early stop at epoch {epoch}, best epoch {best_epoch}")
                break

        if epoch % 25 == 0 or epoch == 1:
            print(f"epoch {epoch:4d} loss={loss.detach().item():.4f} "
                  f"val_{metric}={val_score:.4f} best={best_val:.4f}")

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")

    model.load_state_dict(best_state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, best_val, best_epoch


def stats(values):
    a = np.array(values, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std()), "values": values}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cora",
                        choices=["cora", "amazon", "twitch", "ogb", "elliptic"])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backbone", default="gcn",
                        choices=["gcn", "sage", "gat", "gprgnn"],
                        help="gcn/sage 为本仓库实现；gat/gprgnn 从第三部分的 "
                             "models/ 包加载（需配合 --model-path）")
    parser.add_argument("--model-module", default=None,
                        help="直接指定模型类，如 models.gat:GAT")
    parser.add_argument("--model-kwargs", default=None,
                        help='传给模型构造函数的额外参数，JSON，如 \'{"heads": 4}\'')
    parser.add_argument("--model-path", default=None,
                        help="把该目录加入 sys.path，用于 import 第三部分的 models 包")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--adapt-rounds", type=int, default=3)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--n-test", type=int, default=8)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.model_path:
        import sys
        sys.path.insert(0, str(Path(args.model_path).resolve()))
    model_kwargs = json.loads(args.model_kwargs) if args.model_kwargs else None

    set_seed(args.seed)

    output = args.output or Path("results") / f"{args.dataset}_seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)

    meta = DATASET_META[args.dataset]
    metric = meta["metric"]
    device = torch.device(args.device)

    print(f"[{args.dataset}] metric={metric} device={device}")
    print(f"shift preset = {SHIFT_PRESETS[args.dataset]}")

    clean = load_clean(args.dataset, seed=args.seed).to(device)
    train_graph, _, _ = split_118(clean, seed=args.seed, n_test_graphs=0)
    train_graph = train_graph.to(device)

    num_classes = meta["num_classes"]
    print(describe(args.dataset, train_graph))
    print(f"train={int(train_graph.train_mask.sum())} "
          f"val={int(train_graph.val_mask.sum())} "
          f"test={int(train_graph.test_mask.sum())}")

    print(f"\ntraining ERM ({args.epochs} epochs, patience={args.patience})...")
    t0 = time.time()
    model, best_val, best_epoch = train_model(
        train_graph,
        backbone=args.backbone,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        patience=args.patience,
        metric=metric,
        model_module=args.model_module,
        model_kwargs=model_kwargs,
    )
    print(f"ERM done in {time.time()-t0:.1f}s, best_epoch={best_epoch} val={best_val:.4f}")

    train_acc = compute_metric(model, train_graph, metric, mask_name="train_mask")
    clean_test = compute_metric(model, train_graph, metric, mask_name="test_mask")
    print(f"train={train_acc:.4f} clean_test={clean_test:.4f}")

    torch.save(model.state_dict(), output / "checkpoint.pt")

    test_graphs = make_test_set(clean, args.dataset, n_graphs=args.n_test, seed=args.seed)
    print(f"\ngenerated {args.n_test} shifted test graphs")

    erm_scores, tta_scores, details = [], [], []

    for i, tg in enumerate(test_graphs):
        tg = tg.to(device)
        # 关键修复：评估 mask 必须用 split_118 的 2168 个测试节点，
        # 而不是 make_test_graph 从 clean 继承的 Planetoid 原始 1000 个
        tg.test_mask = train_graph.test_mask.to(device)
        tg.train_mask = train_graph.train_mask.to(device)
        tg.val_mask = train_graph.val_mask.to(device)

        erm_score = compute_metric(model, tg, metric, mask_name="test_mask")

        t1 = time.time()
        reborn = adapt_test_graph(
            model, tg,
            steps=args.adapt_rounds,
            tau=args.tau,
            edge_weight=getattr(tg, "edge_weight", None),
        )
        tta_score = compute_metric(model, reborn, metric, mask_name="test_mask")
        elapsed = time.time() - t1

        erm_scores.append(erm_score)
        tta_scores.append(tta_score)
        details.append({
            "graph": i,
            "erm": erm_score,
            "tta": tta_score,
            "delta": tta_score - erm_score,
            "loss_first": None,
            "loss_last": None,
            "ent_first": None,
            "ent_last": None,
            "tta_sec": round(elapsed, 1),
        })
        print(f"test[{i}] erm={erm_score:.4f} tta={tta_score:.4f} "
              f"delta={tta_score-erm_score:+.4f} "
              f"({elapsed:.1f}s)")

    print(f"\n[{args.dataset}] ERM {stats(erm_scores)['mean']:.4f} ± {stats(erm_scores)['std']:.4f}")
    print(f"[{args.dataset}] TTA {stats(tta_scores)['mean']:.4f} ± {stats(tta_scores)['std']:.4f}")
    print(f"[{args.dataset}] delta mean={np.mean(tta_scores)-np.mean(erm_scores):+.4f}")

    summary = {
        "dataset": args.dataset,
        "backbone": args.backbone,
        "backbone_class": type(model).__name__,
        "model_module": args.model_module,
        "model_kwargs": model_kwargs,
        "metric": metric,
        "seed": args.seed,
        "shift_preset": SHIFT_PRESETS[args.dataset],
        "n_test_graphs": args.n_test,
        "num_nodes": int(clean.num_nodes),
        "num_edges": int(clean.edge_index.size(1)),
        "num_classes": num_classes,
        "best_epoch": best_epoch,
        "validation_score": best_val,
        "train_acc": train_acc,
        "clean_test": clean_test,
        "erm": stats(erm_scores),
        "tta": stats(tta_scores),
        "delta": stats([t - e for t, e in zip(tta_scores, erm_scores)]),
        "details": details,
    }

    config = {
        "dataset": args.dataset,
        "backbone": args.backbone,
        "backbone_class": type(model).__name__,
        "model_module": args.model_module,
        "model_kwargs": model_kwargs,
        "model_path": args.model_path,
        "seed": args.seed,
        "shift_preset": SHIFT_PRESETS[args.dataset],
        "conf_thresh": getattr(args, "conf_thresh", 0.8),
        "adapter_bottleneck": 16,
        "epochs": args.epochs,
        "hidden_dim": args.hidden_dim,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "patience": args.patience,
        "adapt_rounds": args.adapt_rounds,
        "tau": args.tau,
        "n_test": args.n_test,
        "device": str(device),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
    }

    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n输出目录: {output.resolve()}")


if __name__ == "__main__":
    main()
