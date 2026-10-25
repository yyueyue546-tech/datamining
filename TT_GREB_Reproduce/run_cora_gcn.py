from __future__ import annotations

import argparse
import csv
import copy
import json
import platform
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from tt_greb_reproduce.data import load_masked_cora, load_split, move_graph, standard_cora_smoke
from tt_greb_reproduce.models import GCN
from tt_greb_reproduce.tt_greb import adapt_test_graph


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy(model, graph, mask=None):
    with torch.no_grad():
        logits = model(graph.x, graph.edge_index, edge_weight=getattr(graph, "edge_weight", None))
    if mask is not None:
        logits, labels = logits[mask], graph.y[mask]
    else:
        labels = graph.y
    return float((logits.argmax(-1) == labels).float().mean().cpu())


def train(
    train_graph,
    valid_graph,
    num_classes: int,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
):
    model = GCN(train_graph.x.size(1), hidden_dim, num_classes).to(train_graph.x.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state, best_valid = None, -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(train_graph.x, train_graph.edge_index)
        loss = F.cross_entropy(logits, train_graph.y)
        loss.backward()
        optimizer.step()
        model.eval()
        valid_acc = accuracy(model, valid_graph)
        if valid_acc > best_valid:
            best_valid = valid_acc
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, best_valid


def train_masked(
    graph,
    num_classes: int,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
):
    model = GCN(graph.x.size(1), hidden_dim, num_classes).to(graph.x.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state, best_valid = None, -1.0
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(graph.x, graph.edge_index)
        loss = F.cross_entropy(logits[graph.train_mask], graph.y[graph.train_mask])
        loss.backward()
        optimizer.step()
        model.eval()
        valid_acc = accuracy(model, graph, graph.val_mask)
        if valid_acc > best_valid:
            best_valid = valid_acc
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, best_valid


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--split", type=Path)
    source.add_argument("--smoke-standard-cora", action="store_true")
    source.add_argument("--processed-cora", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--adapt-steps", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    set_seed(args.seed)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if args.processed_cora:
        graph = move_graph(load_masked_cora(args.processed_cora), device)
        num_classes = int(graph.y.max()) + 1
        model, valid_acc = train_masked(
            graph, num_classes, args.hidden_dim, args.epochs, args.lr, args.weight_decay
        )
        test_graphs = [graph]
        test_masks = [graph.test_mask]
        source_name = str(args.processed_cora)
    else:
        splits = standard_cora_smoke() if args.smoke_standard_cora else load_split(args.split)
        train_graph = move_graph(splits.train, device)
        valid_graph = move_graph(splits.valid, device)
        test_graphs = [move_graph(graph, device) for graph in splits.test]
        test_masks = [None] * len(test_graphs)
        num_classes = max(int(graph.y.max()) for graph in [train_graph, valid_graph, *test_graphs]) + 1
        model, valid_acc = train(
            train_graph,
            valid_graph,
            num_classes,
            args.hidden_dim,
            args.epochs,
            args.lr,
            args.weight_decay,
        )
        source_name = "standard-cora-smoke" if args.smoke_standard_cora else str(args.split)
    torch.save(model.state_dict(), output / "checkpoint.pt")

    erm = [accuracy(model, graph, mask) for graph, mask in zip(test_graphs, test_masks)]
    tt_greb = []
    adaptation = []
    for graph in test_graphs:
        result = adapt_test_graph(model, graph, steps=args.adapt_steps)
        tt_greb.append(accuracy(model, result.graph, mask))
        adaptation.append({"steps": result.steps, "final_loss": result.final_loss})
    with (output / "test_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["test_graph", "erm_accuracy", "tt_greb_accuracy"])
        writer.writerows(
            (index, erm_score, tt_score)
            for index, (erm_score, tt_score) in enumerate(zip(erm, tt_greb), start=1)
        )
    summary = {
        "seed": args.seed,
        "source": source_name,
        "paper_split_verified": False,
        "train_graphs": 1,
        "valid_graphs": 1,
        "test_graphs": len(test_graphs),
        "evaluation_protocol": "node_masks" if args.processed_cora else "separate_graphs",
        "validation_accuracy": valid_acc,
        "erm_accuracy": erm,
        "tt_greb_accuracy": tt_greb,
        "erm_mean_accuracy": sum(erm) / len(erm),
        "tt_greb_mean_accuracy": sum(tt_greb) / len(tt_greb),
        "adaptation": adaptation,
    }
    config = {
        "seed": args.seed,
        "epochs": args.epochs,
        "hidden_dim": args.hidden_dim,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "adapt_steps": args.adapt_steps,
        "device": str(device),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "source": summary["source"],
        "smoke_only": args.smoke_standard_cora,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
