"""Train and evaluate one configured node-classification backbone."""

import argparse
import copy
import csv
import gzip
import json
import pickle
import random
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_undirected

if __package__:
    from .models import GCN, GraphSAGE
    from .models._elliptic import load_elliptic
else:
    from models import GCN, GraphSAGE
    from models._elliptic import load_elliptic


ROOT = Path(__file__).resolve().parent
MODELS = {"gcn": GCN, "graphsage": GraphSAGE}
DATASETS = ("cora", "amazon_photo", "twitch", "ogb_arxiv", "elliptic")


def read_yaml(path):
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_twitch(root, domains):
    graphs = []
    width = 0
    for domain in domains:
        prefix = root / domain / f"musae_{domain}"
        features = json.loads(prefix.with_name(prefix.name + "_features.json").read_text())
        with prefix.with_name(prefix.name + "_target.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        y = torch.full((max(int(row["new_id"]) for row in rows) + 1,), -1)
        for row in rows:
            y[int(row["new_id"])] = int(row["mature"].lower() == "true")
        with prefix.with_name(prefix.name + "_edges.csv").open(newline="") as handle:
            edges = [(int(row["from"]), int(row["to"])) for row in csv.DictReader(handle)]
        nodes, columns = zip(*((int(node), col) for node, cols in features.items() for col in cols))
        width = max(width, max(columns) + 1)
        graphs.append((y, edges, list(nodes), list(columns)))

    result = []
    for y, edges, nodes, columns in graphs:
        x = torch.zeros(y.numel(), width)
        x[nodes, columns] = 1
        edge_index = to_undirected(torch.tensor(edges).t().contiguous(), num_nodes=y.numel())
        result.append(Data(x=x, y=y, edge_index=edge_index))
    return dict(train=result[:1], valid=result[1:2], test=result[2:])


def load_arxiv(root):
    payload = torch.load(root / "processed/data_processed", map_location="cpu", weights_only=False)
    raw = payload["graph"]
    x = torch.as_tensor(raw["node_feat"], dtype=torch.float32)
    y = torch.as_tensor(payload["labels"], dtype=torch.long).reshape(-1)
    edges = torch.as_tensor(raw["edge_index"], dtype=torch.long)
    splits = {}
    for split in ("train", "valid", "test"):
        with gzip.open(root / f"split/time/{split}.csv.gz", "rt") as handle:
            nodes = torch.as_tensor(np.loadtxt(handle, delimiter=",", dtype=np.int64))
        edge_index, _ = subgraph(nodes, edges, relabel_nodes=True, num_nodes=x.size(0))
        splits[split] = [Data(x=x[nodes], y=y[nodes], edge_index=to_undirected(edge_index))]
    return splits


def load_data(args):
    root = args.data_root
    if args.split_file:
        # Explicitly supplied local split files must be trusted.
        payload = torch.load(args.split_file, map_location="cpu", weights_only=False)
        valid = payload["valid"] if "valid" in payload else payload["val"]
        return {
            key: [value] if isinstance(value, Data) else list(value)
            for key, value in dict(train=payload["train"], valid=valid, test=payload["test"]).items()
        }
    if args.dataset == "elliptic":
        return load_elliptic(root / "elliptic")
    if args.dataset == "twitch":
        return load_twitch(root / "twitch", args.twitch_domains)
    if args.dataset == "ogb_arxiv":
        return load_arxiv(root / "ogbn_arxiv")

    folder = root / ("Planetoid/cora" if args.dataset == "cora" else "Amazon/Photo")
    cache = torch.load(folder / "processed/data.pt", map_location="cpu", weights_only=False)
    edges = cache[0]["edge_index"]
    graphs = []
    for index in range(10):
        with (folder / f"gen/{index}-gcn.pkl").open("rb") as handle:
            x, y = pickle.load(handle)
        graphs.append(Data(x=torch.as_tensor(x, dtype=torch.float32), edge_index=edges,
                           y=torch.as_tensor(y, dtype=torch.long).reshape(-1)))
    return dict(train=graphs[:1], valid=graphs[1:2], test=graphs[2:])


def labels_and_mask(graph):
    y = graph.y.reshape(-1).long()
    mask = y >= 0
    if getattr(graph, "mask", None) is not None:
        mask &= graph.mask.reshape(-1).bool()
    return y, mask


def forward(model, graph):
    y, mask = labels_and_mask(graph)
    logits = model(graph.x, graph.edge_index, getattr(graph, "edge_weight", None))
    return logits[mask], y[mask]


@torch.no_grad()
def evaluate(model, graphs, metric, classes, device, positive_class):
    model.eval()
    scores = []
    for index, graph in enumerate(graphs):
        logits, labels = forward(model, copy.copy(graph).to(device))
        if labels.numel() == 0:
            raise ValueError(f"Evaluation graph {index} has no labeled nodes")
        if metric == "roc_auc":
            positive = torch.searchsorted(classes, classes.new_tensor(positive_class)).item()
            score = roc_auc_score((labels == positive_class).cpu(), logits[:, positive].cpu())
        else:
            predictions = classes[logits.argmax(dim=1)]
            if metric == "accuracy":
                score = (predictions == labels).float().mean().item()
            elif metric == "f1":
                predicted = predictions == positive_class
                actual = labels == positive_class
                denominator = int(predicted.sum() + actual.sum())
                score = 2 * int((predicted & actual).sum()) / denominator if denominator else 0.0
            else:
                raise ValueError(f"Unsupported validation metric: {metric}")
        if not np.isfinite(score):
            raise ValueError(f"Non-finite {metric} on evaluation graph {index}")
        scores.append(float(score))
    return {"mean": float(np.mean(scores)), "per_graph": scores}


def fit(args, config, splits):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_graphs = [copy.copy(graph).to(device) for graph in splits["train"]]
    valid_graphs = [copy.copy(graph).to(device) for graph in splits["valid"]]
    classes = torch.cat([y[mask] for y, mask in map(labels_and_mask, train_graphs)]).unique()
    options = config["training"]
    metric = options["validation_metric"]
    loss_name = options["loss"]
    if classes.numel() < 2 or (loss_name == "bce_with_logits" and classes.tolist() != [0, 1]):
        raise ValueError(f"Unexpected training labels for {loss_name}: {classes.tolist()}")
    if loss_name not in {"cross_entropy", "bce_with_logits"}:
        raise ValueError(f"Unsupported loss: {loss_name}")
    model = MODELS[args.model](train_graphs[0].num_features, out_channels=classes.numel(),
                               **config["model"]).to(device)
    optimizer_type = {"AdamW": torch.optim.AdamW, "Adam": torch.optim.Adam}[options["optimizer"]]
    optimizer = optimizer_type(model.parameters(), lr=options["learning_rate"],
                               weight_decay=options["weight_decay"])
    epochs = args.epochs if args.epochs is not None else options["max_epochs"]
    positive_class = args.positive_class
    best = None
    for epoch in range(1, epochs + 1):
        model.train()
        for graph in train_graphs:
            optimizer.zero_grad()
            logits, labels = forward(model, graph)
            targets = torch.searchsorted(classes, labels)
            if loss_name == "bce_with_logits":
                targets = F.one_hot(targets, num_classes=classes.numel()).to(logits.dtype)
                loss = F.binary_cross_entropy_with_logits(logits, targets)
            else:
                loss = F.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            optimizer.step()

        validation = evaluate(model, valid_graphs, metric, classes, device, positive_class)
        if best is None or validation["mean"] > best["validation"]["mean"]:
            best = {"epoch": epoch, "validation": validation,
                    "model_state_dict": {key: value.detach().cpu().clone()
                                         for key, value in model.state_dict().items()}}
    model.load_state_dict(best["model_state_dict"])
    return model, classes, best


def save_run(args, config, splits, fitted, extra=None):
    model, classes, best = fitted
    metric = config["training"]["validation_metric"]
    test = evaluate(model, splits["test"], metric, classes, classes.device, args.positive_class)
    run = f"{datetime.now():%Y%m%d-%H%M%S}-seed{args.seed}-{uuid4().hex[:8]}"
    relative = Path(args.dataset) / args.model / run
    checkpoint = args.output_root / "checkpoints" / relative.with_suffix(".pt")
    result = args.output_root / "results" / relative.with_suffix(".json")
    metadata = {
        "dataset": args.dataset, "model": args.model, "seed": args.seed,
        "config": config, "epochs": args.epochs or config["training"]["max_epochs"],
        "data_root": str(args.data_root.resolve()),
        "split_file": str(args.split_file.resolve()) if args.split_file else None,
        "twitch_domains": args.twitch_domains if args.dataset == "twitch" and not args.split_file else None,
        "graphs_per_split": {name: len(graphs) for name, graphs in splits.items()},
        "in_channels": splits["train"][0].num_features, "classes": classes.tolist(),
        "metric": metric, "positive_class": args.positive_class,
        "best_epoch": best["epoch"], "validation": best["validation"], "test": test,
        "checkpoint": str(checkpoint.resolve()),
    }
    if extra:
        metadata.update(extra)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    result.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**metadata, "model_state_dict": best["model_state_dict"]}, checkpoint)
    result.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"best_epoch={best['epoch']} val_{metric}={best['validation']['mean']:.6f} "
          f"test_{metric}={test['mean']:.6f}")
    print(f"Checkpoint: {checkpoint}\nResult: {result}")


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def make_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--config", type=Path, help="Override the fixed YAML path")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--split-file", type=Path, help="Trusted torch file with train/valid/test graphs")
    parser.add_argument("--twitch-domains", nargs=7, default=["ENGB", "DE", "ES", "FR", "PTBR", "RU", "TW"],
                        help="Twitch domains in train, validation, then five test positions")
    parser.add_argument("--positive-class", type=int, choices=(0, 1),
                        help="Positive class for F1 or ROC-AUC (default: 0 for Elliptic, 1 otherwise)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=positive_int)
    parser.add_argument("--device", help="Torch device, e.g. cpu or cuda:0")
    return parser


def prepare(args):
    if args.positive_class is None:
        args.positive_class = 0 if args.dataset == "elliptic" else 1
    if args.dataset == "twitch" and not args.split_file and len(set(args.twitch_domains)) != 7:
        raise ValueError("Twitch train, validation and test domains must be distinct")
    config = read_yaml(args.config or ROOT / "configs" / f"{args.dataset}.yaml")[args.model]
    splits = load_data(args)
    if any(not graphs for graphs in splits.values()):
        raise ValueError("Train, validation, and test splits must each contain at least one graph")
    return config, splits


def main():
    args = make_parser(__doc__).parse_args()
    config, splits = prepare(args)
    save_run(args, config, splits, fit(args, config, splits))


if __name__ == "__main__":
    main()
