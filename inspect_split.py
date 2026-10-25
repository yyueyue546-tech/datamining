"""Validate and print statistics for a prepared graph split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tt_greb_reproduce.data import load_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("split", type=Path)
    parser.add_argument("--expected-test-graphs", type=int, default=8)
    args = parser.parse_args()
    splits = load_split(args.split, expected_test_graphs=args.expected_test_graphs)
    graphs = [splits.train, splits.valid, *splits.test]
    stats = []
    for index, graph in enumerate(graphs):
        name = "train" if index == 0 else "valid" if index == 1 else f"test[{index - 2}]"
        stats.append(
            {
                "name": name,
                "nodes": graph.num_nodes,
                "edges": graph.num_edges,
                "features": graph.x.size(1),
                "classes": int(graph.y.max()) + 1,
            }
        )
    print(json.dumps({"train_graphs": 1, "valid_graphs": 1, "test_graphs": len(splits.test), "graphs": stats}, indent=2))


if __name__ == "__main__":
    main()
