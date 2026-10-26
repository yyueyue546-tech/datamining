"""Run the documented local Cora checks and preserve machine-readable evidence."""
from pathlib import Path
import hashlib
import json
import platform
import subprocess
import sys
import time

import torch
import torch_geometric
from torch_geometric.io import read_planetoid_data

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
CACHE = REPO / "data/Cora/processed/data.pt"
if not CACHE.exists():
    graph = read_planetoid_data(str(REPO / "data/Cora/raw"), "Cora")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save((graph.to_dict(), None, graph.__class__), CACHE)

sys.path.insert(0, str(ROOT))
from tt_greb_reproduce.data import load_masked_cora

graph = load_masked_cora(CACHE)
masks = [graph[k] for k in ("train_mask", "val_mask", "test_mask")]
assert graph.num_nodes == 2708 and graph.num_features == 1433
assert [int(m.sum()) for m in masks] == [140, 500, 1000]
assert not bool((masks[0] & masks[1] | masks[0] & masks[2] | masks[1] & masks[2]).any())
evidence = {
    "python": platform.python_version(), "torch": torch.__version__,
    "pyg": torch_geometric.__version__, "cache": str(CACHE),
    "cache_sha256": hashlib.sha256(CACHE.read_bytes()).hexdigest(),
    "nodes": graph.num_nodes, "features": graph.num_features,
    "edge_entries": graph.num_edges, "classes": int(graph.y.max()) + 1,
    "masks": [int(m.sum()) for m in masks], "runs": [],
}
for name, epochs, steps in [
    ("cora_masked_gcn_seed0", 50, 5),
    ("cora_masked_gcn_seed0_repeat", 50, 5),
    ("cora_masked_gcn_seed0_200_20", 200, 20),
    ("cora_masked_gcn_seed0_200_20_repeat", 200, 20),
]:
    output = RESULTS / name
    output.mkdir(exist_ok=True)
    command = [sys.executable, str(ROOT / "run_cora_gcn.py"),
               "--processed-cora", str(CACHE), "--epochs", str(epochs),
               "--adapt-steps", str(steps), "--seed", "0", "--device", "cpu",
               "--output", str(output)]
    print(f"Starting {name}", flush=True)
    start = time.perf_counter()
    with (output / "run.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["evaluation_protocol"] == "node_masks"
    assert not summary["paper_split_verified"]
    assert len(summary["erm_accuracy"]) == len(summary["tt_greb_accuracy"]) == 1
    for filename in ("checkpoint.pt", "config.json", "summary.json", "test_results.csv"):
        assert (output / filename).stat().st_size > 0
    evidence["runs"].append({"name": name, "seconds": time.perf_counter() - start,
                             "command": command, "summary": summary})
    (RESULTS / "verification.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(f"Completed {name}: {summary['erm_mean_accuracy']}, {summary['tt_greb_mean_accuracy']}", flush=True)

evidence["repeat_checks"] = []
for first, second in [(0, 1), (2, 3)]:
    a, b = evidence["runs"][first], evidence["runs"][second]
    same_summary = a["summary"] == b["summary"]
    sa = torch.load(RESULTS / a["name"] / "checkpoint.pt", weights_only=True)
    sb = torch.load(RESULTS / b["name"] / "checkpoint.pt", weights_only=True)
    same_weights = sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    assert same_summary and same_weights
    evidence["repeat_checks"].append({"runs": [a["name"], b["name"]],
                                      "identical_summary": same_summary,
                                      "identical_checkpoint_tensors": same_weights})
(RESULTS / "verification.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
print("All runs and repeat checks passed.", flush=True)
