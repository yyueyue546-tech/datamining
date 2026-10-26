"""Verify that saved model tensors reproduce the reported baseline scores."""
import json
from pathlib import Path
import torch
from tt_greb_reproduce.data import load_masked_cora
from tt_greb_reproduce.models import GCN

root = Path(__file__).resolve().parent
evidence = json.loads((root / "results/verification.json").read_text())
graph = load_masked_cora(evidence["cache"])
checks = []
for run in evidence["runs"]:
    model = GCN(graph.num_features, 64, int(graph.y.max()) + 1).eval()
    model.load_state_dict(torch.load(root / "results" / run["name"] / "checkpoint.pt", weights_only=True))
    with torch.no_grad():
        prediction = model(graph.x, graph.edge_index).argmax(-1)
        valid = float((prediction[graph.val_mask] == graph.y[graph.val_mask]).float().mean())
        test = float((prediction[graph.test_mask] == graph.y[graph.test_mask]).float().mean())
    assert valid == run["summary"]["validation_accuracy"]
    assert test == run["summary"]["erm_mean_accuracy"]
    checks.append({"run": run["name"], "validation_accuracy": valid, "erm_accuracy": test})
(root / "results/checkpoint_verification.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
print(f"PASS: {len(checks)} saved checkpoints reproduce validation and ERM scores")
