"""Check frozen GCN weights and absence of label dependence on a tiny graph."""
import json
from pathlib import Path
import torch
from torch_geometric.data import Data
from tt_greb_reproduce.models import GCN
from tt_greb_reproduce.tt_greb import adapt_test_graph

torch.manual_seed(0)
graph = Data(x=torch.rand(6, 4),
             edge_index=torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
                                      [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]]),
             y=torch.tensor([0, 1, 0, 1, 0, 1]))
model = GCN(4, 4, 2).eval()
before = {k: v.clone() for k, v in model.state_dict().items()}
torch.manual_seed(0)
first = adapt_test_graph(model, graph, steps=1)
changed = graph.clone()
changed.y = 1 - graph.y
torch.manual_seed(0)
second = adapt_test_graph(model, changed, steps=1)
checks = {
    "label_independent_features": torch.equal(first.graph.x, second.graph.x),
    "label_independent_edge_weights": torch.equal(first.graph.edge_weight, second.graph.edge_weight),
    "gnn_parameters_unchanged": all(torch.equal(before[k], v) for k, v in model.state_dict().items()),
    "gnn_parameters_frozen": all(not p.requires_grad for p in model.parameters()),
}
assert all(checks.values()), checks
path = Path(__file__).parent / "results/adaptation_invariants.json"
path.parent.mkdir(exist_ok=True)
path.write_text(json.dumps(checks, indent=2), encoding="utf-8")
print(json.dumps(checks, indent=2))
