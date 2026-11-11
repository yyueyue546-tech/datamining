"""Read the local preprocessed Elliptic time slices."""

import pickle
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data


def load_elliptic(root):
    """Load trusted pickles from the Elliptic directory into 5/5/33 splits."""
    root = Path(root)
    splits = {}
    for split, steps in (
        ("train", range(6, 11)),
        ("valid", range(11, 16)),
        ("test", range(16, 49)),
    ):
        graphs = []
        for step in steps:
            path = root / f"{step}.pkl"
            with path.open("rb") as handle:
                adjacency, labels, features = pickle.load(handle)
            x = torch.as_tensor(features, dtype=torch.float32)
            y = torch.as_tensor(labels, dtype=torch.long).reshape(-1)
            if x.ndim != 2 or y.numel() != x.size(0) or adjacency.shape != (x.size(0), x.size(0)):
                raise ValueError(f"Inconsistent Elliptic graph dimensions in {path}")
            if not torch.isin(y, y.new_tensor([-1, 0, 1])).all():
                raise ValueError(f"Unexpected Elliptic label encoding in {path}")
            edge_index = torch.from_numpy(np.stack(adjacency.nonzero())).long()
            graphs.append(Data(x=x, edge_index=edge_index, y=y, mask=y >= 0, time_step=step))
        splits[split] = graphs
    return splits
