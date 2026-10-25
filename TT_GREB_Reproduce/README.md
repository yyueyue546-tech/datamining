# TT-GREB reproduction scaffold

This directory contains the first reproducible experiment loop for **Test-Time
Graph Rebirth (TT-GREB)**:

```text
Cora split (1 train / 1 validation / 8 test graphs)
  -> train GCN
  -> select the best validation checkpoint
  -> freeze the GCN
  -> ERM on the original test graphs
  -> TT-GREB on the same test graphs
  -> compare mean Accuracy
```

The repository currently contains the standard Planetoid Cora raw files, not the
paper's processed `1/1/8` split. Therefore `--smoke-standard-cora` is only a
pipeline smoke test; it must not be reported as the paper result. For a paper
run, provide a `.pt` split file with this structure:

```python
{
    "train": torch_geometric.data.Data,
    "valid": torch_geometric.data.Data,
    "test": [torch_geometric.data.Data, ...],  # exactly 8 graphs for Cora
}
```

Each graph must have `x`, `edge_index`, and `y`. The loader checks graph indices,
feature dimensions, class ranges, and the expected `1/1/8` count.

## Environment

Use Python 3.10 or 3.11 for the PyTorch 2.6 stack:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch==2.6.0 torch_geometric==2.6.1 ogb numpy
```

## Run the smoke loop

From the workspace root:

```powershell
python TT_GREB_Reproduce/run_cora_gcn.py `
  --smoke-standard-cora `
  --output TT_GREB_Reproduce/results/cora_gcn_seed0_smoke
```

The command writes the checkpoint, per-test-graph scores, configuration, and a
summary JSON. It exercises the complete code path but does not reproduce the
paper's Cora numbers.

## Run the supplied masked Cora graph

The local `Cora/processed/data.pt` is a single standard Planetoid Cora graph
with 140 training, 500 validation, and 1,000 test nodes. To run its conventional
node-mask protocol, point the runner at the processed file:

```powershell
python TT_GREB_Reproduce/run_cora_gcn.py `
  --processed-cora "D:\paper\数据挖掘\Cora\Cora\processed\data.pt" `
  --output TT_GREB_Reproduce/results/cora_masked_gcn_seed0
```

This trains and selects the checkpoint using only `train_mask` and `val_mask`,
then scores only `test_mask`. It is a useful local baseline and implementation
check, but it is not the paper's 1/1/8 multi-graph protocol.

The included adaptation loop is an implementation scaffold for the prototype
and environment reweighting modules. Its current cosine-similarity objective is
a runnable surrogate, not a verified transcription of every term in the paper's
dual contrastive objective. Do not report its scores as a faithful TT-GREB
reproduction until the objective, optimizer schedule, and hyperparameters have
been checked against the final ICDM paper or official implementation.

## Run the paper split

```powershell
python TT_GREB_Reproduce/run_cora_gcn.py `
  --split TT_GREB_Reproduce/data/cora_1_1_8.pt `
  --output TT_GREB_Reproduce/results/cora_gcn_seed0
```

The current GitHub data directory does not contain `cora_1_1_8.pt`; this file
must come from the authors' preprocessing or be generated from the exact split
protocol before claiming a paper reproduction.

For a local smoke artifact and its validation:

```powershell
python TT_GREB_Reproduce/prepare_standard_cora_smoke.py `
  --output TT_GREB_Reproduce/data/cora_standard_smoke.pt
python TT_GREB_Reproduce/inspect_split.py `
  TT_GREB_Reproduce/data/cora_standard_smoke.pt
```
