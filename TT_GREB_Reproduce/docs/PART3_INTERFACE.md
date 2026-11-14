# 第二部分（TT-GREB 主函数）如何承接第三部分的模型

本文对应 `数据挖掘11.9` 文档第 2 节与第 3 节：第 3 节负责 GCN / GraphSAGE / GAT /
GPR-GNN 四个骨干，第 2 节负责 TT-GREB 主函数。两边只需要对齐一个接口，主函数不
再依赖任何具体模型的内部结构。

## 1. 接口契约（第三部分必须满足）

每个骨干模型只要满足下面三条，就能直接进 TT-GREB：

```python
logits = model(x, edge_index, edge_weight=None)        # 必须
logits, embeddings = model.forward_with_embeddings(    # 可选，给基于表示的损失用
    x, edge_index, edge_weight=None)
model.reset_parameters()                               # 可选
```

约束：

| 项 | 要求 |
|---|---|
| 输出 | 直接返回 logits（`[N, C]`），不要返回 Accuracy / AUC；也可以返回 `(logits, emb)`，主函数会自动拆 |
| 类别数 | 模型带 `num_classes` 或 `out_channels` 属性；没有就回退到 `graph.y.max()+1` |
| 边权重 | 原生算子不支持加权传播时，`forward` 可以不接收 `edge_weight`；主函数会自动按签名调用 |
| 初始化 | 构造参数支持 `in_channels / hidden_channels / num_classes / dropout` 任一种常见命名，主函数会自动匹配 |
| 设备 | 模型和输入 tensor 必须在同一 device 上（主函数会先把图 `to(device)` 再调用） |

不需要提供：训练循环、损失函数、数据加载、指标计算。这些都由 TT-GREB 主函数负责。

## 2. 本仓库已经改了什么

| 文件 | 改动 |
|---|---|
| `tt_greb_reproduce/backbones.py`（新增） | 模型注册表、统一前向、嵌入提取、类别数推断、冻结 |
| `tt_greb_reproduce/tt_greb.py` | 不再访问 `model.conv2`，不再写死 `model(x, ei)`；新增 `edge_weight` 透传 |
| `tt_greb_reproduce/models.py` | 自带 GCN / GraphSAGE 补齐 `out_channels`、`forward_with_embeddings`、`reset_parameters` |
| `run_cora_gcn.py` | 新增 `--backbone gat/gprgnn`、`--model-module`、`--model-kwargs`、`--model-path`；训练与评估都走统一前向 |
| `scripts/eval_adapt.py` | 同上，支持从外部 checkpoint + 外部模型类做单图评估 |
| `tests/test_backbones.py`（新增） | 用四种签名不同的假模型验证接口，不依赖第三部分代码即可跑 |

关键点：`adapt_test_graph(model, graph, ...)` 的输入只要求 `model` 是可调用的
`nn.Module`。数据侧重生本身不改任何模型参数，因此与骨干结构完全解耦。

## 3. 第三部分的两个新模型怎么接

### 3.1 目录约定

第三部分的代码按文档放在仓库根的 `models/` 包：

```text
datamining/
├── models/
│   ├── gcn.py
│   ├── graphsage.py
│   ├── gat.py          # 新增
│   └── gprgnn.py       # 新增
└── TT_GREB_Reproduce/
    └── run_cora_gcn.py
```

默认注册表已经写好：

```python
"gat":    "models.gat:GAT"
"gprgnn": "models.gprgnn:GPRGNN"
```

如果类名不同（例如 `class GATNet`），用 `--model-module models.gat:GATNet` 覆盖即可，
不用改代码。

### 3.2 运行命令

```powershell
# Cora + GAT
python TT_GREB_Reproduce/run_cora_gcn.py `
  --dataset cora --seed 0 --backbone gat `
  --model-path D:\VSCodePrograms\datamining `
  --model-kwargs '{"heads": 4}' `
  --output TT_GREB_Reproduce/results_v2/cora_gat_seed0

# Cora + GPR-GNN
python TT_GREB_Reproduce/run_cora_gcn.py `
  --dataset cora --seed 0 --backbone gprgnn `
  --model-path D:\VSCodePrograms\datamining `
  --model-kwargs '{"K": 10, "alpha": 0.1}' `
  --output TT_GREB_Reproduce/results_v2/cora_gprgnn_seed0
```

`--model-kwargs` 里的内容会原样传给模型构造函数；`in_channels`、`hidden_channels`、
`num_classes`、`dropout` 由主函数按签名自动补齐，不用在 JSON 里重复写。

### 3.3 用已有的 checkpoint 只评估

```powershell
python TT_GREB_Reproduce/scripts/eval_adapt.py `
  --backbone gat --seed 0 --graphs 0,3,7 `
  --model-path D:\VSCodePrograms\datamining `
  --model-kwargs '{"heads": 4}' `
  --ckpt TT_GREB_Reproduce/results_v2/cora_gat_seed0/checkpoint.pt
```

## 4. 已验证到什么程度

`tests/test_backbones.py` 用四个签名不同的假模型覆盖了所有分支，全部通过：

- 只接受 `forward(x, edge_index)`；
- 标准三参数带 `edge_weight`；
- `forward` 永远返回 `(logits, emb)`；
- 仿外部仓库风格：`in_channels` 命名 + `forward_with_embeddings` + `reset_parameters`。

流水线回归（Cora seed 0，短训练）：

| 骨干 | ERM | TT-GREB | delta |
|---|---:|---:|---:|
| 自带 GCN | 0.7932 | 0.8224 | +0.0292 |
| 自带 GraphSAGE | 0.7561 | 0.8196 | +0.0635 |
| 外部 `models.gat:GAT`（模拟第三部分） | 0.7744 | 0.8257 | +0.0513 |

外部 GAT 只在 40 epoch 下训练，delta 仍为正，说明接入路径是通的。正式结果需要按
统一训练脚本跑满 epoch，并在多个种子上复核。

## 5. 常见坑

1. **GAT 的 `edge_weight`**：PyG 的 `GATConv` 不把 `edge_weight` 当边权重用。主函数
   默认传 `None`，不会触发问题；如果以后要做边重加权，GAT 需要改用 `edge_attr` 或
   显式加权邻居，不能直接沿用 GCN 的写法。
2. **GPR-GNN 的传播阶数 `K`**：`alpha` / `K` 属于模型结构参数，必须通过
   `--model-kwargs` 传入，并且和 ERM 训练时保持一致，否则 checkpoint 加载会形状不匹配。
3. **类别数**：如果第三部分模型只暴露 `conv2` 且属性名不叫 `out_channels`，请补一个
   `self.num_classes = num_classes`，或在 `--` 里保证 `graph.y` 完整。
4. **冻结时机**：TT-GREB 在每个测试图开始前调用 `freeze_model(model)`，会执行
   `requires_grad_(False)` + `eval()`。不要让第三部分的模型在前向里做训练态操作。
5. **不要改数据划分**：`split_118` 是唯一入口，模型接入不涉及划分，任何骨干都必须
   用同一个 split，否则 ERM / TT 的数字不可比。

## 6. 第三部分交付后需要做的三件事

1. 把 `models/gat.py`、`models/gprgnn.py` 放进仓库根 `models/`，确认类名与默认注册表
   一致（不一致就用 `--model-module`）。
2. 跑一次 `python TT_GREB_Reproduce/tests/test_backbones.py`，再把两条真实命令各跑
   一个 epoch 级别的小规模冒烟，确认 shape / 设备 / checkpoint 都正常。
3. 把四个骨干 × 四个数据集的结果写进统一 `results_v2/`，再用
   `python TT_GREB_Reproduce/scripts/make_report.py` 汇总成表。
