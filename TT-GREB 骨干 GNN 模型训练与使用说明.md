# TT-GREB 骨干 GNN 模型训练与使用说明

本文面向项目组成员，说明如何训练 GCN 和 GraphSAGE 骨干、运行超参数搜索，以及查看结果。当前覆盖 Cora、Amazon-Photo、Twitch-E、OGB-Arxiv 和 Elliptic。GAT 和 GPR-GNN 仍在规划中。

## 1. 环境与目录

在 Windows PowerShell 中进入项目根目录，激活 `ttgreb` 环境。该环境的解释器路径为 `C:\Anaconda\envs\ttgreb\python.exe`。训练入口使用 PyTorch、PyTorch Geometric、NumPy、PyYAML 和 scikit-learn，读取 Elliptic 稀疏矩阵还需要 SciPy。

```powershell
cd D:\VSCodePrograms\Test-Time_Graph_Rebirth
conda activate ttgreb
python -c "import sys, torch, torch_geometric, numpy, scipy, sklearn, yaml; print(sys.executable); print(torch.__version__); print(torch_geometric.__version__)"
```

若出现 `ModuleNotFoundError: No module named 'yaml'`，在已激活的环境中执行 `python -m pip install PyYAML`。Elliptic 在本机多线程验证中曾出现进程崩溃，设为单线程后通过，可在运行前执行 `$env:OMP_NUM_THREADS = '1'`。

```text
datamining/
├── models/
│   ├── gcn.py
│   ├── graphsage.py
│   └── _elliptic.py
├── configs/          # 每个数据集的固定参数与来源说明
├── search_configs/   # 每个数据集的离散搜索候选值
├── data/             # 本地数据和预处理缓存
├── tests/
├── train.py          # 单种子训练和测试评估
├── tune.py           # 可选的超参数搜索
├── checkpoints/      # 运行后生成
└── results/          # 运行后生成
```

两个配置目录都包含 `cora.yaml`、`amazon_photo.yaml`、`twitch.yaml`、`ogb_arxiv.yaml`、`elliptic.yaml`。一个模型类可用于多个数据集，输入维度和类别编码从训练图确定。各数据集和随机种子分别训练并保存权重。

## 2. 单次训练与评估

以下命令均从项目根目录运行。

```powershell
# 短跑验证
python -m datamining.train --dataset cora --model gcn --epochs 2 --seed 0 --device cpu

# 使用 YAML 中的默认轮数
python -m datamining.train --dataset amazon_photo --model graphsage --seed 0
python -m datamining.train --dataset twitch --model gcn --seed 0
python -m datamining.train --dataset ogb_arxiv --model graphsage --seed 0
python -m datamining.train --dataset elliptic --model gcn --seed 0

python -m datamining.train --help
```

每轮在训练图上更新骨干参数，再计算各验证图指标的算术平均值。验证均值最高的 epoch 对应的权重用于最终测试。验证分数相同时保留较早的 epoch。测试阶段直接用该权重预测，得到 ERM 骨干基线。

`--epochs` 覆盖 YAML 中的 `max_epochs`。省略 `--device` 时优先选择 CUDA，若不可用则使用 CPU。`--seed` 默认是 `0`。固定 YAML 的 `repeats: 10` 记录重复实验设置，每次命令只运行一个种子。需要多种子实验时，可分别传入不同的 `--seed` 并在运行后汇总。

### 固定参数

下表汇总当前 YAML。两个骨干均使用 32 维隐藏层、0.0 Dropout、BatchNorm 和 AdamW。GCN 开启邻接归一化与自环，GraphSAGE 的无权路径使用 PyG 的邻居均值聚合。配置中的 `provenance` 记录 GTrans、EERM 和 TT-GREB 相关来源及待核实项。

| 数据集 | 层数 | 学习率（GCN / GraphSAGE） | 权重衰减 | 最大轮数 | 损失 | 验证指标 |
|---|---:|---|---:|---:|---|---|
| Cora | 2 | 0.001 / 0.01 | 0.001 | 200 | Cross-Entropy | Accuracy |
| Amazon-Photo | 2 | 0.001 / 0.01 | 0.001 | 200 | Cross-Entropy | Accuracy |
| Twitch-E | 2 | 0.01 / 0.01 | 0.001 | 200 | BCEWithLogits | ROC-AUC |
| OGB-Arxiv | 5 | 0.01 / 0.01 | 0 | 500 | Cross-Entropy | Accuracy |
| Elliptic | 5 | 0.01 / 0.01 | 0 | 200 | BCEWithLogits | F1 |

固定 YAML 以 `gcn` 和 `graphsage` 为顶层模型键。`model` 段传入模型构造函数，`training` 段包含 `learning_rate`、`weight_decay`、`optimizer`、`max_epochs`、`validation_metric` 和 `loss` 等字段。可通过 `--config` 指定自定义固定配置。

## 3. 可选的自动超参数搜索

```powershell
python -m datamining.tune --dataset cora --model gcn --epochs 2 --trials 2 --seed 0
python -m datamining.tune --help
```

`tune.py` 先运行固定配置，再以同一个种子从候选值的笛卡尔积中抽取不同组合。`--trials` 默认上限为 10，包含固定基线。每组配置独立训练并选出最佳 epoch，最终只对验证分数最高的配置评估测试图。若配置的验证分数相同，保留先运行的一组。

搜索 YAML 的 `space.model` 包含 `hidden_channels`、`num_layers` 和 `dropout`，`space.training` 包含 `learning_rate` 和 `weight_decay`。也可用顶层模型名提供该模型的专用范围，结构仍为 `model` 和 `training`。用 `--search-config` 指定其他搜索 YAML。

## 4. 本地数据与默认划分

| 数据集 | 划分（训练 / 验证 / 测试） | 数据来源与处理 |
|---|---|---|
| Cora | 1 / 1 / 8 | 从 `data/Planetoid/cora/gen` 读 `0-gcn.pkl` 至 `9-gcn.pkl` 的特征与标签，边来自本地图缓存。 |
| Amazon-Photo | 1 / 1 / 8 | 从 `data/Amazon/Photo/gen` 读取同名序列，边来自本地图缓存。 |
| Twitch-E | 1 / 1 / 5 | 默认顺序为 `ENGB DE ES FR PTBR RU TW`。使用各域共用的特征列索引和无向边。 |
| OGB-Arxiv | 1 / 1 / 1 | 读取本地官方时间划分索引，在各划分内构造无向诱导子图。 |
| Elliptic | 5 / 5 / 33 | 复用已有加载器。训练用时间步 6–10，验证用 11–15，测试用 16–48，损失和 F1 只计算有标签节点。 |

Cora 和 Amazon-Photo 默认使用同一组本地偏移图训练 GCN 和 GraphSAGE，文件编号 0 用于训练、1 用于验证、2–9 用于测试。Twitch 可通过 `--twitch-domains` 依次传入训练域、验证域和五个测试域。Twitch 的 ROC-AUC 默认正类为 1，Elliptic 的 F1 默认正类为 0，可通过 `--positive-class` 覆盖。

`--data-root` 可指向另一份本地数据目录。`--split-file` 可加载可信的 PyTorch 划分文件，其中 `train`、`valid`（或 `val`）、`test` 对应 PyG `Data` 对象或其列表。

## 5. 权重与结果

```text
datamining/checkpoints/<dataset>/<model>/<run>.pt
datamining/results/<dataset>/<model>/<run>.json
```

两份文件使用同一个运行名，其中包含时间戳、随机种子和唯一后缀。PT 文件的 `model_state_dict` 保存最佳 epoch 的权重。JSON 记录实际配置、种子、输入维度和类别编码、最佳 epoch、验证与测试的逐图分数及算术平均值。调参结果还记录各试验的配置、最佳 epoch、验证分数和最终胜出试验。`--output-root` 可更改两个输出目录的共同上级路径。

## 6. 验证状态与论文对照

截至 2026-11-11，已用本地数据跑通 Cora + GCN 的两组调参试验，以及 Amazon-Photo、Twitch-E、OGB-Arxiv、Elliptic + GraphSAGE 的单轮训练与评估。各次验证均生成 PT 和 JSON，输出保存在临时目录。这些短跑用于检查数据读取、训练、选优和输出流程。

正式比较 TT-GREB Table 1 前，需要核对偏移数据的生成版本、训练与验证划分、邻接处理和指标聚合方式。Twitch 默认域顺序、OGB-Arxiv 的单测试图划分以及固定参数与论文的逐项一致性仍待核实。获得可对照的 ERM 结果后，可固定相同骨干权重接入 TT-GREB 测试时图适配，并开展组件消融。

## 7. 扩展模型

新增 GAT 或 GPR-GNN 时，在 `models/` 实现对应模型并在 `train.py` 的 `MODELS` 注册。随后为需要运行的数据集添加固定配置，并按模型需要增加搜索范围。模型须接收节点特征、边索引及可选边权，返回节点分类 logits。新增模型先做小图前向和梯度检查，再用本地数据跑通训练和评价。

## 8. 参考来源与可信度

- [TT-GREB 论文：Test-Time Graph Rebirth](https://openreview.net/)：第 4.1 节（骨干模型、验证选优与重复实验）；Table 1（ERM 与 TT-GREB 结果）。应使用项目持有的 PDF 核对具体表格。
- [GTrans 官方仓库](https://github.com/ChandlerBang/GTrans)：`utils.py::reset_args()`、`gtransform_feat.py` 的骨干初始化与训练逻辑，是当前 YAML 的主要参考。
- [EERM 官方仓库](https://github.com/qitianwu/GraphOOD-EERM)：用于核对人工/跨域/时间分布偏移场景及较早的骨干实现。
- [本组数据与参考代码仓库 `perfect-code`](https://github.com/yyueyue546-tech/datamining/tree/perfect-code)：用于核对现有数据和代码；**尚无证据证明其是 TT-GREB 原作者官方源码**。

> **复现边界**：TT-GREB 明确参考前置研究设置，但“参考 GTrans”不代表已经证实作者运行时的每一个参数、归一化细节和数据预处理都与该仓库完全一致。对无法核实的项目保留 `provenance` 标注、配置快照和实验备注。

