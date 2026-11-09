# TT-GREB 骨干 GNN 模型训练与使用说明

> **文档定位**：供项目组成员配置环境、理解骨干模型组织方式、选择“数据集 × 模型”、执行骨干模型训练并查看结果。后续新增 GAT、GPR-GNN 时继续维护本文档。  
> **当前阶段（2026-11-09）**：GCN、GraphSAGE 通用模型与四份 YAML 配置已编写并通过基础自检；**统一 `train.py` 尚未编写、真实数据集训练尚未开始**。本文档中的训练命令是**拟实现的命令接口**，待程序完成后才能执行。  
> **工作目录（Windows）**：`D:\VSCodePrograms\Test-Time_Graph_Rebirth`

## 1. 总体思路：一个模型文件能用于多个数据集

本项目计划使用四个骨干 GNN（GCN、GraphSAGE、GAT、GPR-GNN）处理四个图数据集（Cora、Amazon-Photo、Twitch-E、OGB-Arxiv）。**模型种类决定 `.py` 模型文件；数据集不决定模型文件的数量。**

- `models/gcn.py`：**一个**通用 GCN 类，可适配四个数据集。
- `models/graphsage.py`：**一个**通用 GraphSAGE 类，可适配四个数据集。
- `models/gat.py`、`models/gprgnn.py`：**后续新增**，每种模型各一个通用实现。
- `configs/*.yaml`：**每个数据集一份配置**，其中分别保存各骨干模型的结构与训练参数。
- `train.py`：**一个统一训练入口**，根据命令行的模型名和数据集名加载对应模型、数据、配置并执行训练；四种模型 × 四个数据集共用这一入口，而不是编写 16 个训练文件。

完整扩展后有 **4 份通用模型代码、4 份数据集配置、1 份通用训练入口**；每个数据集与模型组合在每个随机种子下**单独训练并保存权重**。例如 Cora-GCN 和 Amazon-GCN 共用 `gcn.py`，但不共享训练得到的模型权重。

## 2. Python 环境与依赖

### 2.1 已验证的运行环境

| 项目 | 当前配置 |
|---|---|
| 操作系统 | Windows 11 |
| Conda 环境名 | `ttgreb` |
| Python 解释器 | `C:\Anaconda\envs\ttgreb\python.exe` |
| Python 版本 | 3.10.22 |
| PyTorch | `2.4.0+cpu`（CPU 版） |
| PyTorch Geometric | `2.6.1` |
| 主要依赖 | `torch_sparse`、`torch_scatter`、`numpy`、`scipy`、`scikit-learn`、`ogb`、`pandas`、`matplotlib` |
| 验证情况 | 主要库导入成功；`python -m pip check` 输出 `No broken requirements found.` |

**使用已有环境**（推荐）：在 VS Code 中打开项目根目录，打开 PowerShell 终端：

```powershell
conda activate ttgreb
python -c "import sys, torch, torch_geometric; print(sys.executable); print(torch.__version__); print(torch_geometric.__version__)"
python -m pip check
```

预期解释器为 `C:\Anaconda\envs\ttgreb\python.exe`，PyTorch 为 `2.4.0+cpu`，PyG 为 `2.6.1`。VS Code 可通过 `Ctrl + Shift + P` → **Python: Select Interpreter** 选择该解释器。

**从零搭建环境**（只有没有 `ttgreb` 时才需要；不要删除已有正常环境）：

```powershell
conda create -n ttgreb python=3.10 pip -y
conda activate ttgreb
python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==1.26.4 scipy==1.13.1 scikit-learn==1.5.2
python -m pip install torch-geometric==2.6.1
python -m pip install --no-index torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
python -m pip install ogb googledrivedownloader pandas matplotlib
```

上述命令是本项目使用的基础安装方案；具体安装结果取决于 Windows/Python 架构及官方 Wheel 是否可用。后续 `train.py` 如采用 PyYAML 读取 YAML，请检查并按需安装：

```powershell
python -c "import yaml; print(yaml.__version__)"
# 若提示 ModuleNotFoundError，再执行：
python -m pip install pyyaml
```

> 不能把“成功导入 Python 包”等同于“完整训练经过验证”。当前只完成了基础导入和模型小图测试。

## 3. 项目目录与文件状态

以下以**已从 Codex 执行结果确认的文件**为基础。这里不能直接访问你的本地磁盘，因此尚未对全部真实目录执行 `tree` 核验；**不要把建议新增项当成已经存在的文件**。

```text
D:\VSCodePrograms\Test-Time_Graph_Rebirth\
├── models/                         [已存在]
│   ├── __init__.py                 [已存在]
│   ├── gcn.py                      [已存在：通用 GCN]
│   ├── graphsage.py                [已存在：通用 GraphSAGE]
│   ├── gat.py                      [规划：后续新增]
│   └── gprgnn.py                   [规划：后续新增]
├── configs/                        [已存在]
│   ├── cora.yaml                   [已存在]
│   ├── amazon_photo.yaml           [已存在]
│   ├── twitch.yaml                 [已存在]
│   └── ogb_arxiv.yaml              [已存在]
├── tests/                          [已存在]
│   └── test_backbones.py           [已存在：骨干模型自检]
├── data/                           [位置/结构待与本地核对]
├── data_loaders/                   [规划：数据读取和预处理接口]
├── train.py                        [待编写：统一训练入口]
├── evaluate.py                     [可独立编写，或先集成在 train.py]
├── checkpoints/                    [规划：自动保存最佳模型]
├── results/                        [规划：训练日志和评价指标]
└── TT-GREB_骨干模型训练与使用说明.md  [本文档，建议放根目录]
```

当前原始数据、参考仓库代码和其他本地子目录的实际位置，以 VS Code 文件资源管理器为准。可在项目根目录运行 `tree /F` 或 PowerShell 的 `Get-ChildItem` 核实；补齐后更新以上树形结构。

### 文件职责

| 文件/目录 | 用途 | 当前状态 |
|---|---|---|
| `models/gcn.py` | 实现 GCN 图卷积、输出分类 logits / 节点嵌入、可选动态边权 | 已有初版 |
| `models/graphsage.py` | 实现 GraphSAGE；无权时使用原生 `SAGEConv`，加权路径为扩展 | 已有初版 |
| `configs/<dataset>.yaml` | 该数据集上各模型的结构、训练参数及来源说明 | 已有四份 |
| `tests/test_backbones.py` | 验证前向、梯度、基础数值等价性和不同维度接口 | 已有；Codex 报告四项通过 |
| `data_loaders/` | 加载论文所需训练/验证/测试图，不随意重新划分 | 待编写或整合已有加载器 |
| `train.py` | 读取参数、训练、验证、选最佳 checkpoint、保存日志 | **待编写** |
| `evaluate.py` | 对冻结骨干的测试图进行 ERM 评价 | 待编写或合并入训练入口 |
| `checkpoints/` | 每个模型、数据集、随机种子对应最佳参数 | 训练后产生 |
| `results/` | 配置快照、训练日志、各测试图指标、汇总统计 | 训练后产生 |

## 4. 超参数：由配置决定，不写死在模型里

**模型文件定义“能接受哪些参数”，配置文件决定“本次使用什么值”。** 例如 GCN 构造函数接受 `in_channels`、`out_channels`、`hidden_channels`、`num_layers`、`dropout`、`batch_norm`、`normalize`、`add_self_loops` 等（实际名称以已实现代码为准）。

- **模型结构参数**：层数、隐藏维度、Dropout、BatchNorm、GCN 邻接归一化、自环处理等。由 `train.py` 读取后传入 `GCN(...)` 或 `GraphSAGE(...)`。
- **训练参数**：学习率 `lr`、权重衰减 `weight_decay`、优化器、最多训练轮数 `epochs`、随机种子数 `runs`、验证指标。这些由 `train.py` 用来构造优化器、控制循环和选模型；**不是 GCN 权重本身**。
- **输入维度与类别数**：应从本次实际加载的数据取得，**不能把标准 Cora 的 1433/7 直接当成论文人工偏移 Cora 的固定维度**。

### 4.1 已配置的 GCN / GraphSAGE 参数一览

| 数据集 | 模型 | 层数 | 隐藏维度 | Dropout | BN | 学习率 | 权重衰减 | GCN 邻接处理 |
|---|---|---:|---:|---:|---|---:|---:|---|
| Cora | GCN | 2 | 32 | 0.0 | 开 | 0.001 | 0.001 | 对称归一化、补缺失自环 |
| Cora | GraphSAGE | 2 | 32 | 0.0 | 开 | 0.01 | 0.001 | 原生无权邻居均值 |
| Amazon-Photo | GCN | 2 | 32 | 0.0 | 开 | 0.001 | 0.001 | 同上 |
| Amazon-Photo | GraphSAGE | 2 | 32 | 0.0 | 开 | 0.01 | 0.001 | 原生无权邻居均值 |
| Twitch-E | GCN | 2 | 32 | 0.0 | 开 | 0.01 | 0.001 | 同上 |
| Twitch-E | GraphSAGE | 2 | 32 | 0.0 | 开 | 0.01 | 0.001 | 原生无权邻居均值 |
| OGB-Arxiv | GCN | 5 | 32 | 0.0 | 开 | 0.01 | 0 | 同上 |
| OGB-Arxiv | GraphSAGE | 5 | 32 | 0.0 | 开 | 0.01 | 0 | 原生无权邻居均值 |

上述表格汇总**本次 Codex 创建配置时报告的最终生效值**。其主要参考 GTrans 官方代码 `utils.py::reset_args()` 和 `gtransform_feat.py` 中骨干的实际构造参数；**不等于已证明 TT-GREB 作者在 Table 1 中逐项使用完全相同的配置**。具体值以仓库中实际 YAML 为准，修改配置应同步更新本表。

在已创建的四份配置中，Codex 报告还包含：

| 项目 | 当前配置/规则 | 说明 |
|---|---|---|
| 优化器 | AdamW | 按 GTrans 骨干预训练代码选择 |
| 最大 epoch | Cora、Amazon-Photo、Twitch-E：200；OGB-Arxiv：500 | 与**测试时适配 epoch** 不同 |
| 随机种子重复次数 | 10 | TT-GREB 第 4.1 节报告采用 10 次运行 |
| 验证指标 | Cora / Amazon-Photo / OGB-Arxiv：Accuracy；Twitch-E：ROC-AUC | 以实际标签与评价协议为准 |
| 损失函数 | 前三者中的 Cora / Amazon-Photo / OGB-Arxiv：多分类交叉熵；Twitch-E：BCEWithLogits | 须核对标签形状、掩码和类别编码 |
| 来源标注 | `provenance` | 记录官方代码参数、论文规定与本项目暂定项的区别 |

特别提醒：GTrans 使用的是相关论文的参考骨干设置；其源码与 EERM 衍生分支存在 **GCN 归一化、自环和最后一层边权传递** 等差异。若为了 TT-GREB 的边权适配切换模型行为，必须记录配置，并区分“ERM 无权骨干基线”与“加权图适配”模式。

### 4.2 配置结构示意

**下面仅展示含义，不保证与实际 YAML 字段层级逐字相同。** `train.py` 必须以现有四份 YAML 的实际字段为准，避免重复创建不兼容的配置格式。

```yaml
# configs/cora.yaml —— 结构示意
# 一个数据集文件内分别保存 GCN 和 GraphSAGE

gcn:
  model:
    num_layers: 2
    hidden_channels: 32
    dropout: 0.0
    batch_norm: true
    normalize: true
    add_self_loops: true
  training:
    lr: 0.001
    weight_decay: 0.001

graphsage:
  model:
    num_layers: 2
    hidden_channels: 32
    dropout: 0.0
    batch_norm: true
  training:
    lr: 0.01
    weight_decay: 0.001

# 实际文件中还包含训练项及 provenance 来源字段
```

## 5. 从命令行选择“模型 + 数据集”进行训练

### 5.1 统一入口的预期用法

`train.py` 完成后，计划支持以下命令。**当前尚未实现，因此以下是接口约定而非已验证可执行命令**：

```powershell
# 先进入项目根目录并激活环境
cd D:\VSCodePrograms\Test-Time_Graph_Rebirth
conda activate ttgreb

# Cora + GCN
python train.py --dataset cora --model gcn

# Cora + GraphSAGE
python train.py --dataset cora --model graphsage

# Amazon-Photo + GCN / GraphSAGE
python train.py --dataset amazon_photo --model gcn
python train.py --dataset amazon_photo --model graphsage

# Twitch-E + GCN / GraphSAGE
python train.py --dataset twitch --model gcn
python train.py --dataset twitch --model graphsage

# OGB-Arxiv + GCN / GraphSAGE
python train.py --dataset ogb_arxiv --model gcn
python train.py --dataset ogb_arxiv --model graphsage
```

后续增加 GAT、GPR-GNN 时，预期调用方式保持不变：

```powershell
# 只有新增模型及其配置并注册到训练入口后才可运行
python train.py --dataset cora --model gat
python train.py --dataset cora --model gprgnn
```

### 5.2 以 Cora + GCN 为例：`train.py` 应自动执行什么

```text
命令行选择：--dataset cora --model gcn
                  │
                  ├─ 加载 configs/cora.yaml 中 GCN 的 model / training 配置
                  ├─ 加载 Cora 的训练图、验证图、测试图（按已有固定划分）
                  ├─ 从实际数据确定输入特征维度和输出类别数
                  ├─ 构造 models/gcn.py 中的 GCN，并初始化优化器
                  │
                  ├─ 训练图：前向传播 → Loss → 反向传播 → 优化器更新权重
                  ├─ 验证图：评价模型 → 选择验证指标最优的 epoch
                  ├─ 保存最佳模型权重及配置快照
                  │
                  ├─ 用最佳模型、固定权重分别评价测试图（ERM 基线）
                  └─ 保存各图指标、各随机种子指标及统计汇总
```

**区分超参数与模型参数**：一次训练的网络层数、学习率等一般固定；训练循环自动更新卷积层的权重和偏置。不同超参数配置需要单独训练，并**只根据验证集**决定，不能借助测试图调整参数。

**区分最佳 epoch 与最佳超参数**：`train.py` 可以自动依据验证指标保存最佳 checkpoint；但若没有额外实现网格搜索，它**不会自行遍历所有学习率、隐藏维度**。本项目首先固定论文/参考实现给出的参数，不进行大规模搜索。

## 6. 训练完成后应该输出什么？

GCN/GraphSAGE **前向传播直接输出的是每个节点的分类分数（logits）**，不是 Accuracy；训练脚本把 logits 与真实标签比较，计算 Loss、预测类别和评价指标。

建议 `train.py` 至少产生以下文件（输出名称以实际实现为准）：

```text
checkpoints/
└── cora/
    └── gcn/
        ├── seed_0_best.pt         # 训练好的模型参数
        └── ...                    # 其他随机种子
results/
└── cora/
    └── gcn/
        ├── train_log.csv          # 每轮训练 Loss、训练/验证指标
        ├── test_metrics.csv       # 各测试图的最终指标
        ├── summary.csv            # 跨随机种子均值、标准差
        └── config_used.yaml       # 本次实际生效配置及随机种子
```

| 输出 | 含义 | 主要用途 |
|---|---|---|
| 训练 Loss | 训练图上的监督损失 | 检查是否收敛 |
| Train Accuracy / ROC-AUC | 在训练数据上的评价 | 辅助诊断 |
| Val Accuracy / ROC-AUC | 验证图上的评价 | **选择最佳 checkpoint** |
| `.pt` 权重文件 | 骨干模型学习到的参数 | 后续 TT-GREB 加载并冻结 |
| Test Accuracy / ROC-AUC | 最佳模型在测试图上的表现 | 与论文 **ERM** 基线对照 |
| 均值 ± 标准差 | 多随机种子重复实验统计 | 与论文汇总表比较 |

**重要**：只有每个随机种子在其训练期间分别选择 checkpoint、之后才评价测试图，才能保证验证与测试分离。统计方法（按图、按随机种子的聚合顺序）应与论文保持一致，不应凭空假设。

## 7. 数据集、任务与评价指标

| 数据集 | 分布偏移场景 | 当前划分约定 | 骨干基线指标 |
|---|---|---|---|
| Cora | 人工特征偏移 | 1 训练图 + 1 验证图 + 8 测试图 | Accuracy |
| Amazon-Photo | 人工特征偏移 | 1 + 1 + 8 | Accuracy |
| Twitch-E | 跨域偏移 | 1 + 1 + 5（以数据加载核验为准） | ROC-AUC |
| OGB-Arxiv | 时间偏移 | 1 + 1 + 3（以数据加载核验为准） | Accuracy |

不同数据集可能采用不同的标签编码、损失函数、节点掩码或图预处理。**通用训练入口 ≠ 所有数据集使用完全相同的数据加载代码**。

Cora 特别注意：论文人工偏移版本与标准 Planetoid Cora 的节点数和类别数存在差异；此前发现论文统计为 **2703 节点、10 类**，而标准 Cora 常见为 **2708 节点、7 类**。在启动正式训练前，必须检查本地 `.pkl` 的实际数据内容、生成版本和标签含义；不能直接用标准 Cora 的准确率与论文人工偏移结果比较。

## 8. 与 TT-GREB 论文结果对照的顺序

本阶段只训练**原始骨干 GNN（ERM）**，不要提前运行 TT-GREB 的测试时图适配。

1. **首先复现 ERM 基线**：加载最佳权重，未经适配直接预测测试图，汇总 Accuracy / ROC-AUC；与 TT-GREB 论文 **Table 1 的 ERM** 对照。
2. **再接入 TT-GREB**：固定相同骨干权重，在测试时优化图特征/结构，比较适配前后的预测表现。
3. **最后消融实验**：核对 TT-GREB 各组件的实际增益。

论文 Cora 的 ERM 结果可作为初步参照（须先确认数据集和评价口径一致）：GCN **87.49 ± 7.97%**，GraphSAGE **98.48 ± 3.68%**。训练结果接近论文可以增加可信度，但**不能证明实现与论文作者逐行相同**；差异较大时，应先核查数据构造、邻接归一化、优化器及训练/验证协议，不能直接依测试分数调参。

## 9. 添加 GAT 与 GPR-GNN 的扩展步骤

目前仅实现 GCN 和 GraphSAGE。后续增加两类模型时：

1. 分别新增 **一份** `models/gat.py` 与 **一份** `models/gprgnn.py`，保持统一 `forward` 等调用约定，必要时增加节点嵌入和可微边权接口。
2. 四份现有 YAML 中分别新增 `gat`、`gprgnn` 配置段；**不增加新的数据集配置文件**。
3. 在统一模型工厂/`train.py` 注册模型名 `gat`、`gprgnn`，不复制训练循环。
4. 验证特有超参数：GAT 的注意力头数、头拼接；GPR-GNN 的传播阶数、传播权重初始化、`alpha` 等。优先核对 GTrans 官方源码与论文附录，不能强行套用 GCN 的层数。
5. 对每种新模型先进行小图前向/反向测试，再进行 Cora 的真实数据 smoke test，最后才开展完整重复训练。

最终四个通用模型 × 四个数据集 = **16 种实验组合（每个随机种子）**，但仍然是 **4 个模型文件 + 4 个 YAML + 1 个训练入口**。**16 种组合并不意味着只训练一次模型，16 种组合需要分别训练权重。**

## 10. 训练前检查清单与常见问题

- [ ] 已激活 `(ttgreb)`，且 `sys.executable` 指向 `C:\Anaconda\envs\ttgreb\python.exe`。
- [ ] 已安装 YAML 解析库（如果训练脚本用 PyYAML）。
- [ ] `train.py` 的模型名称与 YAML 字段一致；能够打印**最终生效配置**。
- [ ] 数据加载器复用原始训练/验证/测试图，不重新随机划分。
- [ ] Cora 的节点、特征、类别和人工偏移构造已经核对。
- [ ] GCN 的归一化和自环处理与选定参考协议一致。
- [ ] 在训练前已确认损失函数与标签编码匹配。
- [ ] 训练仅使用训练图标签，验证选 checkpoint，测试图只用于最后评价。
- [ ] 每个随机种子重新初始化模型和优化器，保存独立权重。
- [ ] 完成 1 次小规模 smoke test 后再运行完整 10 次实验。
- [ ] 将模型权重、日志、测试指标和配置快照保存到结果目录。

**常见问题**：

- 找不到 `train.py`：目前该文件仍待编写；第五节命令不能立即执行。
- `ModuleNotFoundError: yaml`：先运行 `python -m pip install pyyaml`。
- 输出层类别数不匹配：核查真实标签和 YAML 中有无硬编码类别数。
- 不同实验分数差异大：首先核查数据生成版本、标签与邻接归一化；不要先调测试集。
- Windows 弹出 Python 内存访问错误：可先测试解释器与依赖导入，并定位是哪条测试/训练命令触发；仅凭一次弹窗不能认定环境整体损坏。
- OGB-Arxiv CPU 训练慢或内存压力大：先做小图和 Cora 测试，再评估是否需改用服务器或优化数据加载。

## 11. 参考来源与可信度

- [TT-GREB 论文：Test-Time Graph Rebirth](https://openreview.net/)：第 4.1 节（骨干模型、验证选优与重复实验）；Table 1（ERM 与 TT-GREB 结果）。应使用项目持有的 PDF 核对具体表格。
- [GTrans 官方仓库](https://github.com/ChandlerBang/GTrans)：`utils.py::reset_args()`、`gtransform_feat.py` 的骨干初始化与训练逻辑，是当前 YAML 的主要参考。
- [EERM 官方仓库](https://github.com/qitianwu/GraphOOD-EERM)：用于核对人工/跨域/时间分布偏移场景及较早的骨干实现。
- [本组数据与参考代码仓库 `perfect-code`](https://github.com/yyueyue546-tech/datamining/tree/perfect-code)：用于核对现有数据和代码；**尚无证据证明其是 TT-GREB 原作者官方源码**。

> **复现边界**：TT-GREB 明确参考前置研究设置，但“参考 GTrans”不代表已经证实作者运行时的每一个参数、归一化细节和数据预处理都与该仓库完全一致。对无法核实的项目保留 `provenance` 标注、配置快照和实验备注。

---

**维护建议**：完成 `train.py` 后，应由实现者补充真实的 `--help` 输出、实际 YAML 字段结构、真实目录树、训练命令示例与首次 Cora 测试结果。新增 GAT / GPR-GNN 后，更新第 4、5、9 节，使本文档始终可以交付给其他组员直接使用。
