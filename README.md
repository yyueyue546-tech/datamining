# TT-GREB 本地实验与项目交接说明

本文说明当前目录、脚本分工、已完成实验、保存的证据，以及目录迁移后尚需调整的路径，供下一位成员继续论文复现。

文档更新日期：2026-10-28（北京时间）。历史实验日期：2026-10-26。

当前已完成的实验是：**标准 Cora 单图上的 GCN 训练 → 验证选模 → 冻结 GCN → 原始图测试 → 测试时重加权 → 对比分数，并检查同种子重复性。**

当前完成的是工程链路验证。数据采用标准 Cora 单图节点掩码协议，适配目标仍为仓库中的替代实现；尚未完成作者数据划分、最终论文公式、优化方式和完整实验设置的核对，不能将现有分数作为正式论文复现结果。

## 1．当前目录中的文件怎样分工

以下路径均相对于本 README 所在的仓库根目录 datamining/。

```text
datamining/
├── README.md                              本文：当前项目与交接说明
├── .gitignore
├── data/                                  数据输入与缓存
│   └── Cora/
│       ├── raw/                           本地原始文件
│       └── processed/data.pt              上次实际使用的单图缓存
├── docs/                                  文档及环境记录
│   ├── ENVIRONMENT.md
│   ├── EXPERIMENT_FORMULAS_AND_CODE.md
│   ├── IMPLEMENTATION_SUMMARY.md
│   ├── INIT.md
│   ├── README.md                          原内层说明，注意旧路径
│   ├── REPRODUCTION_FEISHU.md              历史实验报告
│   ├── requirements.txt
│   └── requirements.windows-validated.txt
├── results/                               历史实验及验收记录
│   ├── cora_masked_gcn_seed0/
│   ├── cora_masked_gcn_seed0_repeat/
│   ├── cora_masked_gcn_seed0_200_20/
│   ├── cora_masked_gcn_seed0_200_20_repeat/
│   ├── verification.json
│   ├── adaptation_invariants.json
│   └── checkpoint_verification.json
├── scripts/                               运行、检查和报告入口
│   ├── check_env.py
│   ├── main.py
│   ├── prepare_standard_cora_smoke.py
│   ├── inspect_split.py
│   ├── run_cora_gcn.py
│   ├── verify_local_reproduction.py
│   ├── verify_adaptation_invariants.py
│   ├── verify_saved_checkpoints.py
│   ├── make_reproduction_report.py
│   └── local_fixes.patch
└── TT_GREB_Reproduce/
    └── tt_greb_reproduce/                  核心 Python 包
        ├── __init__.py
        ├── data.py
        ├── models.py
        └── tt_greb.py
```

其他数据集仍保存在 data/ 中；有文件或加载代码，不代表已完成相应实验。Python 自动生成的 __pycache__/ 不属于手写实现。

上次实际运行入口是旧位置 TT_GREB_Reproduce/run_cora_gcn.py。现在 scripts/run_cora_gcn.py 已替换为支持节点 mask 的修复版。历史记录保留原路径，新目录的路径适配尚未全部完成，见第 6 节。

## 2．scripts 中每个文件的作用、输入和输出

| 文件 | 作用 | 输入 | 输出 |
|---|---|---|---|
| [check_env.py](scripts/check_env.py) | 检查基础开发环境、张量计算和反向传播 | 当前 Python 环境，无命令行参数 | 终端版本、计算结果、梯度和通过信息；失败时断言报错 |
| [main.py](scripts/main.py) | 读取 Cora、Amazon-Photo、OGB-Arxiv、Twitch、Elliptic，打印统计，为部分数据构造子图 | 数据文件或加载器缓存，路径写在代码中 | 终端统计、内存中的图对象；没有完整训练与适配结果 |
| [prepare_standard_cora_smoke.py](scripts/prepare_standard_cora_smoke.py) | 按节点编号将标准 Cora 分成十部分并取诱导子图，导出自检划分 | --root 数据位置；--output 必填导出位置 | 含 train、valid、test 的 .pt 文件，图数量为 1/1/8 |
| [inspect_split.py](scripts/inspect_split.py) | 校验多图 split 并打印统计 | 位置参数 split；--expected-test-graphs 默认 8 | 终端 JSON：图数量、节点数、边数、特征维度、类别数等；不主动写文件 |
| [run_cora_gcn.py](scripts/run_cora_gcn.py) | 执行一次训练、选模、冻结、ERM、适配和适配后评分 | 数据来源、训练参数、适配步数、seed、设备、输出目录 | checkpoint.pt、config.json、summary.json、test_results.csv 和终端汇总 |
| [verify_local_reproduction.py](scripts/verify_local_reproduction.py) | 准备缓存，自动执行四次 masked 实验，检查输出及重复性 | 本地 Cora raw 或 processed 数据、同目录主脚本、当前解释器 | 四次实验产物、各次 run.log、总记录 verification.json |
| [verify_adaptation_invariants.py](scripts/verify_adaptation_invariants.py) | 在小图上检查标签依赖和 GCN 冻结行为 | 内部生成的 6 节点图、初始化 GCN、固定随机种子 | adaptation_invariants.json 和终端检查结果 |
| [verify_saved_checkpoints.py](scripts/verify_saved_checkpoints.py) | 重载保存的 GCN，复算验证与 ERM 分数 | verification.json、其中记录的缓存路径、四份 checkpoint | checkpoint_verification.json 和通过提示 |
| [make_reproduction_report.py](scripts/make_reproduction_report.py) | 将实验记录填入中文 Markdown 报告模板 | verification.json 中的环境、图统计、结果及耗时 | REPRODUCTION_FEISHU.md |
| [local_fixes.patch](scripts/local_fixes.patch) | 记录上次两处代码修复 | 对应原版本的代码 | 应用时修改代码；本身不执行实验，不生成分数 |

### 主入口的参数与辅助工具的范围

run_cora_gcn.py 的三种数据来源互斥，必须选择一种：

| 参数 | 含义 |
|---|---|
| --processed-cora | 加载标准 Cora 单图缓存，按 train/val/test mask 训练和评分；上次四次均使用此入口 |
| --split | 加载独立训练图、验证图和测试图列表 |
| --smoke-standard-cora | 自动构造标准 Cora 的 1/1/8 smoke 划分 |

其他主要参数：--output 必填；--epochs 默认 200；--adapt-steps 默认 20；--seed 默认 0；--hidden-dim 默认 64；--lr 默认 0.01；--weight-decay 默认 0.0005；--device 可指定 CPU 或 CUDA。

prepare_standard_cora_smoke.py 和 inspect_split.py 服务于多图 split 路径，没有参与上次四次 masked 实验的主流程。smoke 划分用于检查程序链路，不能视为作者论文划分。inspect_split.py 不能直接把标准 Planetoid processed 元组缓存当作 split 字典检查。

check_env.py 检查 Python 3.11、CPU PyTorch 和基础依赖，不检查完整 PyG/Cora 实验。main.py 也不是完整训练实验入口。

### 上次工作的来源与改动范围

- 新增了三个 verify 脚本和 make_reproduction_report.py；check_env.py 是此前配置环境时新增的脚本。
- 沿用并修复了实验入口和 data.py：测试循环逐一绑定图与 mask；直接读取指定的本地 Cora 缓存，避免隐式推导目录并重新下载。
- 沿用了已有 GCN、适配器和 split 工具，上次没有重写适配目标函数。
- local_fixes.patch 是历史修复记录；当前修复版已包含相应改动，不应重复应用。

## 3．核心代码三个模块分别负责什么

核心包位于 TT_GREB_Reproduce/tt_greb_reproduce/，通常由入口脚本导入。

| 模块 | 职责 | 输入 | 输出 |
|---|---|---|---|
| [data.py](TT_GREB_Reproduce/tt_greb_reproduce/data.py) | PyG 缓存兼容、数据加载与校验、smoke 划分、设备迁移 | 缓存或 split 路径、图对象、设备 | PyG Data 或含 train/valid/test 的 GraphSplits |
| [models.py](TT_GREB_Reproduce/tt_greb_reproduce/models.py) | 定义两层 GCN，支持边权和隐藏表示输出 | x: [N,F]、edge_index: [2,E]、可选 edge_weight: [E] | logits: [N,C]；return_embedding=True 时同时返回隐藏表示 |
| [tt_greb.py](TT_GREB_Reproduce/tt_greb_reproduce/tt_greb.py) | 四个 MLP、图重加权、适配损失与测试时优化 | 冻结 GCN、原始图、步数、学习率及正则参数 | AdaptationResult：重加权图、步数、最终损失 |

__init__.py 当前用于组织 Python 包，没有独立实验逻辑。

GCN 结构为 GCNConv → ReLU → Dropout → GCNConv，本次维度为 1433 → 64 → 7，dropout=0.5。

适配器包含四个 MLP：

| 模块 | 输入 | 输出 |
|---|---|---|
| prototype_node | 节点特征 [N,F] | prototype 特征权重 [N,F] |
| environment_node | 节点特征 [N,F] | environment 特征权重 [N,F] |
| prototype_edge | 边两端特征拼接 [E,2F] | prototype 边权 [E] |
| environment_edge | 边两端特征拼接 [E,2F] | environment 边权 [E] |

同一对象的两类权重按 W = P + (1-P) * E 组合。节点特征变为 x * W_node，最终边权为 W_edge。当前实现保持 edge_index 不变，不新增或硬删除节点、边。

适配默认使用 Adam 学习率 0.01、alpha=0.1、beta=0.1，两个比例目标均为 0.5。损失包含余弦相似度、负环境差异项和权重均值比例正则。四个 MLP 使用一个优化器联合更新，不能据此声明已实现论文全部损失或交替优化。

## 4．上次实验具体是怎么做的——对应脚本说明

### 第一步：准备并检查标准 Cora 数据

**负责脚本：verify_local_reproduction.py；调用模块：data.py。**

查找 data/Cora/processed/data.pt；缺失时调用 read_planetoid_data() 从本地 raw 生成缓存，再通过 load_masked_cora() 加载。

| 数据项 | 本次值 |
|---|---:|
| 图数量 | 1 |
| 节点数 | 2708 |
| 特征维度 | 1433 |
| 类别数 | 7 |
| edge_index 边条目数 | 10556 |
| 训练节点 | 140 |
| 验证节点 | 500 |
| 测试节点 | 1000 |
| 其余节点 | 1068 |

检查三个 mask 互不重叠，并记录缓存 SHA-256。这是传导式节点分类：完整图的特征和连接参与传播，仅训练节点标签进入训练损失，验证标签用于选模，测试标签用于评分。其余节点保留在图中。

本步骤不需要 main.py、prepare_standard_cora_smoke.py 或 inspect_split.py。

### 第二步：启动一次实验，训练 GCN 并选择最佳参数

**负责脚本：run_cora_gcn.py；由 verify_local_reproduction.py 启动。**

组织脚本传入 --processed-cora、--epochs、--adapt-steps、--seed 0、--device cpu 和独立 --output。主脚本调用 train_masked()。

每轮流程：完整图前向传播 → 取 140 个训练节点计算交叉熵 → 反向传播更新 GCN → 在 500 个验证节点上计算 Accuracy。

验证 Accuracy 严格提高时更新内存中的最佳参数。训练结束后恢复最佳参数、冻结 GCN，保存 checkpoint.pt。不会逐轮保存所有模型，本次也没有保存最佳轮次。

GCN 使用隐藏维度 64、dropout=0.5、Adam 学习率 0.01、weight decay=0.0005。

### 第三步：计算原始图上的 ERM 测试分数

**负责脚本：run_cora_gcn.py。**

用选出的冻结 GCN 在原始图预测，通过 accuracy(model, graph, graph.test_mask) 只对 1000 个测试节点评分。四次 ERM Accuracy 都为 79.60%，不需要另一个测试入口。

### 第四步：执行适配，再计算适配后分数

**入口脚本：run_cora_gcn.py；实际计算：tt_greb.py 的 adapt_test_graph()。**

```text
原始图 + 当前四个 MLP
        ↓
生成特征权重和边权，构造重加权图及其他视图
        ↓
通过冻结 GCN 计算隐藏表示
        ↓
计算适配损失并反向传播
        ↓
更新四个 MLP，GCN 参数不变
```

5 步表示更新 MLP 5 次，20 步表示更新 20 次。每一步用原始图与当前参数重新生成权重，不反复累乘上一轮的特征。真实标签不进入适配损失。

循环结束后返回最终图，主脚本使用同一份 GCN 在相同测试节点上评分。适配损失下降不保证测试 Accuracy 提高。

### 第五步：保存本次实验产物

**负责脚本：run_cora_gcn.py；日志由 verify_local_reproduction.py 保存。**

主脚本向 --output 写 checkpoint.pt、config.json、summary.json、test_results.csv。组织脚本将子进程标准输出与错误输出保存到 run.log。

### 第六步：两组配置运行四次，并检查重复性

**负责脚本：verify_local_reproduction.py。**

第二至第五步构成一次实验，组织脚本依次执行四次：

| 顺序 | 目录 | 训练轮数 / 适配步数 | ERM | 适配后 | 变化（百分点） |
|---|---|---|---:|---:|---:|
| 1 | cora_masked_gcn_seed0 | 50 / 5 | 79.60% | 79.20% | -0.40 |
| 2 | cora_masked_gcn_seed0_repeat | 50 / 5 | 79.60% | 79.20% | -0.40 |
| 3 | cora_masked_gcn_seed0_200_20 | 200 / 20 | 79.60% | 79.80% | +0.20 |
| 4 | cora_masked_gcn_seed0_200_20_repeat | 200 / 20 | 79.60% | 79.80% | +0.20 |

每次都启动独立子进程、重新初始化并从头训练，均为 seed=0。累计 GCN 训练 500 轮、适配 50 步，不是一个模型连续训练 500 轮。

短配置先检查完整链路，长配置按启动说明增加计算量。两档同时改变了训练轮数与适配步数，无法单独归因于其中一项；它们不是经过搜索确定的最佳配置。

组织脚本检查退出状态、必要输出、evaluation_protocol=node_masks 和 paper_split_verified=false，再比较第 1/2 次、第 3/4 次的 summary 与 checkpoint 参数张量。两组均完全一致，证据保存为 verification.json。这证明本机当前环境下同种子可重复，不代表多种子统计稳定性或跨平台一致性。

历史实测环境：Python 3.11.17、PyTorch 2.14.1+cpu、PyG 2.6.1、CPU。它与原依赖文件要求的 PyTorch 2.6.0 不同；环境快照不代表论文原环境已验证。

### 第七步：检查标签依赖与 GCN 冻结行为

**负责脚本：verify_adaptation_invariants.py。**

构造独立 6 节点合成图，用原标签与翻转标签分别适配 1 步，并重置相同适配随机种子。检查特征输出一致、边权一致、GCN 参数不变及全部冻结。四项均为 true，保存 adaptation_invariants.json。合成图检查不产生 Cora Accuracy，不混入四次分数。

### 第八步：重载 checkpoint，复核验证与 ERM 分数

**负责脚本：verify_saved_checkpoints.py。**

读取 verification.json 和对应缓存，创建相同 GCN，依次加载四份 checkpoint，在原始图复算验证与 ERM Accuracy，并与记录比较。四份均一致：验证 79.00%，ERM 79.60%。输出 checkpoint_verification.json。

此脚本不重跑适配后评分；适配器未保存在 checkpoint 中。

### 第九步：生成实验报告

**负责脚本：make_reproduction_report.py。**

读取 verification.json，将环境、图统计、分数和耗时填入模板，生成 REPRODUCTION_FEISHU.md，不重新训练或评分。模板包含另外两类检查的描述，但脚本没有读取那两份检查 JSON，交接时应一并核对原始证据。

### 实验步骤与脚本速查

| 工作 | 负责脚本 |
|---|---|
| 准备缓存、检查数据、安排四次实验 | verify_local_reproduction.py |
| 训练、选模、冻结、ERM、适配及适配后评分 | run_cora_gcn.py |
| 保存单次模型、配置和分数 | run_cora_gcn.py |
| 捕获日志、检查输出、比较重复结果 | verify_local_reproduction.py |
| 检查标签依赖与 GCN 冻结 | verify_adaptation_invariants.py |
| 重载模型复核分数 | verify_saved_checkpoints.py |
| 根据记录生成报告 | make_reproduction_report.py |

## 5．实验保存了什么

四次历史实验保存在仓库根目录 [results/](results/)。每次包含：

| 文件 | 内容 | 限制 |
|---|---|---|
| checkpoint.pt | 验证集选出的最佳 GCN 权重和偏置，共 92231 个参数 | 不含数据、适配器参数、优化器状态，不是完整续训存档 |
| config.json | seed、轮数、隐藏维度、学习率、weight decay、适配步数、设备、版本和来源 | 未完整记录全部适配超参数，需结合代码默认值 |
| summary.json | 验证分数、ERM 与适配分数、协议、适配步数、最终损失 | 无逐轮曲线、最佳轮次及逐节点预测 |
| test_results.csv | test_graph、erm_accuracy、tt_greb_accuracy | 本次一行代表一个图条目，分数来自 1000 个测试节点 |
| run.log | 子进程输出及警告 | 无完整逐轮训练损失和 Accuracy 曲线 |

总证据及文档：

- [verification.json](results/verification.json)：环境、缓存路径和 SHA-256、完整历史命令、结果、耗时、重复性比较。
- [adaptation_invariants.json](results/adaptation_invariants.json)：独立小图的四项行为检查。
- [checkpoint_verification.json](results/checkpoint_verification.json)：四份 GCN 的验证与 ERM 分数复核。
- [历史实验报告](docs/REPRODUCTION_FEISHU.md)：过程和结果，仍含旧目录路径。
- [公式与代码说明](docs/EXPERIMENT_FORMULAS_AND_CODE.md)：已有实现解释，不能代替最终论文一致性核验。
- [实测依赖快照](docs/requirements.windows-validated.txt)：上次本机环境记录。

在 node_masks 协议下，summary 中 train/valid/test 的图计数指向同一张图，不能相加成三张独立图。smoke_only=false 仅表示未使用 smoke 参数，不代表正式复现成功。当前目标包含负环境差异项，负 final_loss 本身不自动表示错误。

当前没有保存适配器参数、最终重加权图、逐节点预测和完整训练曲线。结果支持程序可运行及同种子可重复；50/5 下降、200/20 略升，尚未证明稳定增益。

## 6．目录更新后，仍需同步调整的路径

**已完成文件归类与主入口版本替换，尚未完成迁移后的运行验证。以下为读取现有源码确认的问题；本 README 没有同时修改脚本。**

| 位置 | 当前行为 | 需要调整 |
|---|---|---|
| 主脚本与验证工具导入 | 导入 tt_greb_reproduce，包位于 TT_GREB_Reproduce 内 | 统一包安装或 Python 搜索路径；verify_local_reproduction.py 插入 scripts 路径并不能解决核心包位置问题 |
| verify_local_reproduction.py | RESULTS = ROOT / results，ROOT 为脚本目录 | 改用仓库根目录 results；REPO = ROOT.parent 在当前布局下正确，Cora 路径正确 |
| verify_adaptation_invariants.py | 写脚本目录下的 results/adaptation_invariants.json | 改为仓库根目录 results |
| verify_saved_checkpoints.py | 从脚本目录 results 读取证据和模型 | 改用仓库根目录 results；检查历史缓存绝对路径在新机器是否有效 |
| make_reproduction_report.py | 从脚本目录 results 读取，在脚本目录写报告 | 输入改为根目录 results，输出明确指向 docs；更新模板旧路径 |
| main.py 的 Twitch、Elliptic 路径 | 根据脚本位置拼接 data | 移动后指向 scripts/data，应改为仓库根目录 data |
| main.py 的 Cora、Amazon、OGB 路径 | 相对于当前工作目录 | 统一明确的数据根目录，避免启动目录改变位置 |
| prepare_standard_cora_smoke.py 及 smoke 加载默认值 | 相对路径 TT_GREB_Reproduce/data | 明确 smoke 缓存目录，避免另建混淆的数据副本 |
| local_fixes.patch | 引用原目录下的主脚本和数据模块 | 保留作历史记录；应用前核对版本与路径，不重复应用 |
| docs 中旧说明和依赖文件 | 历史命令使用旧目录；依赖已移入 docs | 更新运行命令及依赖文件引用，不假设根目录仍有 requirements.txt |
| verification.json 的 command/cache | 保留历史机器绝对路径与旧输出位置 | 保留历史证据，另给新目录使用方法；迁移机器时重新定位缓存，不伪改历史命令 |

路径修复后应先检查包导入与主脚本 --help，再使用新的独立输出目录进行运行验证，避免覆盖历史证据。本文不提供未经验证的新目录一键运行命令。

下一位成员建议依次完成：

1. 修复上述路径并验证新目录入口，保留历史结果。
2. 核验论文/作者的数据划分、特征处理、标签规则与评分协议。
3. 核对并补齐论文损失、优化方式、超参数及必要保存内容。
4. 按论文设置开展不同随机种子、其他数据集、骨干和 baseline 实验。
5. 汇总均值与波动，完成论文表格对照并更新报告。



## 7．docs 中六份文档的内容说明

本节介绍 docs/ 中仍保留的六份 Markdown 文档，帮助接手者按需要查阅。根目录 README.md 是当前项目总入口；docs/README.md 是迁移前的内层使用说明，两者职责不同。

| 文档 | 主要讲什么 | 什么时候阅读 | 适用阶段与注意事项 |
|---|---|---|---|
| [ENVIRONMENT.md](docs/ENVIRONMENT.md) | Conda 环境创建、CPU PyTorch 安装、镜像源切换、VS Code 解释器选择、基础计算检查及依赖导出 | 配置本地开发环境、理解当时如何安装和检查依赖时 | 主要记录早期基础环境，未完整覆盖后来补装的 PyG 和 SciPy；命令中的脚本与依赖路径尚未随目录迁移更新，结合本 README 第 6、7 节阅读 |
| [EXPERIMENT_FORMULAS_AND_CODE.md](docs/EXPERIMENT_FORMULAS_AND_CODE.md) | 数据字段与 mask、GCN 公式、训练损失、验证选模、参数冻结、四个 MLP、重加权、适配损失、梯度路径、超参数和输出含义 | 理解代码如何计算、修改适配模块、准备对照论文公式时 | 解释当前实现，尚未证明与最终论文一致；其中目录树、结果位置等仍为旧版，公式主要内容已整合到本 README 第 8 节 |
| [IMPLEMENTATION_SUMMARY.md](docs/IMPLEMENTATION_SUMMARY.md) | 早期实现目标、核心文件职责、smoke 启动方式，以及数据、目标函数和实验覆盖范围的缺口 | 追溯最初实现计划和代码骨架时 | 主要围绕 1/1/8 smoke／多图流程，不完整反映后来实际完成的四次 masked 实验；当前进度以根 README 和验收记录为准 |
| [INIT.md](docs/INIT.md) | 新机器获取代码、创建环境、安装依赖、启动标准 Cora mask 或独立多图实验，以及成功条件和后续任务 | 了解原先为接手者设计的启动顺序时 | 多处路径过期；原依赖要求与本机实测环境需要区分；图数量为 1/1/8 不足以证明满足正式论文复现条件 |
| [README.md（docs 内）](docs/README.md) | 原内层英文使用说明，包括依赖安装、smoke 流程、标准 Cora mask 入口、split 数据格式和输出文件 | 查阅原程序的参数约定、数据接口和历史使用方式时 | 属于旧版使用说明，含旧目录和其他机器绝对路径；不作为当前项目总入口，避免与根 README 混淆 |
| [REPRODUCTION_FEISHU.md](docs/REPRODUCTION_FEISHU.md) | 2026-10-26 本地实测过程：代码来源、环境、数据指纹、问题修复、四次运行、结果、耗时、验收和剩余工作 | 确认上次实际执行了什么、为什么修复、实验结果及证据在哪里时 | 是历史实验记录；原命令和结果路径对应迁移前目录，应保留其历史含义，不能直接作为新目录的运行指南 |

### 阅读顺序

1. 先读根目录 README，了解当前状态、脚本分工及迁移问题。
2. 配置环境时查 ENVIRONMENT.md，并结合本 README 第 7 节核对实际版本。
3. 理解和继续实现算法时查 EXPERIMENT_FORMULAS_AND_CODE.md。
4. 核对已完成实验时查 REPRODUCTION_FEISHU.md 和 results/ 中的三份验收 JSON。
5. IMPLEMENTATION_SUMMARY.md、INIT.md 和 docs/README.md 主要用于追溯旧版说明，其中有效内容已并入根 README。

