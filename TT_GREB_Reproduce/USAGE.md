# TT-GREB 使用说明（从零跑通）

这份文档面向第一次拿到仓库的人：照着做，能在 10 分钟内跑出第一组 ERM / TT-GREB
对比结果。方法原理和实验结论见 `README.md` 与 `IMPLEMENTATION_SUMMARY.md`。

---

## 0. 这个项目是什么

在 Cora 等图数据集上训练一个 GCN（ERM 基线），冻结它，然后对 8 张人为注入分布
偏移的测试图执行 **测试时图重生（TT-GREB）**：不改模型参数，只按"类原型 + 邻域
一致性"重加权节点特征，再让冻结的模型重新预测。

输出是每张测试图上的 `ERM` 精度、`TT-GREB` 精度和两者的差 `delta`。

---

## 1. 五分钟跑通（只跑 Cora）

### 1.1 环境

需要 Python 3.10 / 3.11。以 conda 为例：

```bash
conda create -n ttgreb python=3.11 -y
conda activate ttgreb
pip install -r requirements.txt
```

`requirements.txt` 已包含 torch / torch_geometric / numpy / scipy / scikit-learn /
pandas / ogb。CPU 即可运行；有 CUDA 会自动用 GPU。

验证安装：

```bash
python -c "import torch, torch_geometric; print(torch.__version__, torch_geometric.__version__)"
```

### 1.2 数据

数据默认放在**仓库根目录的 `data/`**，也可以用一个环境变量指向任意位置：

```bash
# Linux / macOS
export TT_GREB_DATA=/path/to/data
# Windows PowerShell
$env:TT_GREB_DATA = "D:\path\to\data"
```

Cora 最省事：什么都不放，PyG 会自动下载到 `data/Planetoid/cora/`（需要联网）。
离线环境请按第 2 节的目录放好 raw 文件。

### 1.3 跑一条命令

```bash
cd TT_GREB_Reproduce

# 先做接口自测（不需要数据集，几秒钟）
python tests/test_backbones.py

# 跑 Cora + GCN + seed 0
python run_cora_gcn.py --dataset cora --seed 0 --output results_v2/cora_seed0
```

正常情况下最后会打印（seed 0 的实测值）：

```text
[cora] ERM 0.7932 ± 0.0611
[cora] TTA 0.8224 ± 0.0322
[cora] delta mean=+0.0292
```

单种子数值会随种子波动：ERM 大致在 0.77–0.82、delta 大致在 +0.02–+0.05。
判断方法是否有效要看 10 个种子的均值（0.8010 → 0.8370，+0.0360），不要拿单次
结果下结论。CPU 上单种子 Cora 约 25 秒；8 张测试图的适配合计不到 1 秒。

---

## 2. 数据准备

`data/` 不入库，按下面结构放置。**目录名必须一致**，路径在
`tt_greb_reproduce/paths.py` 里统一配置。

| 数据集 | 目录 | 必需文件 | 来源 |
|---|---|---|---|
| Cora | `data/Planetoid/cora/raw/` | `ind.cora.{x,y,tx,ty,allx,ally,graph,test.index}` | PyG 自动下载，或 Planetoid 原始包 |
| Amazon-Photo | `data/Amazon/Photo/raw/` | `amazon_electronics_photo.npz` | PyG 自动下载 |
| Twitch-DE | `data/twitch/DE/` | `musae_DE_edges.csv`、`musae_DE_target.csv`、`musae_DE_features.json` | [MultiSAGE / Twitch 数据](https://github.com/benedekrozemberczki/datasets) 的 `twitch/DE` |
| Elliptic | `data/elliptic/` | `1.pkl` … `48.pkl`、`elliptic_txs_classes.csv`、`elliptic_txs_edgelist.csv` | Kaggle *Elliptic Data Set*（解压后直接放进来，不要多套一层目录） |
| OGB-Arxiv | `data/ogbn_arxiv/raw/` | `node-feat.csv.gz`、`node-label.csv.gz`、`edge.csv.gz`（`node_year.csv.gz` 可选） | [ogbn-arxiv](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) 原始 csv.gz |

注意事项：

- 目录名大小写敏感（`Planetoid`、`Amazon`、`ogbn_arxiv`）。
- Elliptic 的 pkl 与两个 csv 必须来自同一份数据；加载器会按标签序列自动对齐，
  对不齐会直接报错，不会静默产出错数据。
- Twitch 的 `features.json` / `edges.csv` 用 `new_id` 编号，`target.csv` 的行序不是
  `new_id` 顺序，代码里已经做了重排，不需要手工处理。

---

## 3. 常用命令

全部在 `TT_GREB_Reproduce/` 目录下执行。

### 3.1 单数据集

```bash
# Cora
python run_cora_gcn.py --dataset cora --seed 0 --output results_v2/cora_seed0

# Amazon-Photo
python run_cora_gcn.py --dataset amazon --seed 0 --output results_v2/amazon_seed0

# Twitch-DE（ROC-AUC）
python run_cora_gcn.py --dataset twitch --seed 0 --output results_v2/twitch_seed0

# Elliptic（ROC-AUC；连续特征，重生会按设计跳过）
python run_cora_gcn.py --dataset elliptic --seed 0 --output results_v2/elliptic_seed0

# OGB-Arxiv（节点多，CPU 很慢，建议先用 --epochs 50 试）
python run_cora_gcn.py --dataset ogb --seed 0 --epochs 50 --output results_v2/ogb_seed0
```

### 3.2 多种子

```powershell
# Windows PowerShell
for ($s=0; $s -lt 10; $s++) {
  python run_cora_gcn.py --dataset cora --seed $s --output results_v2/cora_seed$s
}
```

```bash
# Linux / macOS
for s in $(seq 0 9); do
  python run_cora_gcn.py --dataset cora --seed $s --output results_v2/cora_seed$s
done
```

### 3.3 换骨干

```bash
# 本仓库自带的 GraphSAGE
python run_cora_gcn.py --dataset cora --seed 0 --backbone sage --output results_v2/cora_sage_seed0

# 第三部分的 GAT / GPR-GNN（需要 models/ 包在 --model-path 下）
python run_cora_gcn.py --dataset cora --seed 0 --backbone gat \
  --model-path .. --model-kwargs '{"heads": 4}' \
  --output results_v2/cora_gat_seed0
```

可用的 `--backbone`：`gcn` / `sage` / `gat` / `gprgnn`。外部模型的接口约定见
`docs/PART3_INTERFACE.md`。

### 3.4 只评估、不重新训练

```bash
python scripts/eval_adapt.py --seed 0 --graphs 0,3,7
# 指定 checkpoint 或外部模型
python scripts/eval_adapt.py --seed 0 --graphs 7 --ckpt results_v2/cora_seed0/checkpoint.pt
```

### 3.5 汇总成表

```bash
python scripts/make_report.py --root results_v2
```

会在 `results_v2/` 下生成：

- `REPORT.md`：每个数据集 / 骨干的均值、标准差、逐种子表、逐测试图 ERM/TT/delta；
- `results.csv`：一行一个种子，方便直接画图。

---

## 4. 输出文件怎么读

每个 `--output` 目录包含：

| 文件 | 内容 |
|---|---|
| `checkpoint.pt` | 验证集上选出的最佳骨干权重（冻结后用于测试时适配） |
| `summary.json` | 主结果：`erm` / `tta` / `delta` 的 mean、std、逐图数值，以及 `details` 里每张测试图的明细 |
| `config.json` | 本次运行的全部参数（数据集、种子、偏移预设、tau、训练超参、backbone 类名） |

`summary.json` 关键字段：

```json
{
  "dataset": "cora",
  "backbone": "gcn",
  "metric": "acc",
  "seed": 0,
  "clean_test": 0.8700,
  "erm": {"mean": 0.7932, "std": 0.0611, "values": [ ... 8 个数 ... ]},
  "tta": {"mean": 0.8224, "std": 0.0322, "values": [ ... ]},
  "delta": {"mean": 0.0292, "std": 0.0295, "values": [ ... ]},
  "details": [{"graph": 0, "erm": 0.854, "tta": 0.854, "delta": 0.0}, ...]
}
```

看结果时注意：`delta > 0` 才是重生有效，`tta.std < erm.std` 说明重生同时降低了
跨环境的方差。

---

## 5. 预期结果（Cora，10 个种子）

| 骨干 | ERM | TT-GREB | delta | 正增益种子 |
|---|---:|---:|---:|---:|
| GCN | 0.8010 | 0.8370 | +0.0360 | 10/10 |
| GraphSAGE | 0.7907 | 0.8336 | +0.0429 | 10/10 |

其他数据集（单种子）：Amazon-Photo 基本持平（偏移过弱），Twitch-DE 基本持平
（AUC 0.7104 → 0.7071），Elliptic 按设计跳过（连续特征，AUC 不变）。

---

## 6. 常见问题

**Q：`ModuleNotFoundError: No module named 'torch'`**
用错 Python 了。默认的 `python` 可能没装依赖，先 `conda activate ttgreb`，再用
`python -c "import torch"` 确认。

**Q：跑到一半提示找不到数据文件**
检查 `TT_GREB_DATA` 是否指对，以及目录名大小写。可以先跑
`python -c "from tt_greb_reproduce.paths import DATA_ROOT; print(DATA_ROOT)"`。

**Q：Elliptic 的 delta 是 0**
这是预期行为。Elliptic 是有符号连续特征（约 -13 ~ 289），"类原型频率 / 邻域出现率"
没有语义，强行门控会把 AUC 从 0.97 打到 0.91。`adapt_test_graph` 检测到非 0/1 特征
会直接返回原图。

**Q：Twitch 的 AUC 只有 0.7 左右**
正常。这是 9498 节点、3170 维稀疏多热特征、二分类的真实水平，不是 bug。

**Q：OGB-Arxiv 太慢 / 内存爆**
169k 节点、1.16M 边，CPU 上训练很慢。先用 `--epochs 20 --patience 10` 验证流程，
正式跑再加大；内存不足时关闭其他进程，不要在 4GB 以下机器上跑。

**Q：同一条命令两次结果不一样**
没有固定种子。加 `--seed`；不同种子之间 delta 会有 1–2 个点波动，结论要看 10 个
种子的均值。

**Q：results_v2 里有的目录没进 git**
仓库只保留 `REPORT.md` 和 `results.csv`，checkpoint 等大文件被 `.gitignore` 排除，
随时可以用命令重新生成。

**Q：想换数据集路径但不想改代码**
设置环境变量 `TT_GREB_DATA` 即可，`paths.py` 优先读它。

---

## 7. 目录结构

```text
TT_GREB_Reproduce/
├── run_cora_gcn.py              # 主入口：训练 ERM + 8 张偏移图 ERM/TT 对比
├── requirements.txt
├── README.md                    # 方法与结果
├── IMPLEMENTATION_SUMMARY.md    # 实现细节、消融、已知问题
├── USAGE.md                     # 本文件
├── docs/
│   └── PART3_INTERFACE.md       # 如何接入 GAT / GPR-GNN
├── tests/
│   └── test_backbones.py        # 骨干接口自测
├── tt_greb_reproduce/
│   ├── tt_greb.py               # TT-GREB 主函数（原型 / 环境 / 重组 / 迭代）
│   ├── backbones.py             # 模型注册与统一前向接口
│   ├── models.py                # 自带 GCN / GraphSAGE
│   ├── data.py                  # 数据集加载与 50/25/25 划分
│   ├── shift.py                 # 偏移测试图生成
│   └── paths.py                 # 路径与数据集元信息
├── scripts/
│   ├── make_report.py           # 汇总 results_v2 -> REPORT.md / results.csv
│   ├── eval_adapt.py            # 用已有 checkpoint 快速评估
│   └── diagnostics/             # 诊断与改进实验脚本
└── results_v2/                  # 运行输出（只提交 REPORT.md / results.csv）
```

---

## 8. 复现清单

- [ ] 环境装好，`python tests/test_backbones.py` 全部通过
- [ ] Cora 单种子跑通，delta > 0
- [ ] Cora 10 个种子跑完，`make_report.py` 生成表格
- [ ] 换 `--backbone sage` 再跑一遍，确认增益不是 GCN 特有
- [ ] 按需准备其他数据集，确认指标与预期一致
- [ ] 第三部分模型接入后，先跑 `tests/test_backbones.py`，再跑真实模型
