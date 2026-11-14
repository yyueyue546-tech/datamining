# TT-GREB 复现：测试时图重生

用冻结的 GNN + 测试时改图（不改模型参数），在分布偏移的测试图上提升节点分类。

```text
clean graph
  -> 按 50/25/25 划分带标签节点（train / val / test mask）
  -> 在 train mask 上训练 GCN，按 val 指标保存最佳 checkpoint
  -> 冻结 GCN
  -> 由 shift.py 生成 8 张强度递增的偏移测试图（0.2x -> 1.8x）
  -> 每张图分别评估 ERM 与 TT-GREB
  -> 汇总 mean / std 写入 summary.json
```

## 环境

```powershell
conda activate ttgreb     # torch 2.14.1+cpu, torch_geometric 2.8.1, python 3.11
```

数据集放在 `D:\VSCodePrograms\datamining\data`，路径集中在 `tt_greb_reproduce/paths.py`。

## 运行

```powershell
cd D:\VSCodePrograms\datamining\TT_GREB_Reproduce

# 单数据集、单种子
python run_cora_gcn.py --dataset cora --seed 0 --output results_v2/cora_seed0

# 全部 10 个种子（Cora）
for ($s=0; $s -lt 10; $s++) {
  python run_cora_gcn.py --dataset cora --seed $s --output results_v2/cora_seed$s
}

# 汇总成表格
python scripts/make_report.py --root results_v2
```

可选参数：`--adapt-rounds`（迭代轮数，默认 3）、`--tau`（支撑度阈值，默认 0.05）、
`--epochs`、`--hidden-dim`、`--lr`、`--n-test`。`--dataset` 支持
`cora / amazon / twitch / ogb / elliptic`。

## 骨干模型接入（第三部分）

`--backbone` 支持 `gcn / sage / gat / gprgnn`。前两个是本仓库实现，后两个从第三部分的
`models/` 包加载：只要模型满足

```python
logits = model(x, edge_index, edge_weight=None)
logits, embeddings = model.forward_with_embeddings(x, edge_index, edge_weight=None)  # 可选
```

就能直接进 TT-GREB，主函数不依赖 `model.conv2` 这类具体属性。

```powershell
# Cora + 第三部分的 GAT
python run_cora_gcn.py --dataset cora --seed 0 --backbone gat `
  --model-path D:\VSCodePrograms\datamining `
  --model-kwargs '{"heads": 4}' --output results_v2/cora_gat_seed0

# 接口自测（不依赖第三部分代码）
python tests/test_backbones.py
```

细节与常见坑见 `docs/PART3_INTERFACE.md`。

## 数据与偏移协议

| 数据集 | 指标 | 说明 |
|---|---|---|
| Cora | Accuracy | 2708 节点 / 1433 维二值词袋 |
| Amazon-Photo | Accuracy | 7650 节点 / 745 维二值特征 |
| Twitch-DE | ROC-AUC | 9498 节点 / 3170 维多热特征 |
| Elliptic | ROC-AUC | 195,889 节点 / 165 维连续特征（仅有标签节点参与划分与评估） |

`shift.py` 对每张测试图注入两类偏移，强度按 `scale = 0.2, 0.43, ..., 1.8` 线性递增：

- 特征扰动：二值特征按 `sigma` 概率把 1 翻成 0、按 `sigma/20` 概率把 0 翻成 1；连续特征用高斯噪声。
- 结构扰动：按 `drop` 比例对称删边。

## 方法实现

`tt_greb_reproduce/tt_greb.py` 保留了论文的结构与重组公式：

```text
G_Pro = (X ⊙ W^Pro_node, A ⊙ W^Pro_edge)     原型提取器
G_Env = (X ⊙ W^Env_node, A ⊙ W^Env_edge)     环境细化器
W^Comp = (1 - W^Pro) ⊙ W^Env + W^Pro          论文式 (6)
G'_te = (X ⊙ W^Comp_node, A ⊙ W^Comp_edge) -> 冻结的 GNN
```

论文式 (7)(8) 的端到端对比学习目标在"冻结 GNN"下存在平凡解（GCN 对输入正齐次，
`W^Env -> 0` 就能无限拉大 `||Z_te - Z_Env||`），实测会让精度崩塌（Cora test7:
0.6765 -> 0.4682）。因此这里把两个子模块换成有闭式解的稳健估计量：

- `W^Pro_node[i,d] = 1[ P[c_i, d] >= tau ]`，`P` 是高置信伪标签估计的类原型；
- `W^Env_node[i,d] = 1[ mean_{j∈N(i)} X[j,d] >= tau ]`，邻域一致（环境内稳定）成分；
- 伪标签每轮用当前重生图重新预测，即论文的迭代优化。

含义：原型维直接保留；非原型维只有在邻域中稳定出现才保留，从而抑制随环境随机
出现的噪声特征。细节与实测结论见 `IMPLEMENTATION_SUMMARY.md`。

## 结果

Cora，10 个随机种子 × 8 张强度递增的偏移测试图：

| backbone | ERM | TT-GREB | delta | 正增益种子 |
|---|---:|---:|---:|---:|
| GCN | 0.8010 | 0.8370 | **+0.0360** | 10/10 |
| GraphSAGE | 0.7907 | 0.8336 | **+0.0429** | 10/10 |

- 单种子增益：GCN +0.020 ~ +0.050，SAGE +0.012 ~ +0.064；
- 偏移越强增益越大：最轻的 T0 约 ±0.01，最重的 T7 达 +0.07 ~ +0.14；
- TT-GREB 跨 8 张测试图的标准差从 0.061 降到约 0.027，与论文"std 最小"一致。

其他数据集（单种子）：

| 数据集 | 指标 | ERM | TT-GREB | 说明 |
|---|---|---:|---:|---|
| Amazon-Photo | Acc | 0.9383 | 0.9386 | 偏移过弱，ERM 几乎不掉，没有提升空间 |
| Twitch-DE | AUC | 0.7104 | 0.7071 | 基本持平；修复了特征/标签错位 |
| Elliptic | AUC | 0.9704 | 0.9704 | 有符号连续特征，门控语义不适用，按设计跳过 |

完整逐种子 / 逐测试图数据见 `results_v2/REPORT.md`（由 `scripts/make_report.py` 生成）。

配对检验：GCN 逐种子 t 检验 p = 1.2e-6，SAGE p = 3.4e-5，逐测试图 80 对
p = 1.6e-11（64/80 为正）。

精度再往上还有多少：门控这一族已经饱和。把伪标签换成真标签（仅作上界）只能再
多拿约 2 个点，说明瓶颈是伪标签质量；软门控、top-k、补边、邻域 lift、置信度
选择性重生、MC-Dropout、logit 平滑都试过，均未超过当前基线。10 模型 logit 集成
可以把绝对精度抬到 0.8497（干净图 0.8922），但 delta 从 +2.9 缩到 +2.3。
完整表格见 `IMPLEMENTATION_SUMMARY.md`。

## 文件

| 文件 | 用途 |
|---|---|
| `run_cora_gcn.py` | 训练 + 8 张偏移图上的 ERM / TT-GREB 对比，写 summary.json |
| `tt_greb_reproduce/data.py` | 数据集加载与 50/25/25 节点划分 |
| `tt_greb_reproduce/shift.py` | 偏移测试图生成 |
| `tt_greb_reproduce/models.py` | 两层 GCN backbone |
| `tt_greb_reproduce/tt_greb.py` | 原型提取器 / 环境细化器 / 重组 / 迭代优化 |
| `tt_greb_reproduce/paths.py` | 数据集路径与元信息 |
| `scripts/make_report.py` | 汇总 results_v2 生成 REPORT.md / results.csv |
| `scripts/eval_adapt.py` | 用已有 checkpoint 快速评估指定测试图 |
| `scripts/diagnostics/` | 各类诊断脚本（偏移来源、门控方案对比等） |
