# TT-GREB 当前实现说明

## 当前目标

本阶段搭建 `Cora + GCN + seed=0` 的最小实验流程，用于检查数据读取、GCN 训练、验证集选模、冻结模型、ERM 推理和 TT-GREB 测试时图变换能否连通。它是工程 smoke test，不是论文结果复现。

## 文件说明

| 文件 | 用途 |
|---|---|
| `run_cora_gcn.py` | 训练 GCN、按验证集 Accuracy 保存最佳 checkpoint，并比较 8 个测试图上的 ERM 与 TT-GREB Accuracy。 |
| `tt_greb_reproduce/data.py` | 读取并校验图划分；提供标准 PyG Cora 的 smoke split。 |
| `tt_greb_reproduce/models.py` | 两层 GCN backbone。 |
| `tt_greb_reproduce/tt_greb.py` | 节点/边重加权模块及测试时优化流程；当前优化目标是待论文核实的可运行替代实现。 |
| `prepare_standard_cora_smoke.py` | 生成标准 PyG Cora 的确定性 smoke split。 |
| `inspect_split.py` | 检查划分数量、节点数、边数、特征维度和类别数。 |
| `requirements.txt` | 实验所需 Python 依赖版本。 |
| `README.md` | 环境安装、运行命令和数据格式说明。 |

## Smoke test

建议在 Python 3.10/3.11、PyTorch 2.6、PyG 2.6.1 环境运行：

```powershell
python TT_GREB_Reproduce/run_cora_gcn.py `
  --smoke-standard-cora `
  --output TT_GREB_Reproduce/results/cora_gcn_seed0_smoke
```

输出目录包括 GCN checkpoint、每张测试图的 ERM/TT-GREB Accuracy CSV、运行配置和汇总 JSON。标准 PyG Cora 的 smoke split 是为了验证程序链路，不符合论文 Cora 划分，结果不可用于论文对比。

## 当前缺口

1. GitHub 仓库目前的 Cora 数据是标准 Planetoid 数据，不是论文使用的 `1 train / 1 validation / 8 test` 预处理划分。
2. TT-GREB 模块的损失函数、优化策略和超参数尚未与最终 ICDM 论文或作者实现逐项核对。
3. 目前只覆盖 Cora、GCN 和一个随机种子；尚未复现其余数据集、GNN backbone、baseline 或 10-seed 结果。

因此，现阶段可以验证代码结构与运行链路；完成论文版数据划分和算法核对后，才能称为 Cora + GCN 的论文实验复现。
