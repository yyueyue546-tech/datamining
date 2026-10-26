"""Render the completed local verification evidence as a Feishu-friendly Markdown note."""
from pathlib import Path
import json

root = Path(__file__).resolve().parent
e = json.loads((root / "results/verification.json").read_text(encoding="utf-8"))
assert len(e["runs"]) == 4 and len(e["repeat_checks"]) == 2
rows = []
for run in e["runs"]:
    s = run["summary"]
    epochs = 50 if "200_20" not in run["name"] else 200
    steps = 5 if epochs == 50 else 20
    delta = (s["tt_greb_mean_accuracy"] - s["erm_mean_accuracy"]) * 100
    rows.append(f"| {run['name']} | {epochs}/{steps} | {s['validation_accuracy']:.2%} | {s['erm_mean_accuracy']:.2%} | {s['tt_greb_mean_accuracy']:.2%} | {delta:+.2f} | {run['seconds']:.1f} |")
text = '''# TT-GREB 标准 Cora 本地复现记录（Windows）

实验日期：2026 年 10 月 26 日（北京时间）

## 1. 本次完成情况

本次完成标准 Cora 单图节点分类的最小实验链路：加载本地数据 → 训练两层 GCN → 按验证集 Accuracy 选择 checkpoint → 冻结 GCN → ERM 测试 → 测试时图重加权 → Accuracy 对比。

按启动说明先执行 epochs=50、adapt-steps=5，随后执行 epochs=200、adapt-steps=20。两种配置均以 seed=0 独立运行两次，检查汇总结果和 checkpoint 张量是否完全一致。

本记录属于工程链路验证。当前数据是标准单图 Cora，适配模块是仓库提供的替代目标实现，不能将这些分数称为最终论文结果复现。适配没有提升 Accuracy 时也如实记录，不通过测试标签调参。

## 2. 代码来源与目录

仓库：https://github.com/yyueyue546-tech/datamining

下载时提交：`ca798b54501b9aaafc1f127a0578bcb8c8642c7f`

工作区：`D:\\VSCodePrograms\\Test-Time_Graph_Rebirth`

代码目录：`datamining/TT_GREB_Reproduce`。仓库根目录也有同名脚本，本次统一使用上述子目录版本。

本次有本地修复；只重新克隆原始提交不能得到修复后的代码。交接时应同时携带 `local_fixes.patch`、本报告、环境快照，或提交本地修改。对原始提交可在仓库根目录执行 `git apply TT_GREB_Reproduce/local_fixes.patch` 应用两处运行代码修复；验证与报告脚本需另外携带。

## 3. 实测环境

| 项目 | 本次配置 |
|---|---|
| 系统 | Windows |
| Python | PYTHON_VERSION |
| Python 路径 | C:\\Anaconda\\envs\\ttgreb\\python.exe |
| PyTorch | TORCH_VERSION |
| PyTorch Geometric | PYG_VERSION |
| 计算设备 | CPU |
| NumPy | 2.4.6 |
| SciPy | 1.17.1 |

安装 PyG 2.6.1 和 SciPy 后，导入检查及 pip check 通过，实验脚本 --help 正常。运行中出现 torch.jit.script 弃用提示，但没有阻止完成本次实验。

原始 requirements.txt 固定 torch==2.6.0；本次保留原有 2.14.1+cpu，没有执行会降级 PyTorch 的整份依赖安装。该差异是本地环境适配，不代表验证了论文原环境。完整安装快照保存在 requirements.windows-validated.txt；该文件是实测记录，新机器仍需选择对应系统可安装的 wheel。

## 4. 数据与评估协议

此前检查过的外层 Cora/processed/data.pt 在本次运行时已不存在，本次没有删除或移动它。实际使用仓库自带的 data/Cora/raw 原始文件，通过 PyG 的 read_planetoid_data 在本地生成缓存，不重新下载数据、不自行生成论文划分。

实际输入：`D:\\VSCodePrograms\\Test-Time_Graph_Rebirth\\datamining\\data\\Cora\\processed\\data.pt`

| 项目 | 数量 |
|---|---:|
| 图样本 | 1 |
| 节点 | 2708 |
| 节点特征维度 | 1433 |
| 类别 | 7 |
| edge_index 中的有向边条目 | EDGE_ENTRIES |
| 训练节点 | 140 |
| 验证节点 | 500 |
| 测试节点 | 1000 |
| 不属于上述三个掩码的节点 | 1068 |

三个掩码没有重叠。训练只使用 train_mask 的标签；val_mask 用于选择最佳 checkpoint；最终 Accuracy 只对 test_mask 计算。完整图的特征和连接参与前向传播，因此这是传导式节点分类设置。未选中节点仍在图中，但其标签不进入训练损失或评分。

输入缓存 SHA-256：`CACHE_HASH`

本次缓存重新生成后结构符合预期，但由于原缓存已经不在，不能声明与原缓存文件逐字节相同。

## 5. 启动说明检查与实际修复

1. 启动说明的标准单图最小流程可以执行，但不能解释成完整论文复现。仓库自身也明确标注了算法和数据缺口。
2. 原 load_masked_cora 没有直接读取传入缓存，而是调用 Planetoid 重新推导目录。在本次目录状态下触发下载并失败。修复为直接加载指定的本地 data.pt，支持 PyG 的二元组/三元组单图缓存，并保留掩码检查；不会隐式切换到其他缓存。
3. 原测试适配循环没有逐个取对应 mask，而是沿用前一个推导式留下的变量。单图路径数值未受此问题影响，但多图路径可能错误。修复为 zip(test_graphs, test_masks)；没有修改适配目标函数。
4. 原启动说明中“1/1/8 文件通过检查即可用于论文记录”的条件不充分。图数量匹配只是格式检查，还需数据来源、构图方式、预处理、公式、优化策略和超参数一致。
5. 原缓存缺失后，从仓库 raw 文件重新处理时发现缺少 SciPy；已补装。此项为原始稀疏矩阵读取依赖。

没有为提高测试分数而修改划分、替换目标函数、调整 seed 或筛选结果。训练和适配使用仓库原默认超参数，只按说明改变 epochs 和 adapt-steps。

## 6. 实验配置与执行方法

GCN：两层 GCNConv，隐藏维度 64，dropout=0.5，训练 Adam，学习率 0.01，weight decay=0.0005。验证 Accuracy 严格提高时保存最佳状态，再恢复该状态并冻结所有 GCN 参数。

适配：节点和边的 prototype/environment MLP 重加权，Adam 学习率 0.01，alpha=0.1、beta=0.1，两个比例目标均为 0.5。当前实现保持 edge_index 不变，修改节点特征和边权；损失包括余弦相似度项、环境差异项和比例正则项。测试时真实标签仅供最终评分，没有进入适配损失。

以下命令从外层工作区的 PowerShell 执行：

```powershell
cd "D:\\VSCodePrograms\\Test-Time_Graph_Rebirth"
$pythonExe = "C:\\Anaconda\\envs\\ttgreb\\python.exe"

# 本次已安装；新环境按实际需要安装
& $pythonExe -m pip install "torch_geometric==2.6.1" scipy
& $pythonExe -m pip check

# 自动处理缺失缓存、执行 4 次实验，并校验输出和可重复性
& $pythonExe ".\\datamining\\TT_GREB_Reproduce\\verify_local_reproduction.py"

# 根据实际证据生成本报告
& $pythonExe ".\\datamining\\TT_GREB_Reproduce\\verify_adaptation_invariants.py"
& $pythonExe ".\\datamining\\TT_GREB_Reproduce\\verify_saved_checkpoints.py"
& $pythonExe ".\\datamining\\TT_GREB_Reproduce\\make_reproduction_report.py"
```

仅运行一轮 50/5：

```powershell
& $pythonExe ".\\datamining\\TT_GREB_Reproduce\\run_cora_gcn.py" --processed-cora ".\\datamining\\data\\Cora\\processed\\data.pt" --epochs 50 --adapt-steps 5 --seed 0 --device cpu --output ".\\datamining\\TT_GREB_Reproduce\\results\\cora_masked_gcn_seed0"
```

运行 200/20 时将上述两个参数改为 200 和 20，并使用独立输出目录 cora_masked_gcn_seed0_200_20。再次运行相同目录会覆盖其中的同名输出；保留历史实验时使用新目录。

## 7. 实际实验结果

以下均为 Accuracy，变化单位为百分点。耗时为包含 Python 子进程启动、训练和适配的总耗时，不代表纯推理性能。

| 运行目录 | epochs / adapt-steps | 最佳验证 Accuracy | ERM 测试 Accuracy | 当前适配实现测试 Accuracy | 变化（百分点） | 耗时（秒） |
|---|---|---|---|---|---|---:|
RESULT_ROWS

## 8. 结果验收

- 四次运行均正常退出并生成 checkpoint.pt、config.json、summary.json、test_results.csv，另外保留 run.log。
- 四份 checkpoint 重新加载后，重新计算的验证集 Accuracy 和 ERM 测试 Accuracy 均与汇总记录完全一致；证据保存在 results/checkpoint_verification.json。
- evaluation_protocol 均为 node_masks；paper_split_verified 均为 false。
- 两组配置各自重复运行的 summary.json 内容完全一致，checkpoint 的每一个参数张量完全一致。这证明本机当前环境中的 seed=0 可重复性，不代表跨平台或多个种子的统计稳定性。
- 在独立的 6 节点合成图检查中，仅改变真实标签并重置适配随机种子，适配后的节点特征和边权完全一致；GCN 所有参数在适配前后完全一致且保持冻结。检查脚本为 verify_adaptation_invariants.py，证据为 results/adaptation_invariants.json。这是针对实现行为的检查，合成图结果没有混入 Cora Accuracy。
- summary 中 train_graphs=1、valid_graphs=1、test_graphs=1 是代码对各用途的计数；在 node_masks 协议下，它们引用同一张图，不能加总为三张独立图。
- config.json 中 smoke_only=false 仅表示没有使用 --smoke-standard-cora；不能据此认定完成论文复现。本次仍然是标准 Cora 的链路验证。
- 适配步数和最终损失可在 summary 的 adaptation 字段检查；负的 final_loss 在当前包含负环境差异项的目标中并不自动意味着错误。

## 9. 输出与交接材料

结果均位于 `datamining/TT_GREB_Reproduce/results/`：

| 文件/目录 | 作用 |
|---|---|
| cora_masked_gcn_seed0 | 50/5 首次实验 |
| cora_masked_gcn_seed0_repeat | 50/5 重复实验 |
| cora_masked_gcn_seed0_200_20 | 200/20 首次实验 |
| cora_masked_gcn_seed0_200_20_repeat | 200/20 重复实验 |
| verification.json | 数据指纹、环境、完整命令、结果、耗时和重复性检查 |

代码目录另有 verify_local_reproduction.py、make_reproduction_report.py、requirements.windows-validated.txt 和 local_fixes.patch，分别用于重新执行、生成文档、环境记录及修复交接。

## 10. 结论与正式论文复现的后续条件

当前已经跑通并重复验证标准 Cora + GCN + seed=0 的训练与测试时适配链路。结果能够支持本地程序可运行的结论；是否提升以表中实际数值为准，不能预设 TT-GREB 必然优于 ERM。

仍未完成论文正式结果复现：仓库没有可核实的作者 Cora 多图划分；当前适配目标是仓库明确注明的替代实现；没有逐项核对最终论文公式、数据处理与超参数；没有完成其他数据集、backbone、baseline 和 10 个种子实验。

下一阶段应先获取并核验论文/作者的数据划分与算法实现，再开展与论文设置一致的实验。不能将标准单图按节点任意切成十份，便视为获得论文的 1/1/8 图划分。本报告对论文划分的描述来自仓库启动说明，未独立完成原论文验证。
'''
for old, new in {
    "PYTHON_VERSION": e["python"], "TORCH_VERSION": e["torch"],
    "PYG_VERSION": e["pyg"], "EDGE_ENTRIES": str(e["edge_entries"]),
    "CACHE_HASH": e["cache_sha256"], "RESULT_ROWS": "\n".join(rows),
}.items():
    text = text.replace(old, new)
(root / "REPRODUCTION_FEISHU.md").write_text(text, encoding="utf-8")
print(root / "REPRODUCTION_FEISHU.md")
