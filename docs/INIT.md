# TT-GREB 复现启动说明

> Windows 本次实测记录见 `REPRODUCTION_FEISHU.md`。本机保留 Python 3.11.17 / PyTorch 2.14.1+cpu，安装 PyG 2.6.1 和 SciPy；没有执行下文会降级 PyTorch 的通用 requirements 安装命令。实测环境快照见 `requirements.windows-validated.txt`。本机数据路径为 `datamining/data/Cora/processed/data.pt`（相对于外层工作区）；运行代码已修复为直接读取指定缓存。

这份文件用于把当前代码交给另一台电脑、服务器或云环境运行。GitHub 只保存代码，不会自动执行实验；实验必须在安装了 Python、PyTorch 和 PyTorch Geometric 的运行环境中执行。

## 当前阶段是什么

当前代码先验证最小链路：

```text
标准 Cora 单图
  -> GCN 训练
  -> 验证集选择 checkpoint
  -> 固定 GCN 参数
  -> ERM 测试
  -> TT-GREB 测试时图变换
  -> 输出 Accuracy 对比
```

你提供的 `Cora/processed/data.pt` 是标准 Planetoid Cora：一张图、140 个训练节点、500 个验证节点、1000 个测试节点。它不是论文中的 `1 train / 1 validation / 8 test graphs`，所以当前结果只能验证程序链路，不能作为论文表格结果。

## 在另一台机器上开始

### 1. 获取代码

```bash
git clone https://github.com/yyueyue546-tech/datamining.git
cd datamining
```

也可以在 GitHub 网页下载 ZIP 后解压，但使用 `git clone` 更方便同步后续修改。

### 2. 创建 Python 环境

推荐 Python 3.10 或 3.11。Windows PowerShell：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. 安装依赖

```bash
python -m pip install --upgrade pip
pip install -r TT_GREB_Reproduce/requirements.txt
```

如果服务器有 NVIDIA GPU，应按 CUDA 版本从 PyTorch 官方页面安装对应的 PyTorch 2.6 wheel，再安装 `torch_geometric`。

## 先跑标准 Cora 节点 mask 测试

如果远程机器没有 Cora 数据，先让 PyG 下载并缓存标准 Cora：

```bash
python TT_GREB_Reproduce/prepare_standard_cora_smoke.py \
  --root TT_GREB_Reproduce/data \
  --output TT_GREB_Reproduce/data/cora_standard_smoke.pt
```

然后运行单图节点 mask 实验：

```bash
python TT_GREB_Reproduce/run_cora_gcn.py \
  --processed-cora TT_GREB_Reproduce/data/Cora/processed/data.pt \
  --epochs 50 \
  --adapt-steps 5 \
  --output TT_GREB_Reproduce/results/cora_masked_gcn_seed0
```

Windows PowerShell 使用反引号续行，或把命令写成一行：

```powershell
python TT_GREB_Reproduce/run_cora_gcn.py --processed-cora "TT_GREB_Reproduce/data/Cora/processed/data.pt" --epochs 50 --adapt-steps 5 --output "TT_GREB_Reproduce/results/cora_masked_gcn_seed0"
```

结果目录应包含：

```text
checkpoint.pt       验证集选择出的 GCN 参数
config.json         设备、版本、seed 和超参数
summary.json        验证集、ERM、TT-GREB 汇总
test_results.csv    测试结果
```

先用 `epochs=50`、`adapt-steps=5` 检查是否能完成；流程稳定后再增加到 `epochs=200`、`adapt-steps=20`。

## 使用独立图划分

论文版接口要求一个 `.pt` 文件：

```python
{
    "train": Data,
    "valid": Data,
    "test": [Data, Data, Data, Data, Data, Data, Data, Data],
}
```

检查划分：

```bash
python TT_GREB_Reproduce/inspect_split.py path/to/cora_1_1_8.pt
```

运行独立图实验：

```bash
python TT_GREB_Reproduce/run_cora_gcn.py \
  --split path/to/cora_1_1_8.pt \
  --output TT_GREB_Reproduce/results/cora_gcn_seed0
```

只有当该文件确实来自论文/作者的 Cora 预处理，并且检查结果为 1/1/8 时，才可以把输出用于论文复现记录。

## 如何判断第一步成功

标准 mask 测试成功的最低条件：

1. 能加载 `data.pt`，没有 PyTorch 2.6 缓存错误；
2. 训练过程完成并写出 `checkpoint.pt`；
3. `summary.json` 中 `evaluation_protocol` 为 `node_masks`；
4. `test_results.csv` 有 ERM 和 TT-GREB 分数；
5. 能重复运行同一个 `seed=0`，结果稳定到可接受范围。

## 当前不能宣称的内容

当前代码还不能宣称已经完整复现论文，原因是：

- GitHub 仓库目前没有论文版 Cora 的 1/1/8 多图数据；
- 当前 TT-GREB 优化目标是可运行的实现骨架，尚未与最终 ICDM 版本逐项核对；
- 还没有完成 Amazon-Photo、Twitch-E、Elliptic、OGB-Arxiv；
- 还没有完成 SAGE、GAT、GPR-GNN、baseline 和 10 个随机种子。

完成标准 Cora 节点 mask 测试后，下一步是获得作者的 1/1/8 Cora 划分，核对论文公式和超参数，再运行 Cora + GCN + seed=0 的正式多图实验。
