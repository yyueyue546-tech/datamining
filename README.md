# EERM / TT-GREB: Serving GNN Generalization Under Distribution Shifts

本项目为论文《Test-Time Graph Rebirth: Serving GNN Generalization Under Distribution Shifts》的官方代码复现与数据统一划分版本。

本项目针对图神经网络（GNN）在不同分布偏移（Distribution Shifts）下的泛化能力进行测试，涵盖了三种核心场景：

1. **人工特征偏移 (Artificial Shifts)**：Cora, Amazon-Photo
2. **时间偏移 (Temporal Shifts)**：Elliptic, OGB-Arxiv
3. **跨域偏移 (Domain Shifts)**：Twitch

本仓库已**统一划分好所有数据集并打包**，并修复了原版代码中生成脚本的缺陷，省去了繁琐且容易报错的官方数据生成过程。

---

## 一、环境准备 (Environment Setup)

建议使用 Python 3.10 及以上版本，并在独立的虚拟环境中配置。

### 1. 安装 PyTorch

如果有 GPU（强烈推荐）：请前往 PyTorch 官网根据你的 CUDA 版本安装。建议使用 PyTorch 2.4.0 左右版本。

如果只有 CPU，请运行：

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cpu
```

### 2. 安装 PyG 扩展库

**千万不要直接 `pip install torch_scatter`**，在 Windows 下极易触发源码编译报错（提示缺少 C++ 编译环境）。请根据你的 PyTorch 版本和硬件，前往 PyG Wheel 官网找对应命令。

如果你使用的是 CPU + PyTorch 2.4.0，直接运行：

```bash
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
```

### 3. 安装其他依赖

```bash
pip install torch_geometric ogb googledrivedownloader numpy scipy scikit-learn
```

---

## 二、数据准备 (Data Preparation)

> **极其重要：请勿运行 `synthetic/create_synthetic.py`！**
>
> 原作者的该脚本存在严重缺陷（写死了 GAT 生成，且生成的标签是随机乱猜的，会导致模型无法收敛）。本仓库已将所有数据集提前按官方逻辑统一划分好。

1. 下载本仓库根目录下的 `data.zip` 压缩包（或前往 Releases 页面下载）。
2. 将 `data.zip` 解压，确保解压后的文件夹名为 `data`，并与 `synthetic`、`temp_elliptic` 等代码文件夹平级。

最终目录结构应如下：

```text
GraphOOD-EERM/
├── data/
│   ├── Amazon/Photo/gen/        (包含 0-gcn.pkl 到 9-gat.pkl)
│   ├── Planetoid/cora/gen/      (包含 0-gcn.pkl 到 9-gat.pkl)
│   ├── elliptic/                (包含 1.pkl 到 48.pkl)
│   ├── ogbn_arxiv/
│   └── twitch/
├── synthetic/
├── temp_arxiv/
├── temp_elliptic/
├── multigraph/
└── README.md
```

---

## 三、实验运行指南 (Running Experiments)

请在终端中进入对应的代码目录，执行相应的命令。


- **如果是纯 CPU 环境，请在命令末尾加上 `--cpu` 参数。**
- **如果使用 GPU，则不需要加。**

### 场景一：人工特征偏移 (Synthetic Shifts)

进入 `synthetic` 目录：

```bash
cd synthetic
```

跑 Cora 数据集（注：Cora 的划分数据是以 `gat` 后缀生成的）：

```bash
python main.py --method erm --dataset cora --gnn_gen gat --gnn gcn --run 20 --lr 0.001
```

跑 Amazon-Photo 数据集（注：Amazon 的划分数据是以 `gcn` 后缀生成的）：

```bash
python main.py --method erm --dataset amazon-photo --gnn_gen gcn --gnn gcn --run 20 --lr 0.001
```

运行结束后，结果将保存在 `synthetic/results/` 目录下的 `.csv` 文件中。

### 场景二：时间偏移 - Elliptic (Temporal Shifts)

退回到根目录，进入 `temp_elliptic` 目录：

```bash
cd temp_elliptic
```

运行实验（数据量较大，CPU 跑较慢，建议将 `run` 次数设小）：

```bash
python main.py --method erm --dataset elliptic --gnn gcn --run 1 --epochs 200 --lr 0.001 --cpu --data_dir ../data
```

### 场景三：时间偏移 - OGB-Arxiv (Temporal Shifts)

进入 `temp_arxiv` 目录：

```bash
cd temp_arxiv
```

运行实验：

```bash
python main.py --method erm --dataset ogbn-arxiv --gnn gcn --run 1 --lr 0.001
```

> **极度警告**：OGB-Arxiv 包含 16 万节点和 116 万条边，纯 CPU 运行极易导致内存溢出死机。请务必在带有 GPU 的服务器上运行，并去掉 `--cpu` 参数。代码底层的 OGB 库会自动读取 `split/time` 文件夹完成时间划分。

### 场景四：跨域偏移 - Twitch (Domain Shifts)

进入 `multigraph` 目录：

```bash
cd multigraph
```

运行实验（代码会自动处理不同国家网络的跨域划分）：

```bash
python main.py --method erm --dataset twitch-e --gnn gcn --run 5 --lr 0.001
```

---

## 四、常见问题排查 (Troubleshooting)

### 1. `FileNotFoundError: No such file or directory: '../data/.../0-gcn.pkl'`

- **原因**：`data` 文件夹没有放在正确的位置，或者命令里的 `--gnn_gen` 参数写错了。
- **解决**：确认 `data` 在根目录下；检查 `gen` 文件夹里的文件后缀是 `gat` 还是 `gcn`，并修改命令中的 `--gnn_gen` 参数与之匹配。

### 2. `ModuleNotFoundError: No module named 'torch_scatter'`

- **原因**：PyG 扩展库未安装或环境不匹配（例如装到了系统全局环境而不是虚拟环境）。
- **解决**：严格遵循"环境准备"第 2 步，确保激活了虚拟环境后重新安装。

### 3. `FileNotFoundError: './results/xxx.csv'`

- **原因**：代码想保存结果，但当前目录下没有 `results` 文件夹。
- **解决**：在报错的当前目录运行 `mkdir results`（或直接在编辑器里右键新建文件夹命名为 `results`）。

### 4. `RuntimeError: Can't call numpy() on Tensor that requires grad`

- **原因**：如果读者在尝试自己写脚本生成数据，PyTorch 的张量带有梯度，无法直接转 numpy。
- **解决**：使用 `tensor.detach().numpy()` 切断梯度。本仓库已修复 `fix_gen.py`，直接使用打包好的数据即可避开此坑。
