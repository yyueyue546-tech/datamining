# TT-GREB 本地实验：文件、公式与 Python 实现说明

整理日期：2026 年 10 月 26 日。

本文解释本地已经运行的标准 Cora 实验，包括文件用途、数据字段、GCN 训练、测试时适配、公式与代码对应关系，以及结果含义。

**适用范围：本文中的适配公式来自当前仓库的实际代码，尚未与 TT-GREB 最终论文逐项核对。当前完成的是标准 Cora 的工程链路验证，不能据此宣称完整复现论文。**

## 1. 实验流程

```text
读取标准 Cora 单图
    ↓
使用训练节点标签训练两层 GCN
    ↓
根据验证集 Accuracy 选择最佳 checkpoint
    ↓
恢复 checkpoint，冻结 GCN 参数
    ↓
原始图 → GCN → ERM 测试 Accuracy
    ↓
固定 GCN，优化节点特征权重和边权
    ↓
重加权图 → 同一个 GCN → 适配后测试 Accuracy
```

两种测试使用相同的 GCN 参数和测试节点。区别在于输入是否经过测试时重加权。

## 2. 文件目录及用途

以下路径相对于仓库根目录 `datamining/`。本次实际使用 `TT_GREB_Reproduce/` 下的代码；仓库根目录的同名脚本不作为本次实验入口。

```text
datamining/
├─ data/Cora/
│  ├─ raw/                                  原始 Planetoid 文件
│  └─ processed/data.pt                     本地生成的图缓存
└─ TT_GREB_Reproduce/
   ├─ run_cora_gcn.py                        实验入口
   ├─ tt_greb_reproduce/
   │  ├─ data.py                            数据加载与划分检查
   │  ├─ models.py                          两层 GCN
   │  └─ tt_greb.py                         重加权模块与适配损失
   ├─ verify_local_reproduction.py          四次实验及重复性检查
   ├─ verify_adaptation_invariants.py       标签独立性、参数冻结检查
   ├─ verify_saved_checkpoints.py           加载 checkpoint 复算分数
   ├─ make_reproduction_report.py           生成复现记录
   ├─ INIT.md                              启动说明
   ├─ REPRODUCTION_FEISHU.md                实际运行与结果记录
   ├─ EXPERIMENT_FORMULAS_AND_CODE.md        本文
   ├─ requirements.windows-validated.txt    实测依赖快照
   ├─ local_fixes.patch                     两处运行代码修复补丁
   └─ results/                             实验结果与验收记录
```

| 文件 | 主要作用 |
|---|---|
| `data.py` | 直接读取指定的本地 `data.pt`，校验训练、验证、测试掩码；另提供多图接口 |
| `models.py` | 定义两层 `GCNConv`、ReLU、Dropout 和前向传播 |
| `tt_greb.py` | 定义四个 MLP，产生权重，组合新图，并执行适配优化 |
| `run_cora_gcn.py` | 设置随机种子、训练、验证选模、冻结模型、测试、保存输出 |
| `verify_local_reproduction.py` | 依次执行 50/5 和 200/20，各运行两次；缓存缺失时从本地 raw 生成 |
| `verify_adaptation_invariants.py` | 用小型合成图检查适配不依赖标签，且 GCN 参数未改变并保持冻结 |
| `verify_saved_checkpoints.py` | 重新加载保存参数，核对验证分数和 ERM 分数 |
| `make_reproduction_report.py` | 根据 `verification.json` 生成飞书版 Markdown 记录 |
| `requirements.windows-validated.txt` | 记录本次安装的包版本，不是模型配置 |
| `local_fixes.patch` | 保存运行代码差异，不是模型参数；基于原始代码使用，不能对已修复版本重复应用 |

本次修复了两处运行代码：直接读取指定缓存，避免隐式下载；逐张测试图取得对应 mask，避免沿用旧变量。未更改适配目标函数来追求更高分数。

## 3. Cora 数据及节点掩码

将图记为：

$$
G=(X,E,Y).
$$

| 字段或符号 | 含义 | 本次形状或数量 |
|---|---|---|
| `x`，即 $X$ | 节点特征 | $2708\times1433$ |
| `edge_index`，即 $E$ | 边的两个端点索引 | $2\times10556$ |
| `y`，即 $Y$ | 节点类别标签 | 2708 个，类别为 0～6 |
| `train_mask` | 训练节点布尔掩码 | 140 个 True |
| `val_mask` | 验证节点布尔掩码 | 500 个 True |
| `test_mask` | 测试节点布尔掩码 | 1000 个 True |

10556 是 `edge_index` 中的边条目数，包含图中保存的两个方向，不能直接解释为 10556 对不同的无向节点连接。

每个 mask 长度均为 2708，与节点一一对应。True 表示选中，False 表示未选中。三个集合无重叠，另有 1068 个节点未被这三个 mask 选中。

完整图保留在一起，训练、验证和测试通过选择不同节点完成。未选中节点的特征和连接仍可参与传播，但其标签不参与训练损失或评分。这属于完整图可见的传导式节点分类设置。

**本次只有一个图样本，不是 140 张训练图、500 张验证图和 1000 张测试图，也不是论文说明中的 1/1/8 多图划分。**

### 3.1 processed 中的三个文件

| 文件 | 含义 |
|---|---|
| `data.pt` | 图的特征、边、标签和节点掩码 |
| `pre_filter.pt` | PyG 记录的预筛选配置，不是验证集 |
| `pre_transform.pt` | PyG 记录的预变换配置，不是测试集 |

此前检查的两个配置文件均保存字符串 `'None'`，表示未设置相应函数。本次重新生成缓存使用直接保存方式，只生成 `data.pt`；当前加载器不要求另外两个配置文件。

## 4. GCN 模型公式与实现

原图邻接矩阵为 $A$，加入自连接：

$$
\widetilde A=A+I.
$$

设 $\widetilde D$ 为其度矩阵，归一化矩阵为：

$$
S=\widetilde D^{-1/2}\widetilde A\widetilde D^{-1/2}.
$$

两层 GCN：

$$
H=\operatorname{ReLU}(SXW_0+b_0),
$$

$$
H_{\mathrm{drop}}=\operatorname{Dropout}(H,p=0.5),
$$

$$
Z=SH_{\mathrm{drop}}W_1+b_1.
$$

| 变量 | 含义 | 形状 |
|---|---|---|
| $X$ | 输入特征 | $2708\times1433$ |
| $W_0$ | 第一层线性权重，按数学记号写出 | $1433\times64$ |
| $H$ | 节点隐藏表示 | $2708\times64$ |
| $W_1$ | 第二层线性权重，按数学记号写出 | $64\times7$ |
| $Z$ | 七个类别的原始分数，即 logits | $2708\times7$ |

PyTorch 内部线性层存储权重的轴顺序可能与上述数学乘法记号相反。

对应 `models.py`：

```python
self.conv1 = GCNConv(in_dim, hidden_dim)
self.conv2 = GCNConv(hidden_dim, num_classes)

hidden = self.conv1(x, edge_index, edge_weight=edge_weight).relu()
hidden = F.dropout(hidden, p=self.dropout, training=self.training)
logits = self.conv2(hidden, edge_index, edge_weight=edge_weight)
```

自连接、度归一化和邻居聚合由 `GCNConv` 内部处理。使用边权时，归一化也依赖相应边权。

训练时 Dropout 生效，`model.eval()` 时关闭。代码返回的 embedding 是第一层隐藏表示；本次适配在 eval 状态下执行，因此没有随机 Dropout。

## 5. 监督训练损失

对节点 $i$ 和类别 $c$，类别概率为：

$$
p_{ic}=\frac{\exp(Z_{ic})}{\sum_{k=1}^{7}\exp(Z_{ik})}.
$$

只对训练节点集合 $M_{\mathrm{train}}$ 计算交叉熵：

$$
\mathcal L_{\mathrm{train}}
=-\frac{1}{|M_{\mathrm{train}}|}
\sum_{i\in M_{\mathrm{train}}}\log p_{i,y_i}.
$$

对应 `run_cora_gcn.py` 的 `train_masked()`：

```python
logits = model(graph.x, graph.edge_index)
loss = F.cross_entropy(
    logits[graph.train_mask],
    graph.y[graph.train_mask],
)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

`F.cross_entropy` 已包含稳定的 log-softmax，不需要提前手动 softmax。

优化器为：

```python
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.01, weight_decay=5e-4
)
```

`zero_grad()` 清除旧梯度，`backward()` 计算梯度，`step()` 更新 GCN 参数。权重衰减由优化器处理，代码没有将它作为额外项手动加进交叉熵标量。

## 6. 验证集选模与参数冻结

验证 Accuracy：

$$
\operatorname{Acc}_{\mathrm{val}}
=\frac{1}{|M_{\mathrm{val}}|}
\sum_{i\in M_{\mathrm{val}}}
\mathbf1[\arg\max_c Z_{ic}=y_i].
$$

每轮训练后，只有验证分数严格提高才替换最佳状态：

```python
if valid_acc > best_valid:
    best_valid = valid_acc
    best_state = copy.deepcopy(model.state_dict())
```

训练结束后恢复最佳参数并冻结：

```python
model.load_state_dict(best_state)
model.eval()
for parameter in model.parameters():
    parameter.requires_grad_(False)
```

200 epochs 表示训练 200 轮，最终模型可能来自较早的一轮。当前结果没有保存最佳轮次，不能从 summary 推断具体选中哪轮。

## 7. ERM 基线与测试 Accuracy

ERM 是 Empirical Risk Minimization（经验风险最小化）。本实验中，ERM 测试指使用已经监督训练并经验证集选模的 GCN，直接在原始图上评分，不执行适配。

$$
\widehat y_i=\arg\max_c Z_{ic},
$$

$$
\operatorname{Acc}_{\mathrm{test}}
=\frac{1}{|M_{\mathrm{test}}|}
\sum_{i\in M_{\mathrm{test}}}\mathbf1[\widehat y_i=y_i].
$$

代码先对完整图预测，再通过测试 mask 选择节点：

```python
logits = model(graph.x, graph.edge_index)
logits, labels = logits[mask], graph.y[mask]
score = (logits.argmax(-1) == labels).float().mean()
```

本次 79.60% 表示 1000 个测试节点中正确 796 个。

## 8. 测试时适配器：四个 MLP

适配阶段冻结 GCN 参数 $\theta$，只更新适配器参数 $\phi$。

```python
adapter = TTGREBAdapter(graph.x.size(1))
optimizer = torch.optim.Adam(adapter.parameters(), lr=0.01)
```

单个 MLP 的形式为：

$$
\operatorname{MLP}(u)
=\sigma\bigl(W_2\operatorname{ReLU}(W_1u+b_1)+b_2\bigr).
$$

Sigmoid $\sigma$ 将输出限制在 0 和 1 之间。

```python
self.net = nn.Sequential(
    nn.Linear(in_dim, in_dim),
    nn.ReLU(),
    nn.Linear(in_dim, out_dim),
)
return torch.sigmoid(self.net(value))
```

| 模块 | 输出记号 | 输入维度 | 输出形状 |
|---|---|---:|---|
| `prototype_node` | $P_X$ | 1433 | $2708\times1433$ |
| `environment_node` | $E_X$ | 1433 | $2708\times1433$ |
| `prototype_edge` | $P_E$ | 2866 | 10556 个标量 |
| `environment_edge` | $E_E$ | 2866 | 10556 个标量 |

prototype/environment 是模块名称，不表示数据中存在人工标注的“原型节点”或“环境节点”。

对边 $(i,j)$，输入拼接两端特征：

$$
u_{ij}=[x_i\Vert x_j].
$$

```python
src, dst = graph.edge_index
edge_input = torch.cat([x[src], x[dst]], dim=-1)
```

节点模块给每个节点的每个特征维度一个权重，并非每个节点只有一个标量。边模块给每个已保存的边条目一个标量；代码未强制两个相反方向的边权相同。

## 9. 图重加权公式

最终节点特征权重：

$$
W_X=P_X+(1-P_X)\odot E_X.
$$

最终边权：

$$
w_E=P_E+(1-P_E)\odot E_E.
$$

新节点特征：

$$
X'=X\odot W_X.
$$

对应 `TTGREBAdapter.rebirth()`：

```python
node_weight = (1.0 - p_node) * e_node + p_node
edge_weight = (1.0 - p_edge) * e_edge + p_edge
reborn = Data(
    x=graph.x * node_weight,
    edge_index=graph.edge_index,
    y=graph.y,
    edge_weight=edge_weight,
)
```

例如 $P_X=0.8$、$E_X=0.3$，最终权重为 $0.8+0.2\times0.3=0.86$。

当前实现保持 `edge_index` 不变，不新增边、不执行硬删除，也不改变节点数。节点特征和已有边被连续重加权。权重直接乘法主要起保留或衰减作用，但边权会改变 GCN 的归一化，不能将最终输出理解为简单等比例衰减。

## 10. 实际适配目标函数

总损失为：

$$
\boxed{\mathcal L_{\mathrm{adapt}}
=\mathcal L_{\mathrm{pro}}
+\alpha\mathcal L_{\mathrm{env}}
+\beta\mathcal L_{\mathrm{ratio}}}
$$

本次 $\alpha=0.1$、$\beta=0.1$。

```python
loss = loss_pro + alpha * loss_env + beta * regularizer
```

下面将冻结 GCN 的隐藏表示函数记为 $h_\theta$。不同视图使用各自对应的边权及归一化。

### 10.1 prototype 项：余弦相似度

重加权图的表示：

$$
H^R=h_\theta(X',E,w_E).
$$

prototype 视图的表示：

$$
H^P=h_\theta(X\odot P_X,E).
$$

损失为：

$$
\mathcal L_{\mathrm{pro}}
=1-\frac1N\sum_{i=1}^N
\frac{(H_i^R)^\top H_i^P}{\|H_i^R\|_2\|H_i^P\|_2}.
$$

实际实现由 PyTorch 处理接近零的范数：

```python
loss_pro = 1.0 - F.cosine_similarity(
    prototype_embedding, prototype_view, dim=-1
).mean()
```

最小化该项让两个表示的方向接近。注意代码变量 `prototype_embedding` 实际来自完整重加权图，`prototype_view` 来自只乘 `p_node` 的图。

**当前 prototype 视图没有传入 `p_edge`，因此不能把该视图解释为同时采用 prototype 节点权重和 prototype 边权。** `p_edge` 仍通过最终组合边权及正则参与优化。

### 10.2 environment 项：负均方差

原始图表示与环境视图表示：

$$
H^0=h_\theta(X,E),\qquad
H^E=h_\theta(X\odot E_X,E,E_E).
$$

损失为：

$$
\mathcal L_{\mathrm{env}}
=-\frac1{Nd_h}\sum_{i=1}^N\sum_{k=1}^{d_h}
(H^0_{ik}-H^E_{ik})^2,
\qquad d_h=64.
$$

```python
loss_env = -(
    (original_embedding.detach() - environment_embedding) ** 2
).mean()
```

由于负号，最小化该项会推动原始表示与环境表示的差异增大。这是当前代码的数学行为，并不等于已经核实论文完整的环境目标。

### 10.3 权重均值比例正则

单个正则：

$$
R(W,r)=(\operatorname{mean}(W)-r)^2.
$$

总正则：

$$
\mathcal L_{\mathrm{ratio}}
=R(P_X,0.5)+R(P_E,0.5)+R(E_X,0.5)+R(E_E,0.5).
$$

```python
def _ratio_regularizer(weight, target):
    return (weight.mean() - target).square()
```

这里 0.5 是四组软权重各自的均值目标，不代表硬性保留一半节点或边，也不强制最终组合权重均值等于 0.5。

## 11. 适配优化和标签使用边界

每一步适配：

```python
reborn, weights = adapter.rebirth(graph)
# 用冻结 GCN 计算不同视图的表示及损失
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

优化变量是 $\phi$，GCN 参数 $\theta$ 固定：

$$
\min_\phi\mathcal L_{\mathrm{adapt}}(\phi;\theta,G),
\qquad \theta\text{ 固定}.
$$

适配损失没有读取真实标签。虽然构造的 `Data` 对象携带 `y`，其用途是最后评分。

冻结 GCN 参数不意味着适配前向传播可以全部使用 `torch.no_grad()`：梯度仍需经过 GCN 对输入的计算，回传给适配器。原图参考表示在 no-grad 下计算，而被优化视图保留梯度路径。

本次适配损失对完整图节点求平均，没有使用 `test_mask` 限制损失计算范围。`test_mask` 只限制最后的测试评分。

原始 logits 虽在适配开始时计算，但当前损失没有使用它；实际参考量是原始隐藏表示。

## 12. 超参数与运行配置

### 12.1 本次实验配置概述

**GCN：**使用两层 `GCNConv`，隐藏维度为 64，`dropout=0.5`；训练采用 Adam，学习率为 0.01，`weight_decay=0.0005`。每轮训练后计算验证集 Accuracy，仅在验证 Accuracy 严格提高时保存最佳状态。训练结束后恢复该最佳状态，再冻结所有 GCN 参数。Dropout 仅在训练状态生效，验证、测试和适配时关闭。

**适配：**采用节点和边的 prototype/environment MLP 进行重加权，共四个 MLP。适配采用独立的 Adam 优化器，学习率为 0.01，`alpha=0.1`、`beta=0.1`，`prototype_ratio` 和 `environment_ratio` 两个比例目标均为 0.5。当前实现保持 `edge_index` 不变，修改节点特征和已有边的权重；适配损失包括余弦相似度项、环境差异项（负均方差）和权重均值比例正则项。测试时真实标签仅供最终评分，没有进入适配损失。适配阶段仅更新适配器，GCN 参数保持冻结。

这里的比例目标约束各组软权重的均值，不表示硬性保留 50% 的节点或边。上述设置描述当前实际代码，不表示已经与最终论文逐项核对。

### 12.2 参数对照表

| 字段 | 本次值 | 含义 |
|---|---|---|
| `seed` | 0 | 随机种子，控制模型及适配器初始化等随机过程 |
| `epochs` | 50 或 200 | GCN 训练轮数 |
| `hidden_dim` | 64 | GCN 隐藏维度 |
| `learning_rate` | 0.01 | GCN 训练学习率 |
| `weight_decay` | 0.0005 | GCN 优化器权重衰减 |
| `adapt_steps` | 5 或 20 | 适配器更新次数 |
| 适配 `lr` | 0.01 | 适配器 Adam 学习率 |
| `alpha` | 0.1 | 环境损失系数 |
| `beta` | 0.1 | 均值比例正则系数 |
| `prototype_ratio` | 0.5 | prototype 权重均值目标 |
| `environment_ratio` | 0.5 | environment 权重均值目标 |
| `device` | cpu | 运算设备 |
| Python / PyTorch / PyG | 3.11.17 / 2.14.1+cpu / 2.6.1 | 实测环境 |

`config.json` 记录训练配置，但没有完整记录适配 lr、alpha、beta、两个比例目标和 PyG 版本；适配参数由函数默认值提供。因此交接应同时保存配置、代码和依赖快照。

`smoke_only=false` 只表示没有采用 `--smoke-standard-cora` 参数，不能据此认定为论文正式实验。

## 13. 输出字段及损失解释

| 字段 | 含义 |
|---|---|
| `evaluation_protocol: node_masks` | 同一张图内按节点 mask 划分 |
| `paper_split_verified: false` | 未验证为论文多图划分 |
| `train_graphs / valid_graphs / test_graphs` | 用途计数；本次引用同一张图，不能相加为三张独立图 |
| `validation_accuracy` | 最佳 checkpoint 的验证集分数 |
| `erm_accuracy` | 每个测试图在原始输入上的 Accuracy 列表 |
| `tt_greb_accuracy` | 每个测试图适配后的 Accuracy 列表 |
| `erm_mean_accuracy` | 测试图 ERM 分数平均值；本次只有一个元素 |
| `tt_greb_mean_accuracy` | 测试图适配分数平均值 |
| `adaptation.steps` | 适配更新次数 |
| `adaptation.final_loss` | 最后一次更新前计算的适配总损失 |

`final_loss` 不是交叉熵、错误率或 Accuracy。环境项为负，因此负损失不自动代表错误，也不能直接推断精度高低。

代码记录最后一步更新前的 loss，然后执行参数更新；循环后再次生成最终图。因此该值不是对最终图额外重新计算的损失。

`checkpoint.pt` 仅保存训练后选中的 GCN 参数。当前没有保存最终适配器参数或重加权图，不能仅加载 checkpoint 就直接恢复适配后的分数；需要按相同流程重新适配。

## 14. 本次结果与验收

| 配置 | 最佳验证 Accuracy | ERM 测试 Accuracy | 当前适配实现 Accuracy | 变化 |
|---|---:|---:|---:|---:|
| 50 epochs / 5 steps | 79.00% | 79.60% | 79.20% | -0.40 个百分点 |
| 50 epochs / 5 steps，重复 | 79.00% | 79.60% | 79.20% | -0.40 个百分点 |
| 200 epochs / 20 steps | 79.00% | 79.60% | 79.80% | +0.20 个百分点 |
| 200 epochs / 20 steps，重复 | 79.00% | 79.60% | 79.80% | +0.20 个百分点 |

测试节点为 1000 个：50/5 从正确 796 个变为 792 个；200/20 从正确 796 个变为 798 个。

验收结果：

- 两组配置各自重复的 summary 完全一致，checkpoint 参数张量完全一致。
- 四份 checkpoint 重新加载后，验证和 ERM 分数均与记录一致。
- 小型合成图中，仅改变标签并重置适配种子，输出特征和边权完全一致。
- 适配前后 GCN 参数完全一致，并保持冻结。

同种子重复验证的是本机环境下的可重复性，不代表跨平台一致性或多个种子的统计稳定性。200/20 提升两个节点，尚不足以说明方法稳定优于 ERM。

## 15. 与论文正式复现的区别

当前公式中没有显式负样本、温度参数或 InfoNCE 对数比值，也没有硬删除节点或边。它是仓库明确注明的可运行替代目标，不能仅凭模块名称认定其等价于论文算法。

正式复现仍需：

1. 获得并核验论文/作者的 Cora 构图和多图划分；仅检查数量为 1/1/8 不够。
2. 对照最终论文或官方实现核对目标函数、视图构造、梯度路径、优化步骤和超参数。
3. 按相同数据协议完成多随机种子实验，报告均值和波动。
4. 补充论文要求的其他数据集、模型及 baseline。

本次可以准确表述为：**在标准 Cora 单图上训练并冻结两层 GCN，通过四个 MLP 对节点特征及已有边重加权，使用余弦相似度项、负均方差项和权重均值正则进行无标签适配，并在固定测试节点上比较 Accuracy。**

## 16. 阅读代码的建议顺序

1. `data.py` 的 `load_masked_cora()`：理解数据加载和 mask。
2. `models.py` 的 `GCN.forward()`：理解原始预测和 embedding。
3. `run_cora_gcn.py` 的 `train_masked()`：理解训练、验证选模与冻结。
4. `tt_greb.py` 的 `weights()` 和 `rebirth()`：理解权重及新图。
5. `tt_greb.py` 的 `adapt_test_graph()`：逐项对照适配损失与更新。
6. `run_cora_gcn.py` 的 `main()`、`accuracy()`：理解评分和结果保存。

实际命令、耗时、数据 SHA-256 和实验交接细节见同目录的 `REPRODUCTION_FEISHU.md` 及 `results/verification.json`。
