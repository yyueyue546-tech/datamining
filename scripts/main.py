# ============================================================
# 解决 PyTorch 2.6 的 torch.load 兼容性问题
# ============================================================
import torch
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage, NodeStorage, EdgeStorage

torch.serialization.add_safe_globals([
    DataEdgeAttr,
    DataTensorAttr,
    GlobalStorage,
    NodeStorage,
    EdgeStorage,
])


# ============================================================
# 1. Cora
# ============================================================
from torch_geometric.datasets import Planetoid

dataset_cora = Planetoid(root='data/Cora', name='Cora')
data_cora = dataset_cora[0]

print("=" * 40)
print("Cora")
print("=" * 40)
print(f"节点数: {data_cora.num_nodes}")
print(f"边数: {data_cora.num_edges}")
print(f"特征维度: {data_cora.num_features}")
print(f"类别数: {dataset_cora.num_classes}")


# ============================================================
# 2. Amazon-Photo
# ============================================================
from torch_geometric.datasets import Amazon

dataset_amazon = Amazon(root='data/Amazon', name='Photo')
data_amazon = dataset_amazon[0]

print("=" * 40)
print("Amazon-Photo")
print("=" * 40)
print(f"节点数: {data_amazon.num_nodes}")
print(f"边数: {data_amazon.num_edges}")
print(f"特征维度: {data_amazon.num_features}")
print(f"类别数: {dataset_amazon.num_classes}")


# ============================================================
# 3. OGB-Arxiv
# ============================================================
from ogb.nodeproppred import PygNodePropPredDataset

dataset_ogb = PygNodePropPredDataset(name='ogbn-arxiv', root='data/OGB')
data_ogb = dataset_ogb[0]

print("=" * 40)
print("OGB-Arxiv")
print("=" * 40)
print(f"节点数: {data_ogb.num_nodes}")
print(f"边数: {data_ogb.num_edges}")
print(f"特征维度: {data_ogb.num_features}")

split_idx = dataset_ogb.get_idx_split()
print(f"训练集: {len(split_idx['train'])}")
print(f"验证集: {len(split_idx['valid'])}")
print(f"测试集: {len(split_idx['test'])}")


# ============================================================
# 4. Twitch-E (DE) —— 手动构造
# ============================================================
import json
import os
import csv
import torch
from torch_geometric.data import Data

# ============================================================
# 路径
# ============================================================
BASE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'data', 'Twitch', 'raw'
)

FEATURES_PATH = os.path.join(BASE_DIR, 'musae_DE_features.json')
EDGES_PATH    = os.path.join(BASE_DIR, 'musae_DE_edges.csv')
TARGET_PATH   = os.path.join(BASE_DIR, 'musae_DE_target.csv')

print(f"查找路径: {BASE_DIR}")


# ============================================================
# 1. 读特征（JSON）
# ============================================================
with open(FEATURES_PATH, 'r', encoding='utf-8') as f:
    features = json.load(f)

# 确定节点数和最大特征维度
num_nodes = max(int(k) for k in features.keys()) + 1
feat_dim = max(len(v) for v in features.values())
print(f"节点数: {num_nodes}, 最大特征维度: {feat_dim}")

# 创建全零矩阵，把每个节点特征填进去
x = torch.zeros(num_nodes, feat_dim)
for k, v in features.items():
    x[int(k), :len(v)] = torch.tensor(v, dtype=torch.float)

print(f"特征加载完成: {x.shape}")


# ============================================================
# 2. 读边（CSV，带表头 from,to）
# ============================================================
edges = []
with open(EDGES_PATH, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        if len(row) >= 2:
            edges.append([int(row[0]), int(row[1])])
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
print(f"边加载完成: {edge_index.shape}")


# ============================================================
# 3. 读标签（用 mature 列）
# ============================================================
labels = [0] * num_nodes
with open(TARGET_PATH, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        new_id = int(row['new_id'])
        mature = row['mature'].strip().upper()
        labels[new_id] = 1 if mature == 'TRUE' else 0
y = torch.tensor(labels, dtype=torch.long)
print(f"标签加载完成: {y.shape}")


# ============================================================
# 4. 构造 PyG Data
# ============================================================
data_twitch = Data(x=x, edge_index=edge_index, y=y)

print("=" * 40)
print("Twitch-E (DE)")
print("=" * 40)
print(f"节点数: {data_twitch.num_nodes}")
print(f"边数: {data_twitch.num_edges}")
print(f"特征维度: {data_twitch.num_features}")
print(f"类别数: 2")
print(f"标签分布: 0={int((y==0).sum())}, 1={int((y==1).sum())}")

# ============================================================
# 5. Elliptic（按时间步切分）
# ============================================================
import os
import csv
import numpy as np
import torch
from torch_geometric.data import Data

BASE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'data', 'Elliptic', 'raw'
)

FEATURES_PATH = os.path.join(BASE_DIR, 'elliptic_txs_features.csv')
EDGES_PATH    = os.path.join(BASE_DIR, 'elliptic_txs_edgelist.csv')
CLASSES_PATH  = os.path.join(BASE_DIR, 'elliptic_txs_classes.csv')

# --- 5.1 读特征 ---
print("正在读取 Elliptic 特征...")
tx_ids = []          # 原始交易 ID 列表
time_steps = []      # 每个节点对应的时间步
features_list = []   # 每个节点的特征向量

with open(FEATURES_PATH, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        tx_ids.append(int(row[0]))
        time_steps.append(int(row[1]))
        features_list.append([float(x) for x in row[2:]])

tx_ids = np.array(tx_ids)
time_steps = np.array(time_steps)
x = torch.tensor(features_list, dtype=torch.float)

# 建立 txId -> 节点索引 的映射
txid_to_idx = {tx_id: i for i, tx_id in enumerate(tx_ids)}

print(f"特征加载完成: x={x.shape}, 时间步范围: {time_steps.min()}~{time_steps.max()}")

# --- 5.2 读标签 ---
print("正在读取 Elliptic 标签...")
y = torch.full((len(tx_ids),), -1, dtype=torch.long)  # -1 表示 unknown

with open(CLASSES_PATH, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        tx_id = int(row['txId'])
        if tx_id not in txid_to_idx:
            continue
        idx = txid_to_idx[tx_id]
        cls = row['class'].strip()
        if cls == '1':
            y[idx] = 0   # 非法
        elif cls == '2':
            y[idx] = 1   # 合法
        # unknown 保持 -1

labeled = (y >= 0).sum().item()
print(f"标签加载完成: 有标签节点={labeled}, 无标签节点={len(y)-labeled}")

# --- 5.3 读边 ---
print("正在读取 Elliptic 边...")
edges = []
with open(EDGES_PATH, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # 跳过表头
    for row in reader:
        if len(row) < 2:
            continue
        tx1, tx2 = int(row[0]), int(row[1])
        if tx1 in txid_to_idx and tx2 in txid_to_idx:
            edges.append([txid_to_idx[tx1], txid_to_idx[tx2]])

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
print(f"边加载完成: {edge_index.shape}")

# --- 5.4 按时间步切分成多张图 ---
print("正在按时间步切分...")

# 论文划分：训练 1~5，验证 6~10，测试 11~43
train_steps = list(range(1, 6))    # 1,2,3,4,5
val_steps   = list(range(6, 11))   # 6,7,8,9,10
test_steps  = list(range(11, 44))  # 11~43

def build_subgraph(step):
    """取出某个时间步的所有节点，构造一张子图"""
    mask = (time_steps == step)
    node_indices = np.where(mask)[0]
    if len(node_indices) == 0:
        return None

    # 重新编号：全局索引 -> 子图内索引
    global_to_local = {g: l for l, g in enumerate(node_indices)}

    # 节点特征和标签
    sub_x = x[node_indices]
    sub_y = y[node_indices]

    # 只保留两端都在这个时间步的边
    sub_edges = []
    for u, v in edges:
        if u in global_to_local and v in global_to_local:
            sub_edges.append([global_to_local[u], global_to_local[v]])

    if len(sub_edges) == 0:
        sub_edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        sub_edge_index = torch.tensor(sub_edges, dtype=torch.long).t().contiguous()

    return Data(x=sub_x, edge_index=sub_edge_index, y=sub_y)


train_graphs = [build_subgraph(s) for s in train_steps]
val_graphs   = [build_subgraph(s) for s in val_steps]
test_graphs  = [build_subgraph(s) for s in test_steps]

# 打印统计
print("=" * 40)
print("Elliptic")
print("=" * 40)
print(f"训练图数量: {len([g for g in train_graphs if g is not None])}")
print(f"验证图数量: {len([g for g in val_graphs if g is not None])}")
print(f"测试图数量: {len([g for g in test_graphs if g is not None])}")
print()
print(f"第一个训练图: 节点数={train_graphs[0].num_nodes}, 边数={train_graphs[0].num_edges}")
print(f"第一个测试图: 节点数={test_graphs[0].num_nodes}, 边数={test_graphs[0].num_edges}")