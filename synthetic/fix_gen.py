import os
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.nn import GCNConv


class Generator(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = GCNConv(in_c, 10)
        self.conv2 = GCNConv(10, out_c)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


def generate_data(dataset_name, is_amazon=False):
    print(f"加载 {dataset_name} 原始数据...")
    if is_amazon:
        dataset = Amazon(root='../data/Amazon', name='Photo')
        data_dir = '../data/Amazon/Photo/gen'
    else:
        dataset = Planetoid(root='../data/Planetoid', name='Cora')
        data_dir = '../data/Planetoid/cora/gen'

    data = dataset[0]
    x, edge_index, y = data.x, data.edge_index, data.y

    gen = Generator(10, 10)
    os.makedirs(data_dir, exist_ok=True)
    print(f"开始生成 {dataset_name} 的 10 个环境划分文件...")

    for i in range(10):
        env_signal = torch.zeros(x.size(0), 10)
        env_signal[:, i] = 1.0
        spurious_feat = gen(env_signal, edge_index)
        x_new = torch.cat([x, spurious_feat], dim=1)

        # 使用 detach() 切断梯度，然后转为 numpy
        x_numpy = x_new.detach().numpy()
        y_numpy = y.numpy()

        # 生成 gcn 和 gat 后缀，双保险
        with open(os.path.join(data_dir, f'{i}-gcn.pkl'), 'wb') as f:
            pkl.dump((x_numpy, y_numpy), f)
        with open(os.path.join(data_dir, f'{i}-gat.pkl'), 'wb') as f:
            pkl.dump((x_numpy, y_numpy), f)

    print(f"✅ {dataset_name} 数据已成功生成到 {data_dir}\n")


if __name__ == '__main__':
    generate_data('Cora', is_amazon=False)
    generate_data('Amazon-Photo', is_amazon=True)