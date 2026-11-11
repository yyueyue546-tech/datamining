import os
import torch
import numpy as np
from ogb.nodeproppred import NodePropPredDataset


def fix_arxiv():
    print("正在检查并重新下载 OGB-Arxiv 数据（请确保网络畅通，这可能需要几分钟）...")
    # 强制 OGB 库重新去下载缺失的文件
    dataset = NodePropPredDataset(name='ogbn-arxiv', root='./data')

    # OGB 库有时候不会自动补全缺失的文件，我们手动检查一下
    if not os.path.exists('./data/ogbn_arxiv/raw/node-feat.csv.gz'):
        print("❌ 特征文件仍然缺失！请检查网络连接后重试。")
        return

    print("✅ 特征文件已补齐，正在提取测试数据...")
    graph = dataset.graph
    print(f"特征矩阵大小: {graph['node_feat'].shape}")
    print("🎉 恭喜！OGB-Arxiv 数据完整，代码现在可以跑了！")


if __name__ == '__main__':
    fix_arxiv()