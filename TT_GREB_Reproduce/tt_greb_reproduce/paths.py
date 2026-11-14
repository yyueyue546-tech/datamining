"""全局路径配置。所有数据读写必须经由这里，禁止在其它文件里写相对路径。

数据目录优先级：
  1. 环境变量 TT_GREB_DATA（推荐，换机器不用改代码）
  2. 仓库根目录下的 data/（克隆后把数据放这里，默认值）
"""
from __future__ import annotations

import os
from pathlib import Path

# 项目根目录（TT_GREB_Reproduce/）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent

# 数据根目录（唯一权威来源）
DATA_ROOT = Path(os.environ.get("TT_GREB_DATA", REPO_ROOT / "data")).expanduser()

# 输出目录
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = PROJECT_ROOT / "logs"

# 各数据集位置
DATASETS = {
    "cora":    DATA_ROOT / "Planetoid" / "cora",
    "amazon":  DATA_ROOT / "Amazon" / "Photo",
    "twitch":  DATA_ROOT / "twitch" / "DE",
    "ogb":     DATA_ROOT / "ogbn_arxiv",
    "elliptic": DATA_ROOT / "elliptic",
    "facebook100": DATA_ROOT / "facebook100",
}

DATASET_META = {
    "cora":   {"metric": "acc", "num_classes": 7,  "num_features": 1433},
    "amazon": {"metric": "acc", "num_classes": 8,  "num_features": 745},
    "twitch": {"metric": "auc", "num_classes": 2,  "num_features": 3170},
    "ogb":    {"metric": "acc", "num_classes": 40, "num_features": 128},
    # Elliptic 正负样本约 1:9，acc 会被多数类主导，按 EERM/GOOD 惯例用 ROC-AUC
    "elliptic": {"metric": "auc", "num_classes": 2, "num_features": 165},
}

for _d in (OUTPUT_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)
