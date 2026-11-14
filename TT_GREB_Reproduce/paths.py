"""全局路径配置。所有数据读写必须经由这里，禁止在其它文件里写相对路径。"""
from __future__ import annotations

from pathlib import Path

# 数据根目录（唯一权威来源）
DATA_ROOT = Path("D:/VSCodePrograms/datamining/data")

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 输出目录
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = PROJECT_ROOT / "logs"

# 各数据集位置
DATASETS = {
    "cora":    DATA_ROOT / "Cora",
    "amazon":  DATA_ROOT / "Amazon" / "Photo",
    "twitch":  DATA_ROOT / "Twitch",
    "ogb":     DATA_ROOT / "OGB" / "ogbn_arxiv",
}

DATASET_META = {
    "cora":   {"metric": "acc", "num_classes": 7,  "num_features": 1433},
    "amazon": {"metric": "acc", "num_classes": 8,  "num_features": 745},
    "twitch": {"metric": "auc", "num_classes": 2,  "num_features": 128},
    "ogb":    {"metric": "acc", "num_classes": 40, "num_features": 128},
}

for _d in (OUTPUT_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)
