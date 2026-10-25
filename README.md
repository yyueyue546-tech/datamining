# TT_GREB_Reproduce
复现 TT-GREB 论文实验
# 项目简介
本仓库复现 TT-GREB 论文实验，包含 5 个图数据集（Cora、Amazon-Photo、OGB-Arxiv、Twitch-E、Elliptic）的加载与校验脚本。
# 环境要求
- Python 3.9+
- PyTorch 2.6
- PyTorch Geometric 2.6.1
- ogb、numpy
# 安装
pip install torch torch_geometric ogb numpy
# 目录结构
TT_GREB_Reproduce/
├── datasets/
│   └── data/
│       ├── Amazon/
│       ├── Cora/
│       ├── Elliptic/
│       ├── OGB/
│       └── Twitch/
├── main.py
└── README.md
# 使用方法
运行主程序：python main.py
# 常见问题
PyTorch 2.6 加载缓存报错 ：需在任何 torch.load 之前调用torch.serialization.add_safe_globals([...]) 注册安全全局类。
路径解析错误 ：不要混用绝对路径和裸相对路径，建议统一基于脚本位置
拼出数据根目录。

