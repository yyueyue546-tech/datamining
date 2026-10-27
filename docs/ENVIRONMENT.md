# TT-GREB Python 开发环境

独立 Conda 环境：`ttgreb`；Python：3.11；解释器：`C:\Anaconda\envs\ttgreb\python.exe`。

## 安装命令

在 Anaconda Prompt 中运行：

```bat
conda create -n ttgreb python=3.11 pip -y
conda activate ttgreb
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy matplotlib pyyaml pytest
```

本次默认 Anaconda 源发生 SSL 连接中断，创建环境实际使用以下命令（未修改全局源配置）：

```bat
conda create -n ttgreb python=3.11 pip -y --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
```

PyTorch CPU 专用源下载反复中断，本次实际使用清华 PyPI 镜像安装 Windows wheel；安装后通过 `torch.version.cuda is None` 验证 CPU 构建。实际依赖安装命令：

```bat
python -m pip install torch --index-url https://pypi.tuna.tsinghua.edu.cn/simple --timeout 60
python -m pip install numpy matplotlib pyyaml pytest --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

## VSCode

项目 `.vscode/settings.json` 已指定默认解释器。如果 VSCode 已缓存其他解释器，按 `Ctrl+Shift+P`，执行 `Python: Select Interpreter`，选择或输入：

```text
C:\Anaconda\envs\ttgreb\python.exe
```

## 检查与记录

```bat
conda activate ttgreb
cd /d D:\VSCodePrograms\Test-Time_Graph_Rebirth
python check_env.py
python -m pip check
python -m pip freeze > requirements.txt
```

同伴恢复依赖时，先创建 Python 3.11 环境、激活，然后运行：

```bat
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

CPU 版本验证以 `torch.version.cuda is None` 为准；仅检查 CUDA 是否可用不足以确认安装包为 CPU 版本。
