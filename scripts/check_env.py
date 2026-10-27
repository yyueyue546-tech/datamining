"""Verify the TT-GREB CPU development environment."""
import sys
from importlib.metadata import version

import matplotlib
import numpy as np
import pytest
import torch
import yaml


def main():
    print("解释器路径：", sys.executable)
    print("Python 版本：", sys.version)
    for package in ("torch", "numpy", "matplotlib", "pyyaml", "pytest"):
        print(f"{package} 版本：", version(package))
    print("CUDA 是否可用：", torch.cuda.is_available())
    print("PyTorch CUDA 编译版本：", torch.version.cuda)
    assert sys.version_info[:2] == (3, 11), "需要 Python 3.11"
    assert torch.version.cuda is None, "需要 CPU 版 PyTorch"
    assert np.array([1, 2]).sum() == 3
    assert yaml.safe_load("enabled: true")["enabled"] is True
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    loss = (x ** 2).sum()
    loss.backward()
    print("计算结果：", loss.item())
    print("梯度：", x.grad)
    assert loss.item() == 5.0
    assert torch.allclose(x.grad, torch.tensor([2.0, 4.0]))
    print("环境检查通过")


if __name__ == "__main__":
    main()
