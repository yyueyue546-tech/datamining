"""骨干模型注册与统一调用接口（TT-GREB 与第三部分模型之间的唯一桥梁）。

第三部分提供的 GCN / GraphSAGE / GAT / GPR-GNN 只需要满足：

    logits = model(x, edge_index, edge_weight=None)
    logits, embeddings = model.forward_with_embeddings(x, edge_index, edge_weight=None)  # 可选
    model.reset_parameters()                                                             # 可选

本模块负责：
  * 按名字（gcn / sage / gat / gprgnn）或 "包.模块:类名" 创建模型；
  * 按模型真实签名前向（支持 / 不支持 edge_weight 都能调）；
  * 取节点嵌入（forward_with_embeddings / return_embedding 二选一）；
  * 推断类别数（不依赖 model.conv2 这种具体属性名）；
  * 冻结骨干参数。

TT-GREB 主函数只依赖这里，不再直接访问任何具体模型的内部结构。
"""
from __future__ import annotations

import importlib
import inspect
from typing import Any

import torch
from torch import nn

# 名字 -> "模块:类名"。前两个是本仓库自带实现；
# 后两个默认指向第三部分的 models/ 包，实际路径可用 --model-path 追加到 sys.path。
MODEL_REGISTRY: dict[str, str] = {
    "gcn": "tt_greb_reproduce.models:GCN",
    "sage": "tt_greb_reproduce.models:GraphSAGE",
    "graphsage": "tt_greb_reproduce.models:GraphSAGE",
    "gat": "models.gat:GAT",
    "gprgnn": "models.gprgnn:GPRGNN",
}

# 构造参数别名：不同仓库的命名习惯
_IN_DIM_KEYS = ("in_channels", "in_dim", "input_dim", "num_features", "in_feats")
_HIDDEN_KEYS = ("hidden_channels", "hidden_dim", "hidden", "hid_dim")
_OUT_KEYS = ("num_classes", "out_channels", "out_dim", "n_classes")


def load_class(spec: str) -> type:
    """加载 "包.模块:类名" 形式的类。"""
    if ":" not in spec:
        raise ValueError(f"模型路径必须写成 '模块:类名'，收到 {spec!r}")
    mod_name, cls_name = spec.split(":", 1)
    module = importlib.import_module(mod_name)
    try:
        return getattr(module, cls_name)
    except AttributeError as exc:
        raise ImportError(f"{mod_name} 中没有 {cls_name}") from exc


def resolve_model_class(name: str, module: str | None = None) -> type:
    spec = module or MODEL_REGISTRY.get(name.lower())
    if spec is None:
        raise ValueError(
            f"未知骨干 {name!r}。可用名字：{sorted(MODEL_REGISTRY)}，"
            f"或用 module='models.gat:GAT' 指定。"
        )
    try:
        return load_class(spec)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"找不到模型定义 {spec}。第三部分的模型需要满足统一接口，"
            f"并把所在仓库根目录通过 --model-path 加入 import 路径。"
        ) from exc


def build_backbone(
    name: str,
    in_dim: int,
    hidden_dim: int,
    num_classes: int,
    dropout: float = 0.5,
    *,
    module: str | None = None,
    module_kwargs: dict[str, Any] | None = None,
) -> nn.Module:
    """按名字创建骨干。构造参数按目标类签名自动适配。"""
    cls = resolve_model_class(name, module)
    extra = dict(module_kwargs or {})

    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    has_var_kw = any(p.kind == p.VAR_KEYWORD for p in params.values())

    def pick(keys):
        return next((k for k in keys if k in params), None)

    kwargs = dict(extra)
    in_key = pick(_IN_DIM_KEYS)
    hid_key = pick(_HIDDEN_KEYS)
    out_key = pick(_OUT_KEYS)

    if in_key and hid_key and out_key:
        kwargs.setdefault(in_key, in_dim)
        kwargs.setdefault(hid_key, hidden_dim)
        kwargs.setdefault(out_key, num_classes)
        if "dropout" in params:
            kwargs.setdefault("dropout", dropout)
        return cls(**kwargs)

    if has_var_kw:
        kwargs.setdefault("in_channels", in_dim)
        kwargs.setdefault("hidden_channels", hidden_dim)
        kwargs.setdefault("num_classes", num_classes)
        kwargs.setdefault("dropout", dropout)
        return cls(**kwargs)

    # 最后兜底：按位置传参
    return cls(in_dim, hidden_dim, num_classes, **extra)


def _accepts_edge_weight(fn) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    if "edge_weight" in sig.parameters:
        return True
    return any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())


def forward_logits(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """统一前向：自动判断是否接受 edge_weight，自动拆开 (logits, emb) 元组。"""
    forward = model.forward
    if _accepts_edge_weight(forward):
        out = model(x, edge_index, edge_weight=edge_weight)
    else:
        if edge_weight is not None:
            raise TypeError(
                f"{type(model).__name__}.forward 不接受 edge_weight，"
                f"但本次适配传入了边权重；请改用支持 weighted message passing 的模型，"
                f"或让 TT-GREB 保持 edge_weight=None。"
            )
        out = model(x, edge_index)
    return out[0] if isinstance(out, (tuple, list)) else out


def forward_embeddings(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """取节点嵌入：优先 forward_with_embeddings，其次 forward(return_embedding=True)。"""
    if hasattr(model, "forward_with_embeddings"):
        if _accepts_edge_weight(model.forward_with_embeddings):
            out = model.forward_with_embeddings(x, edge_index, edge_weight=edge_weight)
        else:
            out = model.forward_with_embeddings(x, edge_index)
        if isinstance(out, (tuple, list)):
            if len(out) != 2:
                raise ValueError("forward_with_embeddings 应返回 (logits, embeddings)")
            return out[1]
        return out

    params = inspect.signature(model.forward).parameters
    if "return_embedding" in params:
        kwargs = {"return_embedding": True}
        if _accepts_edge_weight(model.forward):
            kwargs["edge_weight"] = edge_weight
        out = model(x, edge_index, **kwargs)
        if isinstance(out, (tuple, list)) and len(out) == 2:
            return out[1]
        raise ValueError(f"{type(model).__name__}.forward 未返回 (logits, embeddings)")

    raise ValueError(
        f"{type(model).__name__} 没有 forward_with_embeddings，"
        f"forward 也没有 return_embedding 参数，无法取节点嵌入。"
    )


def infer_num_classes(model: nn.Module, graph=None) -> int:
    """不依赖具体属性名地推断类别数。"""
    for attr in ("num_classes", "out_channels"):
        value = getattr(model, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    for name in ("conv2", "lin2", "out_layer", "classifier"):
        value = getattr(getattr(model, name, None), "out_channels", None)
        if isinstance(value, int) and value > 0:
            return value
    y = getattr(graph, "y", None) if graph is not None else None
    if y is not None:
        return int(y.max()) + 1
    raise ValueError(
        f"无法从 {type(model).__name__} 推断类别数；"
        f"请给模型加 num_classes 属性，或在 graph.y 中提供标签。"
    )


def freeze_model(model: nn.Module) -> nn.Module:
    """冻结骨干并切到 eval，返回原模型。"""
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
