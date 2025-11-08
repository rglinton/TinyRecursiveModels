# dsl/ops.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, List, Union
import torch

# ----------------------------
# Types
# ----------------------------
ParamSpec = Union[Tuple[int, int], List[int]]

@dataclass
class OpSpec:
    name: str
    params: Dict[str, ParamSpec]
    fn: Callable
    base_cost: float = 1.0
    param_cost: Dict[str, float] | None = None  # cost per parameter name

# Back-compat: some modules import `Op`
Op = OpSpec

# Global registry
OPS_REGISTRY: Dict[str, OpSpec] = {}

def register_op(
    name: str,
    params: Dict[str, ParamSpec],
    *,
    base_cost: float = 1.0,
    param_cost: Dict[str, float] | None = None,
):
    """
    Decorator to register an op with parameter specs and costs.
    `base_cost` is used by mdl_features(); `param_cost` maps param->cost (default 0.1 each).
    """
    def deco(fn: Callable):
        pc = param_cost if param_cost is not None else {p: 0.1 for p in params.keys()}
        OPS_REGISTRY[name] = OpSpec(name=name, params=params, fn=fn, base_cost=base_cost, param_cost=pc)
        return fn
    return deco

# ----------------------------
# Minimal curriculum ops
# ----------------------------
# NOTE: Param ranges are intentionally conservative to ease initial learning.

@register_op(
    "paint",
    params={"x": (0, 29), "y": (0, 29), "color": (0, 9)},
    base_cost=1.0,
)
def op_paint(grid: torch.Tensor, *, x: int, y: int, color: int):
    H, W = grid.shape[-2], grid.shape[-1]
    x = max(0, min(W - 1, int(x)))
    y = max(0, min(H - 1, int(y)))
    color = max(0, min(9, int(color)))
    out = grid.clone()
    out[..., y, x] = color
    return out

@register_op(
    "translate",
    params={"dx": (-3, 3), "dy": (-3, 3)},
    base_cost=1.2,
)
def op_translate(grid: torch.Tensor, *, dx: int, dy: int):
    dx, dy = int(dx), int(dy)
    return torch.roll(grid, shifts=(dy, dx), dims=(-2, -1))

@register_op(
    "copy",
    params={
        "src_x": (0, 29), "src_y": (0, 29),
        "dst_x": (0, 29), "dst_y": (0, 29),
        "w": (1, 6), "h": (1, 6),
    },
    base_cost=1.5,
)
def op_copy(grid: torch.Tensor, *, src_x: int, src_y: int, dst_x: int, dst_y: int, w: int, h: int):
    H, W = grid.shape[-2], grid.shape[-1]
    src_x, src_y = int(src_x), int(src_y)
    dst_x, dst_y = int(dst_x), int(dst_y)
    w, h = int(w), int(h)
    # clamp rectangles to fit
    src_x = max(0, min(W - 1, src_x))
    src_y = max(0, min(H - 1, src_y))
    dst_x = max(0, min(W - 1, dst_x))
    dst_y = max(0, min(H - 1, dst_y))
    w = max(1, min(W, w))
    h = max(1, min(H, h))
    src_x2 = min(W, src_x + w)
    src_y2 = min(H, src_y + h)
    dst_x2 = min(W, dst_x + (src_x2 - src_x))
    dst_y2 = min(H, dst_y + (src_y2 - src_y))
    # effective copy size
    w_eff = min(src_x2 - src_x, dst_x2 - dst_x)
    h_eff = min(src_y2 - src_y, dst_y2 - dst_y)

    out = grid.clone()
    if w_eff > 0 and h_eff > 0:
        out[..., dst_y:dst_y+h_eff, dst_x:dst_x+w_eff] = grid[..., src_y:src_y+h_eff, src_x:src_x+w_eff]
    return out

@register_op(
    "erase",
    params={"x": (0, 29), "y": (0, 29), "w": (1, 6), "h": (1, 6)},
    base_cost=1.0,
)
def op_erase(grid: torch.Tensor, *, x: int, y: int, w: int, h: int):
    H, W = grid.shape[-2], grid.shape[-1]
    x, y, w, h = int(x), int(y), int(w), int(h)
    x = max(0, min(W - 1, x))
    y = max(0, min(H - 1, y))
    x2 = max(x + 1, min(W, x + w))
    y2 = max(y + 1, min(H, y + h))
    out = grid.clone()
    out[..., y:y2, x:x2] = 0
    return out

@register_op(
    "color_map",
    params={"src_color": (0, 9), "dst_color": (0, 9)},
    base_cost=1.0,
)
def op_color_map(grid: torch.Tensor, *, src_color: int, dst_color: int):
    src_color = max(0, min(9, int(src_color)))
    dst_color = max(0, min(9, int(dst_color)))
    out = grid.clone()
    mask = (out == src_color)
    out[mask] = dst_color
    return out

# Which ops are enabled for search/proposal right now
ENABLED_OPS: List[str] = ["paint", "translate", "copy", "erase", "color_map"]

