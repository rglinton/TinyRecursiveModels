from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Union

# --- Op registry ---

ParamSpec = Union[Tuple[int, int], List[int], List[str]]

@dataclass(frozen=True)
class OpSpec:
    name: str
    params: Dict[str, ParamSpec]  # discrete ranges or enums
    base_cost: int = 1

@dataclass
class Op:
    spec: OpSpec
    args: Dict[str, Any]

# NOTE: We keep params simple & discrete.
# For masks, we support: color_eq(val). (You can extend with size_gt / component later.)
OPS_REGISTRY: Dict[str, OpSpec] = {
    "color_map": OpSpec("color_map", params={"src": list(range(10)), "dst": list(range(10))}, base_cost=1),

    # Uses either a persistent selection (set by select) or a one-shot criterion color_eq(val)
    "paint":     OpSpec("paint",     params={"color": list(range(10)), "crit": ["none","color_eq"], "val": list(range(10))}, base_cost=2),

    # Flood fill: for now, fills ALL pixels of seed_color (global region). Seeded BFS can be added later.
    "flood_fill":OpSpec("flood_fill",params={"seed_color": list(range(10)), "color": list(range(10))}, base_cost=2),

    # Axis-aligned copy with clipping
    "copy":      OpSpec("copy",      params={"src_x":(-5,5),"src_y":(-5,5),"w":(1,6),"h":(1,6),"dst_x":(-5,5),"dst_y":(-5,5)}, base_cost=2),

    "reflect":   OpSpec("reflect",   params={"axis":["H","V"]}, base_cost=1),
    "rotate":    OpSpec("rotate",    params={"k":[0,1,2,3]}, base_cost=1),

    "translate": OpSpec("translate", params={"dx":(-5,5),"dy":(-5,5),"mode":["clip","wrap"]}, base_cost=2),

    # Select sets a persistent mask in the executor state. Currently only color criterion.
    "select":    OpSpec("select",    params={"crit":["color","none"], "val": list(range(10))}, base_cost=1),

    # Erase uses either persistent selection or one-shot color_eq(val)
    "erase":     OpSpec("erase",     params={"crit":["none","color_eq"], "val": list(range(10))}, base_cost=1),

    # Simple repetition (tiling with optional gaps), clipped to canvas size
    "repeat":    OpSpec("repeat",    params={"axis":["x","y"], "k":(2,6), "gap":(0,3)}, base_cost=2),
}

# Enable everything (you can restrict for ablations)
ENABLED_OPS = list(OPS_REGISTRY.keys())

def clamp_param(name: str, val, spec: OpSpec):
    rng = spec.params[name]
    if isinstance(rng, tuple):  # integer inclusive range
        lo, hi = rng
        return max(lo, min(hi, int(val)))
    elif isinstance(rng, list):
        if not rng:
            return val
        idx = int(val) % len(rng)
        return rng[idx]
    return val
