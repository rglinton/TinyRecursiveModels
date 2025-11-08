# dsl/program.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Iterable, Tuple, Optional

# Only need the registry here; no Op/OpSpec types required
from dsl.ops import OPS_REGISTRY


# ----------------------------
# Core data structures
# ----------------------------

@dataclass
class ProgramStep:
    op_name: str
    args: Dict[str, int]

    def to_tuple(self) -> Tuple[str, Dict[str, int]]:
        return (self.op_name, dict(self.args))

    @staticmethod
    def from_tuple(t: Tuple[str, Dict[str, int]]) -> "ProgramStep":
        op, args = t
        # normalize to ints where possible
        norm = {k: int(v) for k, v in args.items()}
        return ProgramStep(op_name=op, args=norm)

    def __repr__(self) -> str:
        # nice compact printout like: ('paint', {'x':1,'y':2,'color':3})
        return f"('{self.op_name}', {self.args})"


@dataclass
class Program:
    steps: List[ProgramStep] = field(default_factory=list)

    def add(self, step: ProgramStep) -> None:
        self.steps.append(step)

    def extend(self, steps: Iterable[ProgramStep]) -> None:
        self.steps.extend(list(steps))

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    def to_list(self) -> List[Tuple[str, Dict[str, int]]]:
        return [s.to_tuple() for s in self.steps]

    @staticmethod
    def from_list(items: List[Tuple[str, Dict[str, int]]]) -> "Program":
        return Program([ProgramStep.from_tuple(t) for t in items])

    def __repr__(self) -> str:
        # prints like: [('paint', {...}), ('translate', {...})]
        return "[" + ", ".join(repr(s) for s in self.steps) + "]"


# ----------------------------
# MDL-ish features for Energy
# ----------------------------

def _is_noop_step(step: ProgramStep) -> bool:
    """
    Detect obvious semantic no-ops we want to penalize:
      - translate(0,0)
      - color_map(a -> a)
      - copy with identical src rect == dst rect (and positive w,h)
    Bounds/empty rectangles etc. are left to the executor.
    """
    op = step.op_name
    a: Dict[str, int] = step.args

    if op == "translate":
        return int(a.get("dx", 0)) == 0 and int(a.get("dy", 0)) == 0

    if op == "color_map":
        return int(a.get("src_color", -1)) == int(a.get("dst_color", -2))

    if op == "copy":
        sx, sy = int(a.get("src_x", -999)), int(a.get("src_y", -999))
        dx, dy = int(a.get("dst_x", -999)), int(a.get("dst_y", -999))
        w, h = int(a.get("w", 0)), int(a.get("h", 0))
        return sx == dx and sy == dy and w > 0 and h > 0

    return False


def mdl_features(program: Program) -> Dict[str, float]:
    """
    Produce a compact feature dict describing the program for the Energy head.

    Fields:
      - base_cost: sum of per-op base_cost (if op spec defines it; else 1.0)
      - num_steps: number of steps
      - unique_ops: number of distinct ops used
      - consec_dupes: count of consecutive identical steps (same op+args)
      - noop_steps: count of obvious no-op steps (per _is_noop_step)
    """
    steps: List[ProgramStep] = list(program.steps)
    num_steps = float(len(steps))
    unique_ops = len({s.op_name for s in steps})

    # Sum up costs from op specs; default to 1.0 if not present
    base_cost = 0.0
    for s in steps:
        spec = OPS_REGISTRY.get(s.op_name)
        base_cost += float(getattr(spec, "base_cost", 1.0)) if spec is not None else 1.0

    # Count consecutive duplicates (identical op and identical args)
    consec_dupes = 0
    for i in range(1, len(steps)):
        a, b = steps[i - 1], steps[i]
        if a.op_name == b.op_name and a.args == b.args:
            consec_dupes += 1

    # Obvious no-ops
    noop_steps = sum(1 for s in steps if _is_noop_step(s))

    return dict(
        base_cost=base_cost,
        num_steps=num_steps,
        unique_ops=float(unique_ops),
        consec_dupes=float(consec_dupes),
        noop_steps=float(noop_steps),
    )


