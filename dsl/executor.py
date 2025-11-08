# dsl/executor.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import torch
from dsl.ops import OPS_REGISTRY, OpSpec
from dsl.program import Program, ProgramStep


@dataclass
class ExecTrace:
    loss_per_pair: torch.Tensor               # [B] mean Hamming per example
    steps_applied: List[Tuple[str, Dict[str, int]]]
    valid: bool


class Executor:
    """
    Simple, schema-driven executor:
    - Applies each ProgramStep by looking up the op in OPS_REGISTRY and calling its fn(grid, **args).
    - Args are taken AS-IS from ProgramStep.args (no legacy names like 'src'/'dst' expected).
    - Works on batched grids: x, y_true are [B,H,W] integer tensors.
    """
    def __init__(self, num_colors: int = 10):
        self.num_colors = num_colors

    def _apply_step(self, grid: torch.Tensor, step: ProgramStep) -> torch.Tensor:
        if step.op_name not in OPS_REGISTRY:
            return grid  # unknown op -> no-op

        spec: OpSpec = OPS_REGISTRY[step.op_name]
        # Ensure all required params exist; if missing, skip gracefully (no-op)
        for p in spec.params.keys():
            if p not in step.args:
                return grid

        # Call the op function. Our ops expect keyword params.
        try:
            out = spec.fn(grid, **{k: int(v) for k, v in step.args.items()})
            if not isinstance(out, torch.Tensor):
                return grid
            if out.shape != grid.shape:
                # keep shapes consistent
                return grid
            return out
        except Exception:
            # if op blows up, treat as no-op for robustness
            return grid

    @torch.no_grad()
    def run(
        self,
        program: Program,
        x: torch.Tensor,
        y_true: torch.Tensor,
        return_trace: bool = False,
    ):
        """
        Execute `program` on input grid `x`.
        x, y_true: [B,H,W] (int tensors)
        Returns: (y_hat, trace) if return_trace else y_hat
        """
        assert x.ndim == 3 and y_true.ndim == 3, "x and y_true must be [B,H,W]"
        grid = x.clone()
        steps_applied: List[Tuple[str, Dict[str, int]]] = []

        # Apply each step
        for st in (program.steps or []):
            grid = self._apply_step(grid, st)
            steps_applied.append((st.op_name, dict(st.args)))

        y_hat = grid
        # mean Hamming distance per example
        loss_per_pair = (y_hat != y_true).float().mean(dim=(1, 2))
        trace = ExecTrace(loss_per_pair=loss_per_pair, steps_applied=steps_applied, valid=True)

        return (y_hat, trace) if return_trace else y_hat
