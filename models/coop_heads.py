# models/coop_heads.py
from __future__ import annotations
import random
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn

from dsl.program import Program, ProgramStep
from dsl.ops import OPS_REGISTRY


# ---------- helpers

def _same_step(a: ProgramStep, b: ProgramStep) -> bool:
    if a.op_name != b.op_name:
        return False
    if set(a.args.keys()) != set(b.args.keys()):
        return False
    for k in a.args.keys():
        if int(a.args[k]) != int(b.args[k]):
            return False
    return True


def _is_noop(op_name: str, args: Dict[str, int]) -> bool:
    # translate(0,0)
    if op_name == "translate":
        return int(args.get("dx", 0)) == 0 and int(args.get("dy", 0)) == 0

    # color_map(a -> a)
    if op_name == "color_map":
        return int(args.get("src_color", -1)) == int(args.get("dst_color", -2))

    # copy where source rect == dest rect (no change)
    if op_name == "copy":
        sx, sy = int(args.get("src_x", -999)), int(args.get("src_y", -999))
        dx, dy = int(args.get("dst_x", -999)), int(args.get("dst_y", -999))
        w, h = int(args.get("w", 0)), int(args.get("h", 0))
        return sx == dx and sy == dy and w > 0 and h > 0

    # paint/erase could be true no-ops only if out of bounds; leave to executor bounds
    return False


def _sigmoid_to_int_range(x: torch.Tensor, lo: int, hi: int) -> int:
    """
    Map raw head value to [lo, hi] using sigmoid (centers around mid-range at init).
    """
    s = torch.sigmoid(x).item()  # [0,1]
    val = lo + s * (hi - lo)
    return int(round(val))


# ---------- modules

class ProposerHead(nn.Module):
    """
    Exploratory proposer that samples an op + args with:
      - temperature sampling for op
      - eps-greedy exploration
      - param heads mapped via sigmoid -> [lo,hi] (avoids all-zeros at init)
      - active rejection of no-ops & exact duplicates of the previous step
    """
    def __init__(self, d_model: int = 256, eps_greedy: float = 0.25, temp: float = 1.3, max_resample: int = 6):
        super().__init__()
        self.eps = float(eps_greedy)
        self.temp = float(temp)
        self.max_resample = int(max_resample)

        self.backbone = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.ReLU(),
        )
        self.op_head = nn.Linear(d_model, len(OPS_REGISTRY))

        # one head per parameter *name* (shared by name across ops)
        param_spaces: Dict[str, Any] = {}
        for spec in OPS_REGISTRY.values():
            for p, space in spec.params.items():
                if p not in param_spaces:
                    param_spaces[p] = space

        mlps = {}
        for pname, space in param_spaces.items():
            if isinstance(space, list):
                mlps[pname] = nn.Linear(d_model, len(space))   # categorical
            elif isinstance(space, tuple) and len(space) == 2:
                mlps[pname] = nn.Linear(d_model, 1)             # integer range
            else:
                mlps[pname] = nn.Linear(d_model, 1)             # fallback
        self.param_mlps = nn.ModuleDict(mlps)

    def _sample_args_from_h(self, h: torch.Tensor, spec) -> Dict[str, int]:
        args: Dict[str, int] = {}
        for pname, space in spec.params.items():
            head = self.param_mlps[pname](h).squeeze(0)
            if isinstance(space, list) and len(space) > 0:
                # categorical
                p = torch.softmax(head, dim=-1)
                choice = torch.multinomial(p, num_samples=1).item()
                args[pname] = int(space[choice])
            elif isinstance(space, tuple) and len(space) == 2:
                lo, hi = int(space[0]), int(space[1])
                args[pname] = _sigmoid_to_int_range(head.squeeze(), lo, hi)
            else:
                args[pname] = 0
        return args

    def _postfix_reject_fixes(self, op_name: str, args: Dict[str, int]) -> Dict[str, int]:
        """
        Small targeted nudges to avoid common no-ops without heavy resampling.
        """
        fixed = dict(args)
        if op_name == "color_map" and fixed["src_color"] == fixed["dst_color"]:
            # move dst one step (wrap) to force change
            fixed["dst_color"] = fixed["dst_color"] + 1
        if op_name == "translate" and fixed["dx"] == 0 and fixed["dy"] == 0:
            fixed["dx"] = 1  # minimal non-zero move
        if op_name == "copy":
            # if src==dst, nudge dst
            if (fixed.get("src_x") == fixed.get("dst_x")) and (fixed.get("src_y") == fixed.get("dst_y")):
                fixed["dst_x"] = fixed["dst_x"] + 1
        return fixed

    def forward(
        self,
        z_pool: torch.Tensor,                 # [B,D]
        prev_program: Optional[Program] = None,
        msg_from_B: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[Program, Dict[str, Any]]:
        assert z_pool.dim() == 2, "z_pool should be [B,D]"
        z_global = z_pool.mean(dim=0, keepdim=True)  # [1,D]
        h = self.backbone(z_global)                  # [1,D]

        logits_op = self.op_head(h).squeeze(0)       # [K]
        K = logits_op.shape[-1]

        # epsilon-greedy vs tempered sampling
        if random.random() < self.eps:
            op_idx = random.randrange(K)
        else:
            probs = torch.softmax(logits_op / max(self.temp, 1e-6), dim=-1)
            op_idx = torch.multinomial(probs, num_samples=1).item()

        op_name = list(OPS_REGISTRY.keys())[op_idx]
        spec = OPS_REGISTRY[op_name]
        prev_last = prev_program.steps[-1] if (prev_program is not None and len(prev_program.steps) > 0) else None

        # resample a few times to avoid dup/no-op
        cand_step = None
        for _ in range(self.max_resample):
            args = self._sample_args_from_h(h, spec)
            args = self._postfix_reject_fixes(op_name, args)
            candidate = ProgramStep(op_name=op_name, args=args)
            if prev_last is not None and _same_step(prev_last, candidate):
                continue
            if _is_noop(op_name, args):
                continue
            cand_step = candidate
            break

        if cand_step is None:
            # give up and take whatever we had last (still avoids crash)
            args = self._sample_args_from_h(h, spec)
            cand_step = ProgramStep(op_name=op_name, args=args)

        new_steps = (prev_program.steps if prev_program is not None else []) + [cand_step]
        prog = Program(steps=new_steps)
        zA = {"logits_op": logits_op}
        return prog, zA


class CriticHead(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, 1)
        )

    def forward(
        self,
        z_pool: torch.Tensor,                 # [B,D]
        program: Program,
        trace: Any
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        z_global = z_pool.mean(dim=0, keepdim=True)
        score = self.f(z_global).squeeze(0)  # scalar tensor
        msg = {"critic_score": score}
        return msg, {"critic_score": score}



