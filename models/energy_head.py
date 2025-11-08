# models/energy_head.py
from __future__ import annotations
from typing import Dict, Any

import torch
import torch.nn as nn


class EnergyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_recon = nn.Parameter(torch.tensor(1.0))
        self.w_mdl   = nn.Parameter(torch.tensor(0.05))
        self.w_len   = nn.Parameter(torch.tensor(-0.06))  # small reward for adding useful steps
        self.w_div   = nn.Parameter(torch.tensor(-0.08))  # prefer variety of ops
        self.w_dup   = nn.Parameter(torch.tensor(0.10))   # penalize consecutive duplicates
        self.w_noop  = nn.Parameter(torch.tensor(0.30))   # strongly penalize obvious no-ops
        self.w_crit  = nn.Parameter(torch.tensor(-0.02))  # critic can lower energy a bit

    def forward(
        self,
        mdl: Dict[str, Any],
        trace: Any,
        zA: Dict[str, Any] | None = None,
        zB: Dict[str, Any] | None = None
    ) -> torch.Tensor:
        device = trace.loss_per_pair.device if hasattr(trace, "loss_per_pair") else torch.device("cpu")
        recon = trace.loss_per_pair.mean() if hasattr(trace, "loss_per_pair") else torch.tensor(1.0, device=device)

        base_cost   = torch.tensor(float(mdl.get("base_cost", 0.0)), device=device)
        num_steps   = torch.tensor(float(mdl.get("num_steps", 0)), device=device)
        unique_ops  = torch.tensor(float(mdl.get("unique_ops", 0)), device=device)
        consec_dups = torch.tensor(float(mdl.get("consec_dupes", 0)), device=device)
        noop_steps  = torch.tensor(float(mdl.get("noop_steps", 0)), device=device)

        crit = torch.tensor(0.0, device=device)
        if zB and isinstance(zB, dict) and "critic_score" in zB:
            cs = zB["critic_score"]
            if isinstance(cs, torch.Tensor):
                crit = cs.view(-1).mean()

        E = (
            self.w_recon * recon
            + self.w_mdl   * base_cost
            + self.w_len   * num_steps.clamp(max=6.0)
            + self.w_div   * unique_ops
            + self.w_dup   * consec_dups
            + self.w_noop  * noop_steps
            + self.w_crit  * crit
        )
        return E




