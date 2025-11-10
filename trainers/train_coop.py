# trainers/train_coop.py
from __future__ import annotations
import os, json, math, argparse
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dsl.executor import Executor
from dsl.program import mdl_features, Program, ProgramStep
from dsl.ops import OPS_REGISTRY
from models.coop_heads import ProposerHead, CriticHead
from models.energy_head import EnergyHead
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig  # canonical dataset types


def choose_device() -> str:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


class MinimalEncoder(nn.Module):
    def __init__(self, num_embeddings: int, d_model: int = 256, padding_idx: int | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embed = nn.Embedding(num_embeddings, 16, padding_idx=padding_idx if padding_idx is not None else -1)
        self.conv = nn.Sequential(
            nn.Conv2d(16, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64,128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128,256, 3, padding=1), nn.ReLU(),
        )
        self.proj = nn.Linear(256, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=0, max=self.num_embeddings - 1)
        z = self.embed(x.long())                 # [B,H,W,16]
        z = z.permute(0, 3, 1, 2).contiguous()   # [B,16,H,W]
        z = self.conv(z).mean(dim=[2,3])         # [B,256]
        return self.proj(z)                      # [B,D]


def make_loaders(data_paths: str, batch_size: int):
    cfg = PuzzleDatasetConfig(
        seed=0,
        dataset_paths=[data_paths],
        global_batch_size=batch_size,
        test_set_mode=False,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
    )
    ds_train = PuzzleDataset(cfg, split="train")
    try:
        ds_eval = PuzzleDataset(cfg, split="eval")
    except Exception:
        ds_eval = None

    loaders = {"train": DataLoader(ds_train, batch_size=None)}
    if ds_eval is not None:
        loaders["eval"] = DataLoader(ds_eval, batch_size=None)
    metadata = ds_train.metadata
    return loaders, metadata


class CoopTrainer:
    def __init__(self, args, vocab_size: int, pad_id: int | None, log_jsonl: str | None = None):
        self.device = choose_device()
        self.exec = Executor(num_colors=10)
        self.energy = EnergyHead().to(self.device)
        self.step_penalty = float(getattr(args, "step_penalty", 0.01))
        self.energy_margin = float(getattr(args, "energy_margin", 1.0))
        self.Tmax = int(getattr(args, "Tmax", 6))

        self.encoder = MinimalEncoder(num_embeddings=vocab_size, d_model=256, padding_idx=pad_id).to(self.device)
        d_model = 256
        # ↑ slightly higher ε to encourage exploration across steps
        self.A = ProposerHead(d_model=d_model, eps_greedy=0.15).to(self.device)
        self.B = CriticHead(d_model=d_model).to(self.device)

        self.opt = AdamW(
            [
                {"params": self.encoder.parameters(), "lr": 1e-4, "weight_decay": 1e-2},
                {"params": self.A.parameters(),       "lr": 5e-4, "weight_decay": 1e-2},
                {"params": self.B.parameters(),       "lr": 5e-4, "weight_decay": 1e-2},
                {"params": self.energy.parameters(),  "lr": 1e-3, "weight_decay": 1e-2},
            ]
        )

        self.log_jsonl = log_jsonl
        self._fp = open(log_jsonl, "a") if log_jsonl else None

    def _log(self, rec: Dict[str, Any]):
        if self._fp:
            self._fp.write(json.dumps(rec) + "\n")
            self._fp.flush()

    def pooled_latents(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def _extract_xy(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            bdict = batch[1]
        else:
            bdict = batch
        if "x" in bdict and "y" in bdict:
            x = bdict["x"].to(self.device).long()
            y = bdict["y"].to(self.device).long()
            return x, y
        if "inputs" in bdict and "labels" in bdict:
            x_seq = bdict["inputs"].to(self.device).long()
            y_seq = bdict["labels"].to(self.device).long()
            bsz, L = x_seq.shape
            side = int(round(math.sqrt(L)))
            if side * side != L:
                raise ValueError(f"Expected square grid; got L={L}")
            x = x_seq.view(bsz, side, side)
            y = y_seq.view(bsz, side, side)
            return x, y
        raise KeyError("Batch must contain either ('x','y') or ('inputs','labels').")

    def train_epoch(self, loader) -> float:
        self.energy.train(); self.A.train(); self.B.train(); self.encoder.train()
        total_loss = 0.0
        batch_count = 0
        sp = torch.nn.Softplus()
        entropy_coef = 5e-2  # slightly stronger to diversify ops across steps

        for batch in loader:
            x, y_true = self._extract_xy(batch)  # [B,H,W]
            z_pool = self.pooled_latents(x)      # [B,D]

            # --- ALWAYS append to a growing current program ---
            curr_prog: Program = Program(steps=[])
            best_E, best_prog = float("inf"), None
            msg_B = None
            margin_terms = []
            recon_terms = []
            ent_terms = []
            step_cost = 0.0
            steps_used = 0

            for t in range(self.Tmax):
                # Propose next step by APPENDING to curr_prog
                prog_t, zA = self.A(z_pool, prev_program=curr_prog, msg_from_B=msg_B)
                curr_prog = prog_t  # grow unconditionally

                # entropy bonus on op distribution
                ent_bonus = torch.tensor(0.0, device=self.device)
                if isinstance(zA, dict) and "logits_op" in zA:
                    logits = zA["logits_op"].squeeze(0)
                    probs = torch.softmax(logits, dim=-1)
                    ent = -(probs * (probs.clamp_min(1e-9).log())).sum()
                    ent_bonus = -entropy_coef * ent
                    ent_terms.append(ent_bonus)

                # Execute and score
                _, trace = self.exec.run(prog_t, x, y_true, return_trace=True)
                msg_B, zB = self.B(z_pool, prog_t, trace)
                E_t = self.energy(mdl=mdl_features(prog_t), trace=trace, zA=zA, zB=msg_B)

                # Track best
                if E_t.item() < best_E:
                    best_E, best_prog = E_t.item(), prog_t

                # --- K=5 hard negatives: tweak last step of prog_t ---
                def tweak_last_step(p: Program) -> Program:
                    if len(p.steps) == 0:
                        return Program(steps=[])
                    st = p.steps[-1]
                    spec = OPS_REGISTRY[st.op_name]
                    new_args = dict(st.args)
                    tweaked = False
                    for pname in spec.params.keys():
                        val = new_args[pname]
                        rng = spec.params[pname]
                        if isinstance(rng, list) and len(rng) > 1:
                            try:
                                idx = int(val)
                            except Exception:
                                idx = 0
                            idx = (idx + 1) % len(rng)
                            new_args[pname] = rng[idx]
                            tweaked = True
                            break
                        elif isinstance(rng, tuple) and len(rng) == 2:
                            lo, hi = map(int, rng)
                            try:
                                nv = int(val) + 1
                            except Exception:
                                nv = lo
                            new_args[pname] = max(lo, min(hi, nv))
                            tweaked = True
                            break
                    return Program(steps=p.steps[:-1] + [ProgramStep(op_name=st.op_name, args=new_args)]) if tweaked else Program(steps=p.steps[:-1])

                candidates = []
                for _ in range(5):
                    neg_prog_k = tweak_last_step(prog_t)
                    _, neg_trace_k = self.exec.run(neg_prog_k, x, y_true, return_trace=True)
                    E_neg_k = self.energy(mdl=mdl_features(neg_prog_k), trace=neg_trace_k, zA=zA, zB=msg_B)
                    candidates.append((E_neg_k, neg_prog_k, neg_trace_k))
                E_neg, _, _ = max(candidates, key=lambda t3: t3[0].item())

                margin_terms.append(sp(E_t - E_neg + self.energy_margin))
                recon_terms.append(trace.loss_per_pair.mean())
                step_cost += self.step_penalty
                steps_used = t + 1

                if batch_count < 3 and (t == self.Tmax - 1):
                    print("CURR_PROG:", [(s.op_name, s.args) for s in curr_prog.steps])
                    if best_prog is not None:
                        print("BEST_PROG:", [(s.op_name, s.args) for s in best_prog.steps])

            rank_loss = torch.stack(margin_terms).mean()
            recon_loss = torch.stack(recon_terms).mean()
            ent_loss = torch.stack(ent_terms).mean() if len(ent_terms) else torch.tensor(0.0, device=self.device)

            loss = rank_loss + 0.3 * recon_loss + step_cost + ent_loss

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            total_loss += loss.item()
            batch_count += 1

            rec = {
                "phase": "train",
                "loss": float(loss.item()),
                "rank_loss": float(rank_loss.item()),
                "recon": float(recon_loss.item()),
                "best_E": float(best_E),
                "prog_len": int(len(best_prog.steps if best_prog else [])),
                "steps_used": int(steps_used),
            }
            print(rec)
            self._log(rec)

        return total_loss / max(1, batch_count)

    def close(self):
        if self._fp:
            self._fp.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_paths", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--Tmax", type=int, default=6)
    parser.add_argument("--energy_margin", type=float, default=0.5)
    parser.add_argument("--step_penalty", type=float, default=0.01)
    parser.add_argument("--ckpt_out", type=str, default="checkpoints")
    parser.add_argument("--log_json", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.ckpt_out, exist_ok=True)
    loaders, metadata = make_loaders(args.data_paths, batch_size=args.batch_size)
    trainer = CoopTrainer(
        args,
        vocab_size=metadata.vocab_size,
        pad_id=getattr(metadata, "pad_id", None),
        log_jsonl=args.log_json,
    )
    trainer.Tmax = args.Tmax

    for ep in range(args.epochs):
        loss = trainer.train_epoch(loaders["train"])
        print(f"[ep {ep}] coop loss: {loss:.4f}")

        ckpt_path = os.path.join(args.ckpt_out, f"coop_epoch{ep:03d}.pt")
        torch.save(
            {
                "encoder": trainer.encoder.state_dict(),
                "A": trainer.A.state_dict(),
                "B": trainer.B.state_dict(),
                "energy": trainer.energy.state_dict(),
            },
            ckpt_path,
        )
        print(f"saved: {ckpt_path}")

    trainer.close()


if __name__ == "__main__":
    main()


