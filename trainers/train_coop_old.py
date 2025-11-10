# trainers/train_coop.py
# Minimal, self-contained cooperative trainer (no TRM dependency yet).
# Uses a tiny encoder so you can run immediately; we can swap in TRM later.

import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dsl.executor import Executor
from dsl.program import mdl_features, Program
from models.coop_heads import ProposerHead, CriticHead
from models.energy_head import EnergyHead

# Your repo's dataset entry points (from puzzle_dataset.py)
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig


# ---------- Tiny fallback encoder (replace with TRM later) ----------
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
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,H,W] ints; make sure indices are in [0, num_embeddings-1]
        x = x.clamp(min=0, max=self.num_embeddings - 1)
        z = self.embed(x.long())            # [B,H,W,16]
        z = z.permute(0, 3, 1, 2).contiguous()  # [B,16,H,W]
        z = self.conv(z).mean(dim=[2,3])    # [B,256]
        return self.proj(z)                 # [B,D]

# ---------- Data loaders ----------
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

    # ←—— this is key
    metadata = ds_train.metadata  # has vocab_size, pad_id, ignore_label_id, etc.
    return loaders, metadata


# ---------- Cooperative trainer ----------
class CoopTrainer:
    def __init__(self, args, vocab_size: int, pad_id: int | None):

#        self.device = "cuda" if torch.cuda.is_available() else "cpu"
 
        self.device = (
            "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        self.exec = Executor(num_colors=10)
        self.energy = EnergyHead().to(self.device)
        self.step_penalty = getattr(args, "step_penalty", 0.01)
        self.energy_margin = getattr(args, "energy_margin", 1.0)

        # === Encoder ===
        # v0: MinimalEncoder to unblock training immediately.
        # (Later we will replace encoder with your TRM by editing just two lines.)
        #self.encoder = MinimalEncoder(num_colors=10, d_model=256).to(self.device)
        self.encoder = MinimalEncoder(num_embeddings=vocab_size, d_model=256, padding_idx=pad_id).to(self.device)
        d_model = 256

        # === Heads ===
        self.A = ProposerHead(d_model=d_model).to(self.device)
        self.B = CriticHead(d_model=d_model).to(self.device)

        self.opt = AdamW(
            list(self.energy.parameters()) + list(self.A.parameters()) + list(self.B.parameters()) + list(self.encoder.parameters()),
            lr=1e-4,
            weight_decay=1e-2,
        )

        self.Tmax = getattr(args, "Tmax", 6)

    def pooled_latents(self, x: torch.Tensor) -> torch.Tensor:
        # With MinimalEncoder this already returns [B,D]
        return self.encoder(x)

    def train_epoch(self, loader):
        self.energy.train(); self.A.train(); self.B.train(); self.encoder.train()
        total_loss = 0.0
        batch_count = 0
        
        for batch in loader:
            # Expecting dict with "x" and "y" tensors
            #x, y_true = batch["x"].to(self.device), batch["y"].to(self.device)

            # Dataset yields: (set_name, batch_dict, eff_batch_size)
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                _, bdict, _ = batch[0], batch[1], (batch[2] if len(batch) > 2 else None)
            else:
                bdict = batch  # fallback if it’s already a dict

            x_seq = bdict["inputs"].to(self.device).long()   # [B, L]
            y_seq = bdict["labels"].to(self.device).long()   # [B, L]

            B, L = x_seq.shape
            side = int(round(math.sqrt(L)))
            if side * side != L:
                raise ValueError(f"Expected square grid (L perfect square), got L={L}")

            x = x_seq.view(B, side, side)
            y_true = y_seq.view(B, side, side)
            
            
            z_pool = self.pooled_latents(x)  # [B,D]

            best_E, best_prog = float("inf"), None
            msg_B = None
            margin_terms = []
            step_cost = 0.0

            for t in range(self.Tmax):
                prog_t, zA = self.A(z_pool, prev_program=best_prog, msg_from_B=msg_B)
                y_hat, trace = self.exec.run(prog_t, x, y_true, return_trace=True)
                msg_B, zB = self.B(z_pool, prog_t, trace)
                E_t = self.energy(mdl=mdl_features(prog_t), trace=trace, zA=zA, zB=zB)

                if E_t.item() < best_E:
                    best_E, best_prog = E_t, prog_t

                # simple contrastive negative: shortened program (or empty)
                neg_prog = Program(steps=[]) if len(prog_t.steps) == 0 else Program(steps=prog_t.steps[:-1])
                _, neg_trace = self.exec.run(neg_prog, x, y_true, return_trace=True)
                E_neg = self.energy(mdl=mdl_features(neg_prog), trace=neg_trace, zA=zA, zB=zB)

                margin_terms.append(torch.relu(E_t - E_neg + self.energy_margin))
                step_cost += self.step_penalty

                if best_E.item() < 0.1:
                    break

            loss = torch.stack(margin_terms).mean() + step_cost
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            total_loss += loss.item()
            batch_count += 1
            
        #return total_loss / max(1, len(loader))
        # IterableDataset has no __len__ — average by number of batches seen
        return total_loss / max(1, batch_count)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_paths", type=str, default="data/arc1concept-aug-1000")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--Tmax", type=int, default=6)
    parser.add_argument("--energy_margin", type=float, default=1.0)
    parser.add_argument("--step_penalty", type=float, default=0.01)
    args = parser.parse_args()

    loaders, metadata = make_loaders(args.data_paths, batch_size=args.batch_size)
    trainer = CoopTrainer(args, vocab_size=metadata.vocab_size, pad_id=metadata.pad_id)

    #loaders = make_loaders(args.data_paths, batch_size=args.batch_size)

    #trainer = CoopTrainer(args)
    trainer.Tmax = args.Tmax

    for ep in range(args.epochs):
        loss = trainer.train_epoch(loaders["train"])
        print(f"[ep {ep}] coop loss: {loss:.4f}")


if __name__ == "__main__":
    main()
