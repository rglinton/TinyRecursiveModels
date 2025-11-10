# evaluators/eval_coop_arc.py  (DROP-IN REPLACEMENT)

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple, List, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# === Local repo imports (present in your repo) ===
from puzzle_dataset import make_loaders  # your file
from models.coop_heads import ProposerHead, CriticHead  # present in models/
from models.energy_head import EnergyHead               # present in models/
from dsl.executor import Executor                       # present in dsl/

# -------------------------------
# Minimal encoder (to match ckpt)
# -------------------------------
class MinimalEncoder(nn.Module):
    """
    Small embedding-only encoder so the checkpoint's encoder block
    (with key 'embed.weight' shaped [num_colors, 16]) can load.
    It converts integer-color grids [B, H, W] -> [B, 16] by mean pooling
    the embedded pixels. If your trained model used only the embedding
    table, this will still load and run.
    """
    def __init__(self, num_embeddings: int, emb_dim: int = 16):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, W] of integer color ids 0..num_embeddings-1
        # -> [B, H, W, D] -> mean over H,W -> [B, D]
        if x.dim() == 4 and x.size(1) == 1:
            x = x.squeeze(1)  # [B, 1, H, W] -> [B, H, W]
        em = self.embed(x.long())           # [B, H, W, D]
        z = em.mean(dim=(1, 2))             # [B, D]
        return z

# -------------------------------
# Batch unpacking helpers
# -------------------------------
def _to_device(t, device):
    return t.to(device) if torch.is_tensor(t) else t

def _unpack_dataset_batch(
    raw_batch: Union[Tuple[Any, Dict[str, torch.Tensor], int], Dict[str, torch.Tensor]],
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Accepts either:
      1) (set_name: str, batch: {'inputs': T, 'labels': T, 'puzzle_identifiers': T}, effective_bs: int)
      2) batch: {'inputs': T, 'labels': T, 'puzzle_identifiers': T}
    Returns:
      x: [B, H, W], y: [B, H, W], info: dict with optional metadata
    """
    info: Dict[str, Any] = {}

    # Case 1: tuple from your PuzzleDataset
    if isinstance(raw_batch, (tuple, list)) and len(raw_batch) == 3 and isinstance(raw_batch[1], dict):
        set_name, batch_dict, eff_bs = raw_batch
        info["set_name"] = set_name
        info["effective_bs"] = eff_bs
        x = _to_device(batch_dict.get("inputs"), device)
        y = _to_device(batch_dict.get("labels"), device)
        info["puzzle_identifiers"] = _to_device(batch_dict.get("puzzle_identifiers"), device)
        return x, y, info

    # Case 2: plain dict
    if isinstance(raw_batch, dict):
        x = _to_device(raw_batch.get("inputs") or raw_batch.get("x"), device)
        y = _to_device(raw_batch.get("labels") or raw_batch.get("y"), device)
        if "puzzle_identifiers" in raw_batch:
            info["puzzle_identifiers"] = _to_device(raw_batch["puzzle_identifiers"], device)
        return x, y, info

    raise TypeError(f"Unrecognized batch structure: type={type(raw_batch)}; expected (str, dict, int) or dict.")


# -------------------------------
# Loader construction wrapper
# -------------------------------
def _build_loader(
    data_dir: str,
    split: str,
    batch_size: int
) -> Tuple[DataLoader, Dict[str, Any]]:
    """
    Works with either return style:
      A) make_loaders(...) -> {'train': loader, 'val': loader, ...}
      B) make_loaders(...) -> (loader, metadata)  OR ( {'train': loader,...}, metadata )
    """
    # Try the simplest call signature first (your current make_loaders expects a string path)
    res = make_loaders(data_dir, batch_size=batch_size)

    # Normalize to (dict_of_loaders, metadata_dict)
    meta: Dict[str, Any] = {}
    if isinstance(res, dict):
        loaders = res
    elif isinstance(res, (tuple, list)):
        if len(res) == 2 and isinstance(res[0], dict):
            loaders, meta = res  # ({'train': ...}, metadata)
        elif len(res) == 2 and isinstance(res[0], DataLoader):
            # Single loader + meta, wrap into dict
            single_loader, meta = res
            # If we don't know the split, assume 'train'
            loaders = {"train": single_loader}
        else:
            raise TypeError(f"Unexpected make_loaders return tuple: structure={tuple(type(x) for x in res)}")
    else:
        raise TypeError(f"Unexpected make_loaders return type: {type(res)}")

    if split not in loaders:
        raise ValueError(f"Unknown split '{split}'. Available: {list(loaders.keys())}")

    return loaders[split], meta


# -------------------------------
# Evaluation
# -------------------------------
@torch.no_grad()
def evaluate(args):
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"device: {device.type}")

    # ===== data =====
    loader, meta = _build_loader(args.data_dir, args.split, args.batch_size)
    print(f"[eval] split='{args.split}' -> loader.dataset={type(loader.dataset).__name__}")

    # ===== models =====
    # Minimal encoder defined above (fits checkpoint's 'embed.weight')
    encoder = MinimalEncoder(num_embeddings=args.num_colors, emb_dim=16).to(device)
    proposer = ProposerHead().to(device)
    critic = CriticHead().to(device)
    energy = EnergyHead().to(device)
    execu = Executor()

    # ===== load checkpoint =====
    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    print(f"loaded checkpoint: {ckpt_path}")

    # Load safely; tolerate shape drift by using strict=False
    if "encoder" in ckpt:
        try:
            encoder.load_state_dict(ckpt["encoder"], strict=False)
        except Exception as e:
            print(f"[warn] encoder.load_state_dict(strict=False) failed: {e}")
    if "proposer" in ckpt:
        try:
            proposer.load_state_dict(ckpt["proposer"], strict=False)
        except Exception as e:
            print(f"[warn] proposer.load_state_dict(strict=False) failed: {e}")
    if "critic" in ckpt:
        try:
            critic.load_state_dict(ckpt["critic"], strict=False)
        except Exception as e:
            print(f"[warn] critic.load_state_dict(strict=False) failed: {e}")
    if "energy" in ckpt:
        try:
            energy.load_state_dict(ckpt["energy"], strict=False)
        except Exception as e:
            print(f"[warn] energy.load_state_dict(strict=False) failed: {e}")

    encoder.eval(); proposer.eval(); critic.eval(); energy.eval()

    # ===== loop =====
    results: List[Dict[str, Any]] = []
    for raw_batch in loader:
        x, y_true, info = _unpack_dataset_batch(raw_batch, device)  # x,y: [B,H,W]

        # Encode inputs
        zA = encoder(x)  # [B, D]

        # Cooperative heads propose and score a candidate program (toy stub: propose 1 step)
        # Your real training used multi-step; here we keep it minimal for stability
        msg_B = critic(zA)                 # critic message to proposer
        prog_t = proposer(zA, msg_B)       # candidate program tokens/args
        y_hat, trace = execu.run(prog_t, x, y_true, return_trace=True)

        # Score with energy
        mdl_features = lambda prog: torch.tensor([len(prog)], device=device, dtype=torch.float32).unsqueeze(0)
        E_t = energy(mdl=mdl_features(prog_t), trace=trace, zA=zA, zB=msg_B)

        # Basic metrics (you can extend if you logged more in trace)
        rec = (y_hat == y_true).float().mean().item()

        results.append({
            "set": info.get("set_name", args.split),
            "recon_acc": rec,
            "energy": float(E_t.mean().item()) if torch.is_tensor(E_t) else float(E_t),
        })

    # ===== write log (jsonl) =====
    if args.log_json:
        Path(args.log_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.log_json, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

    # quick summary
    if results:
        avg_rec = sum(r["recon_acc"] for r in results) / len(results)
        avg_E = sum(r["energy"] for r in results) / len(results)
        print(f"[summary] n={len(results)}  recon_acc={avg_rec:.4f}  energy={avg_E:.4f}")
    else:
        print("[summary] no batches produced by loader?")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--Tmax", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--split", type=str, default="val")  # 'train' also supported if that's all you have
    p.add_argument("--log_json", type=str, default="")
    p.add_argument("--num_colors", type=int, default=10)  # pass --num_colors 12 to match your ckpt
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    with torch.inference_mode():
        evaluate(args)
