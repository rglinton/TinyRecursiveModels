# evaluators/eval_coop_arc.py
import argparse, json, math, os, sys
from types import SimpleNamespace

import torch
from torch import nn

# Repo-local imports (match your tree)
from puzzle_dataset import PuzzleDataset
from models.coop_heads import ProposerHead, CriticHead
from models.energy_head import EnergyHead
from dsl.executor import Executor
# Use the SAME encoder class that was used for training
from trainers.train_coop import MinimalEncoder


def _tensorize_square_from_seq(seq_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    seq_tensor: [B, L] integers
    returns: [B, H, W] with H*W = L, or raises if not square.
    """
    if not torch.is_tensor(seq_tensor):
        seq_tensor = torch.as_tensor(seq_tensor)
    seq_tensor = seq_tensor.to(device).long()
    if seq_tensor.dim() == 1:
        seq_tensor = seq_tensor.unsqueeze(0)  # [1, L]
    assert seq_tensor.dim() == 2, f"expected [B, L], got {list(seq_tensor.shape)}"
    B, L = seq_tensor.shape
    side = int(round(math.sqrt(L)))
    if side * side != L:
        raise ValueError(f"Cannot reshape L={L} into square grid")
    return seq_tensor.view(B, side, side)


def _iter_puzzle_batches(ds, device, max_batches=None):
    """
    Your PuzzleDataset yields (set_name: str, batch_dict: dict, effective_bs: int|None).
    We do NOT wrap it in a DataLoader — it’s already batched.
    Yields (x:[B,H,W], y:[B,H,W], set_name:str, effective_bs:int|None).
    """
    seen = 0
    for item in ds:
        # Expect exactly 3-tuple
        if not (isinstance(item, (tuple, list)) and len(item) == 3 and isinstance(item[1], dict)):
            # Skip anything malformed
            continue

        set_name, bdict, eff = item
        if "inputs" in bdict and "labels" in bdict:
            x = _tensorize_square_from_seq(bdict["inputs"], device)
            y = _tensorize_square_from_seq(bdict["labels"], device)
        elif "x" in bdict and "y" in bdict:
            x = torch.as_tensor(bdict["x"], device=device).long()
            y = torch.as_tensor(bdict["y"], device=device).long()
        else:
            # No usable keys; skip
            continue

        yield x, y, set_name, eff
        seen += 1
        if max_batches is not None and seen >= max_batches:
            break


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--split", default="train", choices=["train", "val", "test", "all"])
    p.add_argument("--batch_size", type=int, default=32)  # not used (already batched), kept for CLI parity
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--log_json", default=None)
    p.add_argument("--max_batches", type=int, default=5)
    args = p.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"device: {device.type}")

    # ==== Dataset (iterable, already batched) ====

    from puzzle_dataset import PuzzleDatasetConfig  # add near other imports if not present

    cfg = PuzzleDatasetConfig(
        dataset_paths=[args.data_dir],
        seed=0,
        global_batch_size=args.batch_size,
        test_set_mode=False,   # must be a bool
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
    )

    ds = PuzzleDataset(cfg, split=args.split)  # pass the config object (positional), not a string


    # Try to read these if exposed; fall back to sensible defaults
    vocab_size = getattr(ds, "num_colors", 12)
    seq_len = getattr(ds, "seq_len", None)  # may not exist; purely informational
    sets = getattr(ds, "sets", ["all"])
    mean_examples = getattr(ds, "mean_puzzle_examples", None)

    print(f"[eval] split='{args.split}' | seq_len={seq_len} | vocab_size={vocab_size} | sets={sets} | mean_puzzle_examples={mean_examples}")

    # ==== Models (match training) ====
    encoder = MinimalEncoder(num_embeddings=vocab_size, d_model=256, padding_idx=None).to(device)
    A = ProposerHead().to(device)
    B = CriticHead().to(device)
    energy = EnergyHead().to(device)
    execu = Executor()

    # ==== Load checkpoint ====
    ckpt = torch.load(args.ckpt_path, map_location=device)
    print("loaded checkpoint:", args.ckpt_path)
    print("[ckpt] top-level keys:", list(ckpt.keys()))
    for k in ["encoder", "A", "B", "energy"]:
        if k in ckpt:
            sub = ckpt[k]
            # print up to ~8 subkeys for readability
            subkeys = list(sub.keys())
            head = ", ".join(subkeys[:8]) + (" ..." if len(subkeys) > 8 else "")
            print(f"[ckpt:{k}] subkeys: [{head}]")
        else:
            print(f"[ckpt:{k}] MISSING")

    # Strict=False so minor cosmetic diffs don’t crash loading
    encoder.load_state_dict(ckpt.get("encoder", {}), strict=False)
    A.load_state_dict(ckpt.get("A", {}), strict=False)
    B.load_state_dict(ckpt.get("B", {}), strict=False)
    energy.load_state_dict(ckpt.get("energy", {}), strict=False)

    print(f"[models] encoder.embed rows={encoder.embed.weight.shape[0]} (ckpt rows={ckpt.get('encoder',{}).get('embed.weight', torch.empty(0)).shape[0] if 'encoder' in ckpt else 'n/a'})")

    # ==== Iterate a few batches and print real shapes ====
    processed = 0
    for x, y, set_name, eff in _iter_puzzle_batches(ds, device, max_batches=args.max_batches):
        if processed < 3:
            print({
                "set": set_name,
                "effective_batch_size": eff,
                "inputs": {"shape": list(x.shape), "dtype": str(x.dtype), "device": str(x.device)},
                "labels": {"shape": list(y.shape), "dtype": str(y.dtype), "device": str(y.device)},
            })

        # If you want to run a single proposer step + energy:
        # z = encoder(x)  # [B, H, W, d] internally per your MinimalEncoder
        # step, zA = A(z, prev_program=None, msg_from_B=None)
        # _, trace = execu.run(step, x, y, return_trace=True)
        # E = energy(mdl=None, trace=trace, zA=zA, zB=None)

        processed += 1

    print(f"[done] processed {processed} batch(es)")

    # Optional: tiny summary log
    if args.log_json:
        summary = {
            "set": list(sets),
            "effective_batch_size": None,
            "seq_len": seq_len,
            "vocab_size": vocab_size,
            "mean_puzzle_examples": mean_examples,
        }
        os.makedirs(os.path.dirname(args.log_json), exist_ok=True)
        with open(args.log_json, "a") as f:
            f.write(json.dumps(summary) + "\n")
        print(summary)


if __name__ == "__main__":
    main()

