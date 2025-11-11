# evaluators/eval_coop_arc.py
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# ---- Dataset (matches the working path you confirmed) ------------------------
#from puzzle_dataset import PuzzleDataset
from types import SimpleNamespace
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig

# ---- Models (all from THIS repo) ---------------------------------------------
from trainers.train_coop import MinimalEncoder          # <- actual training encoder
from models.coop_heads import ProposerHead, CriticHead
from models.energy_head import EnergyHead
from dsl.executor import Executor

def _unpack_batch(batch, device):
    """
    Robustly extract (x, y, pid) from many possible batch layouts.
    Handles:
      - dict with keys like: x/inputs/X, y/labels/Y/target/targets, pid/puzzle_identifiers/id
      - tuple/list: (x, y, pid) or (x, y)
      - dicts with unknown keys: picks first 2-3 tensor-like values in a stable order
    Moves tensors to device; converts numpy arrays to tensors first.
    """
    import numpy as np
    import torch

    def is_tensor_like(v):
        return isinstance(v, torch.Tensor) or isinstance(v, np.ndarray)

    def to_tensor(v):
        if isinstance(v, np.ndarray):
            return torch.from_numpy(v)
        return v

    x = y = pid = None

    # --- Dict case: try known names first
    if isinstance(batch, dict):
        # Known names
        x = batch.get("x") or batch.get("inputs") or batch.get("input") or batch.get("X")
        y = (batch.get("y") or batch.get("labels") or batch.get("label")
             or batch.get("target") or batch.get("targets") or batch.get("Y"))
        pid = batch.get("pid") or batch.get("puzzle_identifiers") or batch.get("ids") or batch.get("id")

        # Fallback: pick first few tensor-like values in key order if x/y missing
        if x is None or y is None:
            tensorish = []
            for k in batch.keys():
                v = batch[k]
                if is_tensor_like(v):
                    tensorish.append((k, v))
            # Assign in order if present
            if x is None and len(tensorish) >= 1:
                x = tensorish[0][1]
            if y is None and len(tensorish) >= 2:
                y = tensorish[1][1]
            if pid is None and len(tensorish) >= 3:
                pid = tensorish[2][1]

    # --- Tuple/List case
    elif isinstance(batch, (list, tuple)):
        if len(batch) >= 3:
            x, y, pid = batch[0], batch[1], batch[2]
        elif len(batch) == 2:
            x, y = batch[0], batch[1]
        elif len(batch) == 1:
            x = batch[0]
        else:
            raise TypeError(f"Empty batch structure: {batch}")

    else:
        raise TypeError(f"Unrecognized batch type: {type(batch)}")

    # Convert numpy -> torch and move to device
    if x is not None:
        x = to_tensor(x)
        if hasattr(x, "to"):
            x = x.to(device)
    if y is not None:
        y = to_tensor(y)
        if hasattr(y, "to"):
            y = y.to(device)
    if pid is not None:
        pid = to_tensor(pid)
        if hasattr(pid, "to"):
            pid = pid.to(device)

    return x, y, pid

def _unpack_batch_old(batch, device):
    """
    Accepts batches as dicts or tuples/lists and returns (x, y, pid) on the correct device.
    - Dict: tries keys {'x'|'inputs'}, {'y'|'labels'}, {'pid'|'puzzle_identifiers'}.
    - Tuple/List: accepts (x, y, pid) or (x, y).
    """
    import numpy as np
    import torch

    x = y = pid = None

    if isinstance(batch, dict):
        x = batch.get("x", batch.get("inputs"))
        y = batch.get("y", batch.get("labels"))
        pid = batch.get("pid", batch.get("puzzle_identifiers"))
    elif isinstance(batch, (list, tuple)):
        if len(batch) >= 3:
            x, y, pid = batch[0], batch[1], batch[2]
        elif len(batch) == 2:
            x, y = batch[0], batch[1]
        elif len(batch) == 1:
            x = batch[0]
        else:
            raise TypeError(f"Empty batch structure: {batch}")
    else:
        raise TypeError(f"Unrecognized batch type: {type(batch)}")

    # Convert numpy -> torch
    if x is not None and isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if y is not None and isinstance(y, np.ndarray):
        y = torch.from_numpy(y)
    if pid is not None and isinstance(pid, np.ndarray):
        pid = torch.from_numpy(pid)

    # Move to device (if tensors)
    if hasattr(x, "to"):
        x = x.to(device)
    if hasattr(y, "to"):
        y = y.to(device)
    if hasattr(pid, "to"):
        pid = pid.to(device)

    return x, y, pid

def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_ckpt(path: str | Path):
    ckpt = torch.load(path, map_location="cpu")
    print(f"loaded checkpoint: {path}")
    # Pretty-print the keys and a few subkeys
    top = list(ckpt.keys())
    print(f"[ckpt] top-level keys: {top}")
    for k in ("encoder", "A", "B", "energy"):
        if k in ckpt and isinstance(ckpt[k], dict):
            sub = list(ckpt[k].keys())
            if len(sub) > 8:
                show = f"{', '.join(sub[:7])} ..."
            else:
                show = ", ".join(sub)
            print(f"[ckpt:{k}] subkeys: [{show}]")
    return ckpt

def _build_loader(data_dir: str, split: str, batch_size: int):
    """
    Build a DataLoader using the PuzzleDatasetConfig expected by PuzzleDataset.
    This version does NOT rely on cfg.num_colors and infers vocab size from the dataset.
    """
    # Build a minimal config matching your PuzzleDataset signature
    cfg = PuzzleDatasetConfig(
        dataset_paths=[data_dir],
        seed=0,
        global_batch_size=batch_size,
        test_set_mode=False,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
    )

    # Instantiate dataset (your class reads what it needs from cfg)
    ds = PuzzleDataset(cfg, split=split)

    # Loader
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # --- Infer metadata robustly ---
    seq_len = getattr(ds, "seq_len", 900)

    vocab_size = None
    # Try common attributes:
    for attr in ("num_colors", "vocab_size"):
        if hasattr(ds, attr):
            vocab_size = getattr(ds, attr)
            break
    # Try palettes/collections:
    if vocab_size is None:
        for attr in ("palette", "colors", "color_vocab"):
            if hasattr(ds, attr):
                try:
                    vocab_size = len(getattr(ds, attr))
                    break
                except Exception:
                    pass
    # Fallback
    if vocab_size is None:
        vocab_size = 12

    meta = SimpleNamespace(
        seq_len=seq_len,
        vocab_size=vocab_size,
        sets=["all"],
    )
    return dl, meta


def _build_models_from_ckpt(ckpt: dict, device: torch.device):
    # Determine vocab size directly from the checkpoint so we match rows exactly.
    if "encoder" not in ckpt or "embed.weight" not in ckpt["encoder"]:
        raise RuntimeError("Checkpoint missing encoder/embed.weight; cannot size embeddings.")
    num_embeddings = ckpt["encoder"]["embed.weight"].shape[0]

    # MinimalEncoder from the ACTUAL training code
    # d_model=256 matches your training config; padding_idx left None unless present in ckpt buffer names
    encoder = MinimalEncoder(
        num_embeddings=num_embeddings,
        d_model=256,
        padding_idx=None,
    ).to(device)

    proposer = ProposerHead().to(device)
    critic = CriticHead().to(device)
    energy = EnergyHead().to(device)
    execu = Executor()  # pure python helper

    # Load weights (non-strict only if necessary)
    missing, unexpected = encoder.load_state_dict(ckpt["encoder"], strict=False)
    if missing or unexpected:
        print(f"[warn:encoder] missing={missing} unexpected={unexpected}")

    if "A" in ckpt:
        m, u = proposer.load_state_dict(ckpt["A"], strict=False)
        if m or u:
            print(f"[warn:A(ProposerHead)] missing={m} unexpected={u}")
    if "B" in ckpt:
        m, u = critic.load_state_dict(ckpt["B"], strict=False)
        if m or u:
            print(f"[warn:B(CriticHead)] missing={m} unexpected={u}")
    if "energy" in ckpt:
        m, u = energy.load_state_dict(ckpt["energy"], strict=False)
        if m or u:
            print(f"[warn:energy] missing={m} unexpected={u}")

    # Report final sizes for sanity
    emb = encoder.state_dict().get("embed.weight", None)
    emb_rows = emb.shape[0] if emb is not None else None
    print(f"[models] encoder.embed rows={emb_rows} (ckpt rows={num_embeddings})")

    return encoder, proposer, critic, energy, execu


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--log_json", type=str, default=None)
    p.add_argument("--max_batches", type=int, default=5)
    args = p.parse_args()

    device = _device()
    print(f"device: {device.type}")

    # Data
    loader, meta = _build_loader(args.data_dir, args.split, args.batch_size)

    # Checkpoint & models (built from ACTUAL training encoder)
    ckpt = _load_ckpt(args.ckpt_path)
    encoder, proposer, critic, energy, execu = _build_models_from_ckpt(ckpt, device)

    # Iterate a few batches to confirm shapes & readiness (no forward pass)
    processed = 0

    for i, batch in enumerate(loader):
        x, y_true, pid = _unpack_batch(batch, device)

        # Optional: print a tiny summary the first few batches
        if i < 3:
            def shape_of(t):
                try:
                    return list(t.shape)
                except Exception:
                    return None
            print({
                "set": "all",
                "effective_batch_size": (x.shape[0] if hasattr(x, "shape") else None),
                "inputs": {"shape": shape_of(x), "dtype": str(getattr(x, "dtype", None)), "device": str(getattr(x, "device", None))},
                "labels": {"shape": shape_of(y_true), "dtype": str(getattr(y_true, "dtype", None)), "device": str(getattr(y_true, "device", None))},
                "puzzle_identifiers": {"shape": shape_of(pid), "dtype": str(getattr(pid, "dtype", None)), "device": str(getattr(pid, "device", None))},
            })

        # If you’re not actually computing anything yet, just count batches
        processed += (x.shape[0] if hasattr(x, "shape") else 0)

        if args.max_batches is not None and (i + 1) >= args.max_batches:
            break

    print(f"[done] processed {i+1} batch(es); total_effective={processed}")


    # Optional logging
    # ---- Summary / logging (dot access for SimpleNamespace) ----
    seq_len = getattr(meta, "seq_len", None)
    vocab_size = getattr(meta, "vocab_size", None)
    sets = getattr(meta, "sets", None)
    mean_puzzle_examples = getattr(meta, "mean_puzzle_examples", None)

    summary_hdr = {
        "set": sets if sets is not None else "unknown",
        "effective_batch_size": effective_bs if 'effective_bs' in locals() else None,
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "mean_puzzle_examples": mean_puzzle_examples,
    }
    print(summary_hdr)
    

if __name__ == "__main__":
    main()
