#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig

# ---------------------------
# Utilities
# ---------------------------

def _resolve_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def _bool_test_mode(split: str) -> bool:
    # Your dataset exposes a "test mode" flag rather than arbitrary split names.
    # Treat anything not explicitly 'train' as test/val.
    return split.lower() != "train"

def _load_checkpoint(ckpt_path: Optional[str]):
    if not ckpt_path:
        return None
    p = Path(ckpt_path)
    if not p.exists():
        print(f"[warn] checkpoint not found: {ckpt_path}")
        return None
    ckpt = torch.load(str(p), map_location="cpu")
    print(f"loaded checkpoint: {ckpt_path}")
    # Log top-level keys and subkeys to see what we actually saved during train
    if isinstance(ckpt, dict):
        print("[ckpt] top-level keys:", list(ckpt.keys()))
        for k, v in ckpt.items():
            if isinstance(v, dict):
                print(f"[ckpt:{k}] subkeys: {list(v.keys())[:10]}{' ...' if len(v)>10 else ''}")
            elif isinstance(v, (list, tuple)):
                print(f"[ckpt:{k}] list/tuple len={len(v)}")
            else:
                print(f"[ckpt:{k}] type={type(v)}")
    else:
        print(f"[ckpt] unexpected type: {type(ckpt)}")
    return ckpt

def _open_log(log_json: Optional[str]):
    if not log_json:
        return None
    Path(log_json).parent.mkdir(parents=True, exist_ok=True)
    return open(log_json, "w", encoding="utf-8")

# ---------------------------
# Main evaluation (diagnostic)
# ---------------------------

@torch.no_grad()
def evaluate(args):
    device = _resolve_device()
    print(f"device: {device.type}")

    # ---------- Build dataset/loader exactly as your repo defines ----------
    cfg = PuzzleDatasetConfig(
        seed=42,
        dataset_paths=[args.data_dir],          # list[str]
        global_batch_size=int(args.batch_size), # dataset enforces global batch size
        test_set_mode=_bool_test_mode(args.split),
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
    )
    dataset = PuzzleDataset(cfg)

    # IterableDataset: set batch_size=None so DataLoader yields exactly what __iter__ yields
    loader = DataLoader(dataset, batch_size=None)

    # ---------- Optional: peek at checkpoint structure (no model assumptions) ----------
    ckpt = _load_checkpoint(args.ckpt_path)

    # ---------- Iterate a few batches just to prove wiring + shape ----------
    log_f = _open_log(args.log_json)
    n_seen = 0
    n_print = 5  # print first few to keep output readable

    for item in loader:
        # Per your dataset, each item is a triple: (set_name, batch, N_effective)
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            print(f"[warn] unexpected item from dataset (type={type(item)}): {item}")
            continue

        set_name, batch, n_eff = item
        # Respect requested split: dataset emits both train/test over time
        # Only process the one the user asked for
        want_test = _bool_test_mode(args.split)
        is_test = (str(set_name).lower() != "train")
        if want_test != is_test:
            # Skip items from the other split
            continue

        # batch should be a dict with inputs/labels/puzzle_identifiers per your file
        if not isinstance(batch, dict):
            print(f"[warn] batch is not a dict (type={type(batch)}); skipping")
            continue

        x = batch.get("inputs", None)
        y = batch.get("labels", None)
        meta = batch.get("puzzle_identifiers", None)

        # Move tensors to device when applicable; inputs/labels may be tensors or lists
        def to_dev(t):
            return t.to(device) if torch.is_tensor(t) else t

        x = to_dev(x)
        y = to_dev(y)

        # Print concise diagnostics for first few batches
        if n_seen < n_print:
            def shape_of(t):
                if torch.is_tensor(t):
                    return list(t.shape)
                if isinstance(t, (list, tuple)):
                    return f"list_len={len(t)}"
                return type(t).__name__

            print(f"[{args.split}] N_eff={n_eff} x_shape={shape_of(x)} y_shape={shape_of(y)} "
                  f"meta_keys={list(meta.keys()) if isinstance(meta, dict) else type(meta).__name__}")

        # Optionally write a tiny JSONL row per batch (counts only)
        if log_f:
            row = {
                "split": args.split,
                "N_effective": int(n_eff) if isinstance(n_eff, (int, float)) else None,
                "x_is_tensor": bool(torch.is_tensor(x)),
                "y_is_tensor": bool(torch.is_tensor(y)),
            }
            log_f.write(json.dumps(row) + "\n")

        n_seen += 1
        if args.max_batches is not None and n_seen >= args.max_batches:
            break

    if log_f:
        log_f.close()

    print(f"[done] processed {n_seen} batch(es) for split='{args.split}'")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, help="Path to a single ARC-like dataset dir")
    p.add_argument("--split", default="train", choices=["train", "val", "test"],
                   help="Your dataset uses a test mode flag; 'val' aliases 'test'")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--ckpt_path", type=str, default=None)
    p.add_argument("--log_json", type=str, default=None)
    p.add_argument("--max_batches", type=int, default=50,
                   help="For quick diagnostics; set None to run all.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # Alias val->test for this dataset’s notion of test_set_mode
    if args.split == "val":
        args.split = "test"
    evaluate(args)
