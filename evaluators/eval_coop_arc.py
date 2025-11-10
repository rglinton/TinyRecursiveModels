#!/usr/bin/env python3
import argparse
import json
import os
from typing import Optional

import torch

# === repo-local imports (from your repo) ===
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True, help="Path to ARC dataset directory (e.g., data/arc1concept-aug-1000)")
    p.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Dataset split to iterate")
    p.add_argument("--batch_size", type=int, default=32, help="Global batch size (PuzzleDatasetConfig.global_batch_size)")
    p.add_argument("--ckpt_path", type=str, default="", help="Optional path to a checkpoint (.pt) to inspect")
    p.add_argument("--log_json", type=str, default="", help="Optional path to a JSONL log file")
    p.add_argument("--max_batches", type=int, default=5, help="Max batches to iterate before exiting")
    p.add_argument("--seed", type=int, default=0, help="Seed used in PuzzleDatasetConfig")
    return p.parse_args()


def build_config(args) -> PuzzleDatasetConfig:
    """
    Construct the config object your PuzzleDataset actually expects.
    These fields mirror the model in puzzle_dataset.py in your repo.
    """
    # Train mode = shuffle groups, Test mode = iterate examples in order.
    test_mode = (args.split != "train")
    cfg = PuzzleDatasetConfig(
        seed=args.seed,
        dataset_paths=[args.data_dir],      # IMPORTANT: list[str], not a single string
        global_batch_size=args.batch_size,  # the dataset makes local_batch_size = global // num_replicas
        test_set_mode=test_mode,
        epochs_per_iter=1,                  # safe default; iteration-level internal batching
        rank=0,                             # single-process eval
        num_replicas=1                      # single-process eval
    )
    return cfg


def maybe_open_jsonl(path: str) -> Optional[object]:
    if not path:
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, "a", encoding="utf-8")


def main():
    args = parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"device: {device.type}")

    # ---- Build dataset config & dataset (this matches your repo's expected API) ----
    cfg = build_config(args)
    ds = PuzzleDataset(config=cfg, split=args.split)

    # A tiny peek at metadata that exists in your PuzzleDatasetMetadata
    # (seq_len, vocab_size, etc. are defined in your repo; no guessing).
    meta = ds.metadata
    print(
        f"[eval] split='{args.split}' | seq_len={meta.seq_len} | "
        f"vocab_size={meta.vocab_size} | sets={list(meta.sets)} | "
        f"mean_puzzle_examples={meta.mean_puzzle_examples:.2f}"
    )

    # ---- Optionally load & dump checkpoint keys (no model assumptions here) ----
    if args.ckpt_path and os.path.isfile(args.ckpt_path):
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        print(f"loaded checkpoint: {args.ckpt_path}")
        if isinstance(ckpt, dict):
            print(f"[ckpt] top-level keys: {list(ckpt.keys())}")
            for k in ckpt.keys():
                sub = ckpt[k]
                if isinstance(sub, dict):
                    subkeys = list(sub.keys())
                    # show a compact view
                    head = ", ".join(subkeys[:8])
                    tail = "" if len(subkeys) <= 8 else " ..."
                    print(f"[ckpt:{k}] subkeys: [{head}]{tail}")
        else:
            print("[ckpt] unexpected checkpoint structure (not a dict)")

    # ---- Iterate the dataset directly (it already yields ready-made batches) ----
    out = maybe_open_jsonl(args.log_json)
    total_effective = 0
    batches = 0

    try:
        for (set_name, batch, effective_global_bs) in ds:
            # batch is a dict with 'inputs', 'labels', 'puzzle_identifiers' (per your repo)
            x = batch["inputs"].to(device)
            y = batch["labels"].to(device)
            ids = batch["puzzle_identifiers"].to(device)

            info = {
                "set": set_name,
                "effective_batch_size": int(effective_global_bs),
                "inputs": {"shape": list(x.shape), "dtype": str(x.dtype), "device": str(x.device)},
                "labels": {"shape": list(y.shape), "dtype": str(y.dtype), "device": str(y.device)},
                "puzzle_identifiers": {"shape": list(ids.shape), "dtype": str(ids.dtype), "device": str(ids.device)},
            }
            print(info)
            if out is not None:
                out.write(json.dumps(info) + "\n")

            total_effective += int(effective_global_bs)
            batches += 1
            if batches >= args.max_batches:
                break

    finally:
        if out is not None:
            out.close()

    print(f"[done] processed {batches} batch(es); total_effective={total_effective}")


if __name__ == "__main__":
    main()
