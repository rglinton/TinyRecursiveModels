# evaluators/eval_coop_arc.py

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, Union

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Canonical import points from THIS repo
from trainers.train_coop import make_loaders, MinimalEncoder
from models.coop_heads import ProposerHead, CriticHead
from models.energy_head import EnergyHead
from dsl.executor import Executor
from dsl.program import Program


# -----------------------------
# Helpers
# -----------------------------
def _resolve_loader(
    obj: Union[DataLoader, Dict[str, DataLoader], Tuple[DataLoader, ...], list],
    split: str
) -> DataLoader:
    """Accept loaders in multiple shapes and return the DataLoader for the split."""
    if isinstance(obj, DataLoader):
        return obj
    if isinstance(obj, dict):
        if split not in obj:
            raise ValueError(f"Split '{split}' not in loaders dict: {list(obj.keys())}")
        return obj[split]
    if isinstance(obj, (tuple, list)):
        idx_map = {"train": 0, "val": 1, "test": 2}
        if len(obj) < 3:
            # be forgiving: try to map common 2-pack or 1-pack cases
            if split == "train" and len(obj) >= 1:
                return obj[0]
            if split in ("val", "test") and len(obj) >= 2:
                return obj[1]
            raise ValueError(f"Cannot map split '{split}' to loaders of length {len(obj)}.")
        return obj[idx_map[split]]
    raise TypeError(f"Unsupported loaders type: {type(obj)}")

from torch.utils.data._utils.collate import default_collate

def _make_index_fetcher(dataset_obj, meta_obj=None):
    """
    Return a callable fetch(idx) that yields a single sample (x,y,...) or None if not found.
    Tries (in order): dataset itself, common wrapper attributes, then meta methods.
    """
    # 1) If dataset_obj itself is indexable
    if hasattr(dataset_obj, "__getitem__"):
        return lambda idx: dataset_obj[idx]

    # 2) Try common wrapper attributes that hold the real dataset
    for attr in ("dataset", "base", "inner", "ds", "wrapped", "underlying", "source"):
        wrapped = getattr(dataset_obj, attr, None)
        if wrapped is not None and hasattr(wrapped, "__getitem__"):
            return lambda idx, w=wrapped: w[idx]

    # 3) Try meta_obj methods/attributes (fetch/get/getitem) or nested datasets
    if meta_obj is not None:
        # callable fetchers (fetch, get, getitem)
        for attr in ("fetch", "get", "getitem"):
            fn = getattr(meta_obj, attr, None)
            if callable(fn):
                return lambda idx, f=fn: f(idx)

        # nested datasets on meta
        for attr in ("dataset", "ds", "source", "data", "underlying"):
            wrapped = getattr(meta_obj, attr, None)
            if wrapped is not None and hasattr(wrapped, "__getitem__"):
                return lambda idx, w=wrapped: w[idx]

    return None


from torch.utils.data._utils.collate import default_collate

def _unpack_batch(batch, device):
    """
    Supported batch shapes:
      • dict with 'x' and ('y' | 'y_true' | 'target') tensors
      • tuple/list whose first two items are tensors -> (x, y)
    Anything else fails fast with a clear message.
    """
    import torch

    # dict case
    if isinstance(batch, dict):
        x = batch.get("x", None)
        y = batch.get("y", batch.get("y_true", batch.get("target", None)))
        if torch.is_tensor(x) and torch.is_tensor(y):
            return x.to(device), y.to(device)
        raise TypeError(
            "Batch dict does not contain tensor 'x' and 'y'/'y_true'/'target'. "
            f"Got keys: {list(batch.keys())}."
        )

    # tuple/list case
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        a, b = batch[0], batch[1]
        if torch.is_tensor(a) and torch.is_tensor(b):
            return a.to(device), b.to(device)

    # otherwise: print a short diagnostic and fail fast
    bt = type(batch).__name__
    preview = batch if isinstance(batch, (list, tuple)) else [batch]
    types = [type(x).__name__ for x in preview[:4]]
    raise TypeError(
        "Unrecognized batch structure. Expected dict with tensor 'x' and 'y', or a tuple/list of two tensors. "
        f"Got {bt} with element types {types}. This means the eval loader isn't the tensor batch "
        "that training used—hence we force using trainers.train_coop.make_loaders above."
    )

def _exact_recon_acc(y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Per-item exact reconstruction (all cells equal). Returns float tensor [B]."""
    # match shapes; grids are discrete ints in this repo
    eq = (y_hat == y_true)
    # Exact match per item
    per_item = eq.view(eq.shape[0], -1).all(dim=1).float()
    return per_item

@torch.no_grad()
def evaluate(args):
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"device: {device.type}")

    # ===== Data (mirror training exactly) =====
    # We import make_loaders from trainers/train_coop.py because that's what training used.
    try:
        from trainers.train_coop import make_loaders
    except Exception as e:
        raise ImportError(f"Failed to import trainers.train_coop.make_loaders: {e}")

    # IMPORTANT: make_loaders expects a STRING path (not a list)
    res = make_loaders(
        args.data_dir,
        batch_size=args.batch_size
    )
        
    # make_loaders may return either a dict OR a (dict, meta) tuple; handle both
    if isinstance(res, tuple):
        loaders = res[0]
    else:
        loaders = res

    if not isinstance(loaders, dict):
        raise TypeError(
            f"Expected loaders to be a dict of split->DataLoader, got {type(loaders).__name__}."
        )
    if args.split not in loaders:
        raise ValueError(f"Unknown split '{args.split}'. Available: {list(loaders.keys())}")

    loader = loaders[args.split]
    print(f"[eval] split='{args.split}' -> loader.dataset={type(loader.dataset).__name__}")

    # ===== Models (mirror training setup) =====
    # MinimalEncoder in your repo requires num_embeddings; we take it from args.num_colors
    from models.common import MinimalEncoder
    from models.coop_heads import ProposerHead, CriticHead
    from models.energy_head import EnergyHead
    from dsl.executor import Executor

    encoder = MinimalEncoder(num_embeddings=args.num_colors).to(device)
    proposer = ProposerHead().to(device)
    critic = CriticHead().to(device)
    energy = EnergyHead().to(device)
    execu = Executor()

    # ===== Load checkpoint (if provided) =====
    ckpt = None
    if args.ckpt_path and os.path.isfile(args.ckpt_path):
        ckpt = torch.load(args.ckpt_path, map_location=device)
        print(f"loaded checkpoint: {args.ckpt_path}")

        # Be lenient: keys may differ between train/eval code paths
        if "encoder" in ckpt:
            encoder.load_state_dict(ckpt["encoder"], strict=False)
        if "proposer" in ckpt:
            proposer.load_state_dict(ckpt["proposer"], strict=False)
        if "critic" in ckpt:
            critic.load_state_dict(ckpt["critic"], strict=False)
        if "energy" in ckpt:
            energy.load_state_dict(ckpt["energy"], strict=False)

    # ===== Eval loop =====
    encoder.eval(); proposer.eval(); critic.eval(); energy.eval()
    total = 0
    correct = 0

    pbar = tqdm(loader, desc=f"eval[{args.split}]")
    for batch in pbar:
        x, y_true = _unpack_batch(batch, device)  # <— uses the helper below

        # Encode once
        zA = encoder(x)

        # Proposer suggests candidate programs (your proposer head returns tokens/args)
        prog_toks, prog_args = proposer(zA)

        # Execute and score with critic + energy
        y_hat, trace = execu.run((prog_toks, prog_args), x, y_true, return_trace=True)

        # Example “correctness” proxy: exact match on output grid (adjust if you use a different metric)
        is_correct = (y_hat == y_true).all(dim=(-1, -2)).all(dim=1) if y_hat.ndim == y_true.ndim else (y_hat == y_true).all()
        correct += int(is_correct.sum().item())
        total += x.size(0)

        pbar.set_postfix(acc=f"{(correct/total):.3f}", n=total)

    acc = correct / max(1, total)
    print(f"[eval:{args.split}] accuracy={acc:.4f}")

    # ===== Optional logging =====
    if args.log_json:
        os.makedirs(os.path.dirname(args.log_json), exist_ok=True)
        with open(args.log_json, "a") as f:
            f.write(json.dumps({
                "split": args.split,
                "accuracy": float(acc),
                "total": int(total),
                "num_colors": int(args.num_colors),
                "ckpt": args.ckpt_path or ""
            }) + "\n")


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--num_colors", type=int, default=10, help="discrete color embeddings (ARC has 10)")
    p.add_argument("--Tmax", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--ckpt_path", type=str, default="")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--log_json", type=str, default="logs/eval_val.jsonl")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
