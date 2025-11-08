import torch
import json
import argparse
from tqdm import tqdm

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.minimal_encoder import MinimalEncoder
from models.energy_head import EnergyHead
from models.coop_heads import ProposerHead, CriticHead
from dsl.executor import Executor
from dsl.program import Program
from puzzle_dataset import make_loaders



@torch.no_grad()
def evaluate(args):
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"device: {device.type}")

    # ===== Data =====
    loaders = make_loaders([args.data_dir], batch_size=args.batch_size, num_workers=0)
    if args.split not in loaders:
        raise ValueError(f"Unknown split '{args.split}'. Available: {list(loaders.keys())}")
    loader = loaders[args.split]

    # ===== Models (mirror training setup) =====
    encoder = MinimalEncoder().to(device)
    proposer = ProposerHead().to(device)
    critic = CriticHead().to(device)
    energy = EnergyHead().to(device)
    execu = Executor()

    # ===== Load checkpoint =====
    ckpt = torch.load(args.ckpt_path, map_location=device)
    missing, unexpected = encoder.load_state_dict(ckpt.get("encoder", {}), strict=False)
    print(f"encoder.load_state_dict -> missing: {missing} unexpected: {unexpected}")
    missing, unexpected = proposer.load_state_dict(ckpt.get("proposer", {}), strict=False)
    print(f"proposer.load_state_dict -> missing: {missing} unexpected: {unexpected}")
    missing, unexpected = critic.load_state_dict(ckpt.get("critic", {}), strict=False)
    print(f"critic.load_state_dict -> missing: {missing} unexpected: {unexpected}")
    missing, unexpected = energy.load_state_dict(ckpt.get("energy", {}), strict=False)
    print(f"energy.load_state_dict -> missing: {missing} unexpected: {unexpected}")

    print(f"loaded checkpoint: {args.ckpt_path}")

    # ===== Evaluation =====
    energy.eval()
    proposer.eval()
    critic.eval()
    encoder.eval()

    total = 0
    correct = 0
    best_Es = []
    prog_lens = []
    steps_used = []

    for batch_idx, batch in enumerate(tqdm(loader, desc=f"eval:{args.split}")):
        x, y_true = batch["input"].to(device), batch["output"].to(device)
        zA = encoder(x)
        zB = encoder(y_true)

        prog = Program()
        best_prog = None
        best_E = float("inf")

        for t in range(args.Tmax):
            proposal = proposer(zA)
            trace = execu.run(proposal, x, y_true, return_trace=True)
            E = energy(mdl=proposal, trace=trace, zA=zA, zB=zB)
            if E < best_E:
                best_E = E
                best_prog = proposal

        is_correct = execu.compare(best_prog, y_true)
        total += 1
        correct += int(is_correct)
        best_Es.append(best_E)
        prog_lens.append(len(best_prog))
        steps_used.append(args.Tmax)

        print({
            "phase": f"eval:{args.split}",
            "batch_idx": batch_idx,
            "best_E": best_E,
            "prog_len": len(best_prog),
            "steps_used": args.Tmax,
            "pass_at_1_batch": correct / total,
        })

    summary = {
        "split": args.split,
        "Tmax": args.Tmax,
        "Pass@1": correct / total if total > 0 else 0.0,
        "correct": correct,
        "total": total,
        "mean_bestE": float(torch.tensor(best_Es).mean()) if best_Es else 0.0,
        "mean_prog_len": float(torch.tensor(prog_lens).float().mean()) if prog_lens else 0.0,
        "mean_steps_used": float(torch.tensor(steps_used).float().mean()) if steps_used else 0.0,
    }

    print("summary:", summary)
    if args.log_json:
        with open(args.log_json, "a") as f:
            f.write(json.dumps(summary) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--Tmax", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--log_json", type=str, default=None)
    args = parser.parse_args()

    evaluate(args)


if __name__ == "__main__":
    main()
