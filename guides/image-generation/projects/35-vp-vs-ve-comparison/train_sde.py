"""Train the SAME U-Net under the VP or VE forward process.

Run twice, changing only --family:
    python train_sde.py --family vp
    python train_sde.py --family ve
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
from train import EMA, infinite, mnist_loader  # noqa: E402
from unet import UNet  # noqa: E402

from sde import make_sde, sde_loss  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["vp", "ve"])
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--ema-decay", type=float, default=0.995)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    sde = make_sde(args.family, T=args.T, device=device)
    model = UNet().to(device)
    ema = EMA(model, args.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    batches = infinite(mnist_loader(args.data_dir, args.batch_size))

    out = HERE / f"checkpoints/{args.family}.pt"
    log = HERE / f"outputs/train_log_{args.family}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    rows = [("step", "loss")]
    running, t0 = 0.0, time.time()

    for step in range(1, args.steps + 1):
        x0, _ = next(batches)
        loss = sde_loss(sde, model, x0.to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema.update(model)
        running += loss.item()
        if step % 50 == 0:
            rows.append((step, f"{running / 50:.4f}"))
            print(f"{args.family} step {step:5d}/{args.steps} | loss {running / 50:.4f} "
                  f"| {step / (time.time() - t0):.2f} it/s", flush=True)
            running = 0.0

    torch.save({"ema": ema.shadow.state_dict(), "T": args.T, "family": args.family}, out)
    with open(log, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
