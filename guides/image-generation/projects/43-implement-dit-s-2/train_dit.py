"""Train the mini-DiT as a class-conditional diffusion model.

The diffusion math is project 24's, byte for byte — only the denoiser
changed from U-Net to transformer. That drop-in swap is the entire point of
the DiT paper: diffusion never cared what predicts the noise.

Run:
    python train_dit.py            # ~4 min on a multicore CPU
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
from diffusion import GaussianDiffusion  # noqa: E402
from train import EMA, infinite, mnist_loader  # noqa: E402

from dit import DIT_MINI, DiT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3500)  # DiT needs more steps than the U-Net; see README
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)  # transformers at this scale take a hotter lr than the U-Net
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--ema-decay", type=float, default=0.995)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--out", default=str(HERE / "checkpoints/dit_mini.pt"))
    ap.add_argument("--log", default=str(HERE / "outputs/train_log.csv"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    model = DiT(**DIT_MINI).to(device)
    print(f"DiT params: {sum(p.numel() for p in model.parameters()):,}")
    ema = EMA(model, args.ema_decay)
    diffusion = GaussianDiffusion(T=args.T, schedule="linear", device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    batches = infinite(mnist_loader(args.data_dir, args.batch_size))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    rows = [("step", "loss")]
    running, t0 = 0.0, time.time()
    for step in range(1, args.steps + 1):
        x0, y = next(batches)
        loss = diffusion.loss(model, x0.to(device), model_kwargs={"y": y.to(device)})
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema.update(model)
        running += loss.item()
        if step % 50 == 0:
            rows.append((step, f"{running / 50:.4f}"))
            print(f"step {step:5d}/{args.steps} | loss {running / 50:.4f} "
                  f"| {step / (time.time() - t0):.2f} it/s", flush=True)
            running = 0.0

    torch.save({"ema": ema.shadow.state_dict(), "T": args.T,
                "schedule": "linear", "config": DIT_MINI}, args.out)
    with open(args.log, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
