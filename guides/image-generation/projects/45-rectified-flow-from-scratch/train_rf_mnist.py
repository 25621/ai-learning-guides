"""Rectified flow on MNIST: project 24's U-Net, retargeted from noise to
velocity. Note what is absent versus phase 5: no schedule, no alpha-bar,
no T — `t` is just a real number in [0, 1] fed to the same time embedding
(scaled by 1000 so the sinusoidal frequencies are well spread).

Run:
    python train_rf_mnist.py           # ~3 min on a multicore CPU
    python sample_rf_mnist.py
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

from rf import rf_loss  # noqa: E402


class VelocityUNet(UNet):
    """Same U-Net; t in [0,1] is scaled into the sinusoidal embedding's
    comfortable range."""

    def forward(self, x, t, **kw):
        return super().forward(x, 1000.0 * t, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--out", default=str(HERE / "checkpoints/rf_mnist.pt"))
    ap.add_argument("--log", default=str(HERE / "outputs/train_log.csv"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    model = VelocityUNet().to(device)
    ema = EMA(model, 0.995)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    batches = infinite(mnist_loader(args.data_dir, args.batch_size))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    rows = [("step", "loss")]
    running, t0 = 0.0, time.time()
    for step in range(1, args.steps + 1):
        x0, _ = next(batches)
        loss = rf_loss(model, x0.to(device))
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

    torch.save({"ema": ema.shadow.state_dict()}, args.out)
    with open(args.log, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
