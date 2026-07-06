"""Train the EDM-parameterized model on MNIST.

Same U-Net, same data, same budget as the DDPM runs — only the bookkeeping
changed: sigma-space instead of timesteps, preconditioning instead of raw
eps-prediction, log-normal sigma sampling instead of uniform t.

Run:
    python train_edm.py            # ~3 min on a multicore CPU

The guide frames this project around your CIFAR-10 model; the recorded demo
runs MNIST for CPU speed — swap the loader (project 25) and it is the same
code.
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

from edm import EDMDenoiser, edm_loss  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ema-decay", type=float, default=0.995)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--out", default=str(HERE / "checkpoints/edm_mnist.pt"))
    ap.add_argument("--sigma-log", default=str(HERE / "outputs/loss_by_sigma.csv"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    model = EDMDenoiser().to(device)
    ema = EMA(model, args.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    batches = infinite(mnist_loader(args.data_dir, args.batch_size))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.sigma_log).parent.mkdir(parents=True, exist_ok=True)
    sigma_rows = [("step", "sigma", "unweighted_mse")]
    t0 = time.time()

    for step in range(1, args.steps + 1):
        x0, _ = next(batches)
        loss, sigma, per_sample = edm_loss(model, x0.to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema.update(model)

        # Record (sigma, raw MSE) pairs from the tail of training for the
        # loss-balance figure.
        if step > args.steps // 2:
            for s, l in zip(sigma[:8].tolist(), per_sample[:8].tolist()):
                sigma_rows.append((step, f"{s:.4f}", f"{l:.5f}"))
        if step % 50 == 0:
            print(f"step {step:5d}/{args.steps} | weighted loss {loss.item():.4f} "
                  f"| {step / (time.time() - t0):.2f} it/s", flush=True)

    torch.save({"ema": ema.shadow.state_dict()}, args.out)
    with open(args.sigma_log, "w", newline="") as f:
        csv.writer(f).writerows(sigma_rows)
    print(f"saved {args.out} and {args.sigma_log}")


if __name__ == "__main__":
    main()
