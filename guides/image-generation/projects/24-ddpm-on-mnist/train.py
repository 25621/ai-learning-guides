"""Train a small DDPM on MNIST.

Run:
    python train.py --out checkpoints/mnist_linear.pt

The defaults (a ~300k-param U-Net, T = 300, 1000 optimizer steps) are sized
so training finishes in a couple of minutes on a multicore CPU. For the
paper-style setting pass --T 1000 --steps 4000 (and ideally --device cuda);
quality improves steadily with budget.
"""

import argparse
import copy
import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from diffusion import GaussianDiffusion
from unet import UNet


class EMA:
    """Exponential moving average of model weights; sampling from the EMA
    copy is noticeably better than from the raw weights, especially on
    short runs."""

    def __init__(self, model, decay: float = 0.995):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.lerp_(p, 1.0 - self.decay)
        for s, b in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(b)


def mnist_loader(data_dir: str, batch_size: int) -> DataLoader:
    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]  # [0,1] -> [-1,1]
    )
    ds = datasets.MNIST(data_dir, train=True, download=True, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)


def infinite(loader):
    while True:
        yield from loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--schedule", default="linear", choices=["linear", "cosine"])
    ap.add_argument("--ema-decay", type=float, default=0.995)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="checkpoints/mnist_linear.pt")
    ap.add_argument("--log", default="outputs/train_log.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device

    model = UNet().to(device)
    ema = EMA(model, args.ema_decay)
    diffusion = GaussianDiffusion(T=args.T, schedule=args.schedule, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    batches = infinite(mnist_loader(args.data_dir, args.batch_size))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"U-Net parameters: {n_params:,} | schedule: {args.schedule} | device: {device}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    log_rows = [("step", "loss")]
    running, t0 = 0.0, time.time()

    for step in range(1, args.steps + 1):
        x0, _ = next(batches)
        loss = diffusion.loss(model, x0.to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema.update(model)

        running += loss.item()
        if step % 50 == 0:
            avg = running / 50
            log_rows.append((step, f"{avg:.4f}"))
            rate = step / (time.time() - t0)
            print(f"step {step:5d}/{args.steps} | loss {avg:.4f} | {rate:.2f} it/s", flush=True)
            running = 0.0

    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.shadow.state_dict(),
            "T": args.T,
            "schedule": args.schedule,
            "steps": args.steps,
        },
        args.out,
    )
    with open(args.log, "w", newline="") as f:
        csv.writer(f).writerows(log_rows)
    print(f"saved checkpoint to {args.out} and loss log to {args.log}")


if __name__ == "__main__":
    main()
