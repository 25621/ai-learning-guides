"""Train a class-conditional DDPM.

Identical to project 24's training loop except the batch labels are passed to
the model, which folds them into the AdaGN conditioning vector.

Run:
    python train_conditional.py

The demo run uses MNIST so it finishes in minutes on a CPU; pass
--dataset cifar10 to train on CIFAR-10 with labels (the guide's original
target — use a GPU, --T 1000, and far more steps for recognizable classes).
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
from diffusion import GaussianDiffusion  # noqa: E402
from train import EMA, infinite  # noqa: E402

from unet_conditional import ConditionalUNet  # noqa: E402


def make_loader(dataset: str, data_dir: str, batch_size: int):
    if dataset == "mnist":
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
        ds = datasets.MNIST(data_dir, train=True, download=True, transform=tf)
        config = dict(in_ch=1, image_size=28, attn_resolutions=(7,))
    else:
        tf = transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
        )
        ds = datasets.CIFAR10(data_dir, train=True, download=True, transform=tf)
        config = dict(in_ch=3, image_size=32, attn_resolutions=(8,))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    return loader, config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"])
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--schedule", default="linear", choices=["linear", "cosine"])
    ap.add_argument("--ema-decay", type=float, default=0.995)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--out", default=str(HERE / "checkpoints/conditional.pt"))
    ap.add_argument("--log", default=str(HERE / "outputs/train_log.csv"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device

    loader, config = make_loader(args.dataset, args.data_dir, args.batch_size)
    model = ConditionalUNet(num_classes=10, **config).to(device)
    ema = EMA(model, args.ema_decay)
    diffusion = GaussianDiffusion(T=args.T, schedule=args.schedule, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    batches = infinite(loader)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"conditional U-Net parameters: {n_params:,} | dataset: {args.dataset}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    log_rows = [("step", "loss")]
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
            avg = running / 50
            log_rows.append((step, f"{avg:.4f}"))
            rate = step / (time.time() - t0)
            print(f"step {step:5d}/{args.steps} | loss {avg:.4f} | {rate:.2f} it/s", flush=True)
            running = 0.0

    torch.save(
        {
            "ema": ema.shadow.state_dict(),
            "model": model.state_dict(),
            "T": args.T,
            "schedule": args.schedule,
            "config": config,
            "dataset": args.dataset,
        },
        args.out,
    )
    with open(args.log, "w", newline="") as f:
        csv.writer(f).writerows(log_rows)
    print(f"saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
