"""Train a DDPM on CIFAR-10 (32x32 RGB).

Reuses the U-Net and diffusion code from project 24 — only the data, the input
channels, and the model size change. Two configurations:

- default flags: a small model and short schedule for a CPU "smoke run" that
  verifies the whole pipeline in a few minutes and shows what early training
  looks like
- --full: the paper-scale recipe (used on a GPU) that reaches the FID-below-20
  milestone: wider U-Net, 2 res blocks per level, T = 1000, hundreds of
  thousands of steps

Run (smoke):
    python train_cifar.py
Run (GPU, full):
    python train_cifar.py --full --device cuda --T 1000 --steps 300000
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "24-ddpm-on-mnist"))
from diffusion import GaussianDiffusion  # noqa: E402
from train import EMA, infinite  # noqa: E402
from unet import UNet  # noqa: E402


class CIFAR10Npz(Dataset):
    """CIFAR-10 from a plain .npz file: images (N, 32, 32, 3) uint8, labels (N,).

    The official mirror (cs.toronto.edu) is slow or unreachable from some
    networks. Export the dataset to npz from any source you can reach and
    drop it in the data dir; the loader below prefers it when present.
    """

    def __init__(self, npz_path: Path, transform):
        arr = np.load(npz_path)
        self.images, self.labels = arr["images"], arr["labels"]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.transform(Image.fromarray(self.images[i])), int(self.labels[i])


def cifar_dataset(data_dir: str, train: bool, transform) -> Dataset:
    npz = Path(data_dir) / f"cifar10_{'train' if train else 'test'}.npz"
    if npz.exists():
        return CIFAR10Npz(npz, transform)
    return datasets.CIFAR10(data_dir, train=train, download=True, transform=transform)


def cifar_loader(data_dir: str, batch_size: int) -> DataLoader:
    tf = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),  # the one augmentation DDPM uses
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ]
    )
    ds = cifar_dataset(data_dir, train=True, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)


def build_model(full: bool) -> tuple[UNet, dict]:
    if full:
        # Ho et al. 2020 scale: ~35M params. Needs a GPU and O(100k) steps.
        config = dict(
            in_ch=3, base_ch=128, ch_mults=(1, 2, 2, 2), num_res_blocks=2,
            attn_resolutions=(16,), image_size=32,
        )
    else:
        # Smoke-run scale: same shape, ~0.3M params, minutes on a CPU.
        config = dict(
            in_ch=3, base_ch=16, ch_mults=(1, 2, 2), num_res_blocks=1,
            attn_resolutions=(8,), image_size=32,
        )
    return UNet(**config), config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)  # ~6 min on a multicore CPU
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--schedule", default="linear", choices=["linear", "cosine"])
    ap.add_argument("--ema-decay", type=float, default=0.995)
    ap.add_argument("--full", action="store_true", help="paper-scale model")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="checkpoints/cifar_smoke.pt")
    ap.add_argument("--log", default="outputs/train_log.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device

    model, config = build_model(args.full)
    model = model.to(device)
    ema = EMA(model, args.ema_decay)
    diffusion = GaussianDiffusion(T=args.T, schedule=args.schedule, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    batches = infinite(cifar_loader(args.data_dir, args.batch_size))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"U-Net parameters: {n_params:,} | config: {config} | device: {device}")

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
            print(f"step {step:6d}/{args.steps} | loss {avg:.4f} | {rate:.2f} it/s", flush=True)
            running = 0.0

    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.shadow.state_dict(),
            "T": args.T,
            "schedule": args.schedule,
            "steps": args.steps,
            "config": config,
        },
        args.out,
    )
    with open(args.log, "w", newline="") as f:
        csv.writer(f).writerows(log_rows)
    print(f"saved checkpoint to {args.out} and loss log to {args.log}")


if __name__ == "__main__":
    main()
