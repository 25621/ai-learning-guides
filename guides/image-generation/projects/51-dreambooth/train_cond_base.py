"""Pretrain a small class-conditional DDPM on MNIST (labels 0-9). This is the
phase-5 'class-conditional DDPM' model, re-run here as the frozen base that the
personalization projects adapt. Projects 52/55/56 import `train` from here so
they all share one base recipe.

    python train_cond_base.py       # ~3 min on CPU
"""

import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
sys.path.insert(0, str(HERE.parent / "28-class-conditional-ddpm"))

from diffusion import GaussianDiffusion  # noqa: E402
from train import EMA, infinite  # noqa: E402
from unet_conditional import ConditionalUNet  # noqa: E402

DATA_DIR = HERE / "data"


def mnist_labeled(data_dir, batch_size):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    ds = datasets.MNIST(str(data_dir), train=True, download=True, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)


def train(data_dir, out, steps=1600, T=200, num_classes=10, seed=0):
    torch.manual_seed(seed)
    config = dict(in_ch=1, image_size=28, attn_resolutions=(7,))
    model = ConditionalUNet(num_classes=num_classes, **config)
    ema = EMA(model, 0.995)
    diffusion = GaussianDiffusion(T=T, schedule="linear", device="cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    batches = infinite(mnist_labeled(data_dir, 64))
    print(f"cond U-Net params: {sum(p.numel() for p in model.parameters()):,}")

    t0 = time.time()
    for step in range(1, steps + 1):
        x0, y = next(batches)
        loss = diffusion.loss(model, x0, model_kwargs={"y": y})
        opt.zero_grad(); loss.backward(); opt.step(); ema.update(model)
        if step % 100 == 0:
            print(f"step {step}/{steps} | loss {loss.item():.4f} | "
                  f"{step / (time.time() - t0):.1f} it/s", flush=True)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ema": ema.shadow.state_dict(), "T": T, "config": config,
                "num_classes": num_classes}, out)
    print(f"saved {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--out", default=str(HERE / "checkpoints/cond_base.pt"))
    ap.add_argument("--steps", type=int, default=1600)
    args = ap.parse_args()
    train(Path(args.data_dir), args.out, args.steps)
