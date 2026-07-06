"""Pretrain a small unconditional DDPM on MNIST — the frozen base that LoRA
adapts. This is exactly the phase-5 'DDPM on MNIST' model (same U-Net, same
Gaussian diffusion); we just re-run it here so the project is self-contained.

    python train_base.py            # ~3 min on CPU

Reuses ../24-ddpm-on-mnist/{unet,diffusion,train}.py via sys.path.
"""

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
from unet import UNet  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"


def mnist_loader(data_dir, batch_size):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    ds = datasets.MNIST(str(data_dir), train=True, download=True, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)


def main(data_dir=DATA_DIR, steps=1400, T=200, out=None):
    torch.manual_seed(0)
    out = out or HERE / "checkpoints/base_mnist.pt"
    model = UNet()
    ema = EMA(model, 0.995)
    diffusion = GaussianDiffusion(T=T, schedule="linear", device="cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    batches = infinite(mnist_loader(data_dir, 64))
    print(f"base U-Net params: {sum(p.numel() for p in model.parameters()):,}")

    t0 = time.time()
    for step in range(1, steps + 1):
        x0, _ = next(batches)
        loss = diffusion.loss(model, x0)
        opt.zero_grad(); loss.backward(); opt.step(); ema.update(model)
        if step % 100 == 0:
            print(f"step {step}/{steps} | loss {loss.item():.4f} | "
                  f"{step / (time.time() - t0):.1f} it/s", flush=True)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ema": ema.shadow.state_dict(), "T": T,
                "config": dict(in_ch=1, image_size=28)}, out)
    print(f"saved {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--steps", type=int, default=1400)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    main(Path(args.data_dir), args.steps, args.T, args.out)
