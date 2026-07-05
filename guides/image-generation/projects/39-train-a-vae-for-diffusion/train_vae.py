"""Train the compression VAE and verify it is a good compressor.

Run:
    python train_vae.py            # ~3 min on a multicore CPU

The guide frames this project around CelebA with LPIPS + adversarial losses;
the recorded demo keeps the recipe's structure (reconstruction + tiny KL +
perceptual) at MNIST scale so it trains in minutes. See the README for what
changes at photo scale.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
sys.path.insert(0, str(HERE.parent / "26-cosine-vs-linear-schedule"))
from feature_net import load_feature_net  # noqa: E402

from vae import VAE, vae_loss  # noqa: E402


def padded_mnist_loader(data_dir: str, batch_size: int, train: bool = True):
    tf = transforms.Compose([
        transforms.Pad(2),  # 28 -> 32 so two stride-2 convs give a clean 8x8
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])
    ds = datasets.MNIST(data_dir, train=train, download=True, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=2,
                      drop_last=train)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--kl-weight", type=float, default=1e-4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--out", default=str(HERE / "checkpoints/vae.pt"))
    ap.add_argument("--log", default=str(HERE / "outputs/train_log.csv"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    model = VAE().to(device)
    feature_net = load_feature_net(HERE / "checkpoints/feature_net.pt",
                                   args.data_dir, device)
    for p in feature_net.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    loader = padded_mnist_loader(args.data_dir, args.batch_size)

    def batches():
        while True:
            yield from loader

    it = batches()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    rows = [("step", "loss", "recon_mse", "kl")]
    t0 = time.time()
    for step in range(1, args.steps + 1):
        x, _ = next(it)
        loss, mse, kl = vae_loss(model, feature_net, x.to(device),
                                 kl_weight=args.kl_weight)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            rows.append((step, f"{loss.item():.4f}", f"{mse.item():.4f}",
                         f"{kl.item():.3f}"))
            print(f"step {step:5d}/{args.steps} | loss {loss.item():.4f} "
                  f"| recon {mse.item():.4f} | kl {kl.item():.2f} "
                  f"| {step / (time.time() - t0):.1f} it/s", flush=True)

    # The SD-style scaling factor: 1 / std of the latents, so the diffusion
    # model sees roughly unit-variance inputs (SD's famous 0.18215).
    model.eval()
    with torch.no_grad():
        x_val = next(iter(padded_mnist_loader(args.data_dir, 2048, train=False)))[0]
        z = model.encode(x_val.to(device), sample=False)
        scale_factor = float(1.0 / z.std())
    print(f"latent std {z.std():.3f} -> scale_factor {scale_factor:.4f}")

    torch.save({"model": model.state_dict(), "scale_factor": scale_factor}, args.out)
    with open(args.log, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
