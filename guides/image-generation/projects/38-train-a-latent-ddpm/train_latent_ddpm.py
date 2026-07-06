"""Train a DDPM in the frozen VAE's latent space — a miniature Stable Diffusion.

Pipeline, exactly as in the LDM paper:
1. Freeze project 39's VAE.
2. Encode the whole dataset ONCE into 8x8x4 latents (cached to disk —
   this is also how production LDM training works).
3. Multiply by the scale factor so latents are ~unit variance
   (SD's 0.18215 moment).
4. Train project 24's U-Net on the latents as if they were images.

Run (after project 39's train_vae.py):
    python train_latent_ddpm.py            # ~1 min of training on CPU
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
sys.path.insert(0, str(HERE.parent / "39-train-a-vae-for-diffusion"))
from diffusion import GaussianDiffusion  # noqa: E402
from train import EMA  # noqa: E402
from train_vae import padded_mnist_loader  # noqa: E402
from unet import UNet  # noqa: E402
from vae import VAE  # noqa: E402

LATENT_UNET = dict(in_ch=4, base_ch=16, ch_mults=(1, 2), num_res_blocks=1,
                   attn_resolutions=(4,), image_size=8)


def load_frozen_vae(device: str):
    ckpt = torch.load(HERE.parent / "39-train-a-vae-for-diffusion/checkpoints/vae.pt",
                      map_location=device, weights_only=True)
    vae = VAE().to(device)
    vae.load_state_dict(ckpt["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae, ckpt["scale_factor"]


@torch.no_grad()
def encode_dataset(vae, scale_factor, data_dir: str, cache: Path, device: str):
    """One pass over MNIST -> a (60000, 4, 8, 8) tensor of scaled latents."""
    if cache.exists():
        return torch.load(cache, weights_only=True)
    zs = []
    for x, _ in padded_mnist_loader(data_dir, 512):
        zs.append(vae.encode(x.to(device), sample=False) * scale_factor)
    z = torch.cat(zs)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(z, cache)
    print(f"cached {tuple(z.shape)} latents (std {z.std():.3f}) to {cache}")
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--out", default=str(HERE / "checkpoints/latent_ddpm.pt"))
    ap.add_argument("--log", default=str(HERE / "outputs/train_log.csv"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    vae, scale_factor = load_frozen_vae(device)
    latents = encode_dataset(vae, scale_factor, args.data_dir,
                             HERE / "checkpoints/latents.pt", device)

    model = UNet(**LATENT_UNET).to(device)
    print(f"latent U-Net parameters: {sum(p.numel() for p in model.parameters()):,}")
    ema = EMA(model, 0.995)
    diffusion = GaussianDiffusion(T=args.T, schedule="linear", device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    rows = [("step", "loss")]
    running, t0 = 0.0, time.time()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, latents.size(0), (args.batch_size,))
        loss = diffusion.loss(model, latents[idx].to(device))
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

    elapsed = time.time() - t0
    torch.save({"ema": ema.shadow.state_dict(), "T": args.T,
                "schedule": "linear", "config": LATENT_UNET,
                "scale_factor": scale_factor, "train_seconds": elapsed}, args.out)
    with open(args.log, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"saved {args.out} ({elapsed:.0f}s of training)")


if __name__ == "__main__":
    main()
