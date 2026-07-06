"""Sample the latent DDPM, decode through the frozen VAE, and measure the
latent-vs-pixel speed advantage directly.

Run (after train_latent_ddpm.py):
    python sample_latent.py
"""

import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
sys.path.insert(0, str(HERE.parent / "39-train-a-vae-for-diffusion"))
from diffusion import GaussianDiffusion  # noqa: E402
from unet import UNet  # noqa: E402

from train_latent_ddpm import LATENT_UNET, load_frozen_vae  # noqa: E402


def bench_step(model, shape, reps=20):
    """Median forward+backward time per training step, in milliseconds."""
    x = torch.randn(*shape)
    t = torch.randint(0, 300, (shape[0],))
    times = []
    for _ in range(reps):
        t0 = time.time()
        model(x, t).square().mean().backward()
        model.zero_grad(set_to_none=True)
        times.append(time.time() - t0)
    return sorted(times)[len(times) // 2] * 1000


def main():
    device = "cpu"
    ckpt = torch.load(HERE / "checkpoints/latent_ddpm.pt", map_location=device,
                      weights_only=True)
    model = UNet(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["ema"])
    model.eval()
    diffusion = GaussianDiffusion(T=ckpt["T"], schedule=ckpt["schedule"], device=device)
    vae, _ = load_frozen_vae(device)

    out_dir = HERE / "outputs"
    out_dir.mkdir(exist_ok=True)

    # --- sample in latent space, decode once at the end ---
    torch.manual_seed(7)
    t0 = time.time()
    z, _ = diffusion.p_sample_loop(model, (64, 4, 8, 8), device=device)
    latent_sample_s = time.time() - t0
    with torch.no_grad():
        x = vae.decoder(z / ckpt["scale_factor"]).clamp(-1, 1)
    decode_s = time.time() - t0 - latent_sample_s
    save_image(F.interpolate(x, scale_factor=2, mode="nearest"),
               out_dir / "samples.png", nrow=8, normalize=True, value_range=(-1, 1))
    print(f"sampled 64 in {latent_sample_s:.1f}s (+{decode_s:.1f}s decode)")

    # --- the whole point, measured: latent vs pixel cost ---
    pixel_model = UNet()  # project 24's default 28x28 model
    rows = [("quantity", "pixel-space (28x28x1)", "latent-space (8x8x4)")]
    ms_pix = bench_step(pixel_model, (64, 1, 28, 28))
    ms_lat = bench_step(UNet(**LATENT_UNET), (64, 4, 8, 8))
    rows.append(("dimensions per example", "784", "256 (3.1x fewer)"))
    rows.append(("train step, batch 64 (median ms)",
                 f"{ms_pix:.0f}", f"{ms_lat:.0f} ({ms_pix / ms_lat:.1f}x faster)"))
    t0 = time.time()
    diffusion.p_sample_loop(model, (8, 4, 8, 8), device=device)
    lat8 = time.time() - t0
    rows.append(("300-step sampling, batch 8 (s)", "see project 24", f"{lat8:.1f}"))
    with open(out_dir / "speed_comparison.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    for r in rows[1:]:
        print(f"{r[0]}: pixel {r[1]} | latent {r[2]}")

    # --- the reconstruction ceiling: the best the LDM could ever look ---
    from train_vae import padded_mnist_loader  # noqa: E402

    x_real, _ = next(iter(padded_mnist_loader(str(HERE / "data"), 16, train=False)))
    with torch.no_grad():
        ceiling = vae.decoder(vae.encode(x_real, sample=False)).clamp(-1, 1)
    save_image(F.interpolate(torch.cat([x_real, ceiling]), scale_factor=2, mode="nearest"),
               out_dir / "vae_ceiling.png", nrow=16, normalize=True, value_range=(-1, 1))
    print(f"wrote {out_dir}/samples.png, speed_comparison.csv, vae_ceiling.png")


if __name__ == "__main__":
    main()
