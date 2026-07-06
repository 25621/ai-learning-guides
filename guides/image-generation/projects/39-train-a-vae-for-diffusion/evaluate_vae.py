"""Verify the VAE is a good compressor before trusting it with diffusion.

Run (after train_vae.py):
    python evaluate_vae.py
"""

import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
sys.path.insert(0, str(HERE.parent / "26-cosine-vs-linear-schedule"))
import plot_style as ps  # noqa: E402
from feature_net import load_feature_net  # noqa: E402

from train_vae import padded_mnist_loader  # noqa: E402
from vae import VAE  # noqa: E402


def main():
    device = "cpu"
    ckpt = torch.load(HERE / "checkpoints/vae.pt", map_location=device,
                      weights_only=True)
    model = VAE().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    out_dir = HERE / "outputs"
    out_dir.mkdir(exist_ok=True)
    data_dir = str(HERE / "data")

    x, _ = next(iter(padded_mnist_loader(data_dir, 2048, train=False)))
    with torch.no_grad():
        z = model.encode(x, sample=False)
        recon = model.decoder(z)

    # 1) Originals over reconstructions — the eyeball test.
    pairs = torch.cat([x[:16], recon[:16].clamp(-1, 1)])
    save_image(F.interpolate(pairs, scale_factor=2, mode="nearest"),
               out_dir / "reconstructions.png", nrow=16, normalize=True,
               value_range=(-1, 1))
    print(f"wrote {out_dir}/reconstructions.png")

    # 2) Numbers: pixel MSE and feature-space distance, plus the latent stats
    #    the diffusion model will care about.
    net = load_feature_net(HERE / "checkpoints/feature_net.pt", data_dir, device)
    with torch.no_grad():
        mse = F.mse_loss(recon, x).item()
        f_dist = F.mse_loss(net.features(recon[:, :, 2:-2, 2:-2]),
                            net.features(x[:, :, 2:-2, 2:-2])).item()
        agree = (net(recon[:, :, 2:-2, 2:-2]).argmax(1)
                 == net(x[:, :, 2:-2, 2:-2]).argmax(1)).float().mean().item()
    rows = [
        ("metric", "value"),
        ("compression", "32x32x1 -> 8x8x4 (4x fewer numbers)"),
        ("pixel MSE", f"{mse:.5f}"),
        ("feature-space MSE", f"{f_dist:.5f}"),
        ("classifier agreement (orig vs recon)", f"{agree:.3f}"),
        ("latent std (pre-scaling)", f"{z.std():.3f}"),
        ("scale_factor (1/std)", f"{ckpt['scale_factor']:.4f}"),
    ]
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    for r in rows[1:]:
        print(f"{r[0]}: {r[1]}")

    # 3) Per-channel latent histograms — is the space diffusion-friendly?
    # All four channels in one hue: the message is that they coincide, so
    # channel identity is deliberately not encoded.
    fig, ax = ps.new_axes(6.4, 3.8)
    centers = torch.linspace(-4, 4, 60)
    for c in range(4):
        vals = (z[:, c] * ckpt["scale_factor"]).flatten()
        hist = torch.histc(vals, bins=60, min=-4, max=4)
        ax.plot(centers, hist / hist.sum(), linewidth=1.6,
                color=ps.SERIES[0], alpha=0.55)
    gauss = torch.exp(-centers**2 / 2)
    ax.plot(centers, gauss / gauss.sum(), linewidth=1.6, linestyle="--",
            color=ps.INK_MUTED)
    ax.annotate("N(0,1) reference", (2.0, float((gauss / gauss.sum())[45]) + 0.004),
                color=ps.INK_MUTED, fontsize=9)
    ps.finish(fig, ax, "Scaled latent distribution, all 4 channels overlaid",
              "latent value (after scale_factor)", "fraction",
              out_dir / "latent_histograms.png")


if __name__ == "__main__":
    main()
