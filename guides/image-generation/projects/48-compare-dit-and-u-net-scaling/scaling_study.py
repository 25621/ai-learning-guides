"""DiT vs U-Net at three sizes each: quality against training compute.

Protocol: six unconditional models, identical data/optimizer/steps; per-model
we record measured wall-clock per step (the compute axis) and a feature-space
Frechet distance on 256 DDIM-50 samples (the quality axis, project 26's
metric). Every model sees the same 600-step budget, so the plot answers:
"per unit of compute at THIS scale, which backbone buys more quality?"

The honest expectation from the guide — "U-Net is faster per FLOP at small
scales; DiT scales further" — means the U-Net should win here. The point of
the exercise is the protocol, which is exactly how you would detect the
crossover at larger scale.

Run:
    python scaling_study.py            # ~10 min on a multicore CPU
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
for p in ("24-ddpm-on-mnist", "25-ddpm-on-cifar-10", "26-cosine-vs-linear-schedule",
          "27-ddim-sampler", "43-implement-dit-s-2"):
    sys.path.insert(0, str(HERE.parent / p))
import plot_style as ps  # noqa: E402
from ddim import DDIMSampler  # noqa: E402
from diffusion import GaussianDiffusion  # noqa: E402
from dit import DiT  # noqa: E402
from feature_net import load_feature_net  # noqa: E402
from fid import frechet_distance, gaussian_stats  # noqa: E402
from train import infinite, mnist_loader  # noqa: E402
from unet import UNet  # noqa: E402

CONFIGS = [
    ("unet-S", lambda: UNet(base_ch=8)),
    ("unet-M", lambda: UNet(base_ch=16)),
    ("unet-L", lambda: UNet(base_ch=32)),
    ("dit-S", lambda: DiT(dim=64, depth=4, heads=4, num_classes=0)),
    ("dit-M", lambda: DiT(dim=128, depth=6, heads=4, num_classes=0)),
    ("dit-L", lambda: DiT(dim=192, depth=8, heads=6, num_classes=0)),
]


def run_one(name, build, steps, data_dir, seed=0):
    torch.manual_seed(seed)
    model = build()
    n_params = sum(p.numel() for p in model.parameters())
    diffusion = GaussianDiffusion(T=300, schedule="linear")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    batches = infinite(mnist_loader(data_dir, 64))

    t0 = time.time()
    last_losses = []
    for step in range(1, steps + 1):
        x0, _ = next(batches)
        loss = diffusion.loss(model, x0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step > steps - 100:
            last_losses.append(loss.item())
    train_s = time.time() - t0

    # Quality: Frechet distance in classifier feature space, 256 DDIM-50 samples.
    model.eval()
    torch.manual_seed(7)
    x = DDIMSampler(diffusion, num_steps=50).sample(
        model, (256, 1, 28, 28), device="cpu"
    ).clamp(-1, 1)
    net = load_feature_net(HERE / "checkpoints/feature_net.pt", data_dir, "cpu")
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    test = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
    reals = torch.stack([test[i][0] for i in range(2048)])
    with torch.no_grad():
        fd = frechet_distance(*gaussian_stats(net.features(reals)),
                              *gaussian_stats(net.features(x)))
    avg_loss = sum(last_losses) / len(last_losses)
    print(f"{name:7s} | {n_params / 1e6:5.2f}M params | {train_s:5.0f}s train "
          f"| loss {avg_loss:.4f} | FD {fd:6.1f}", flush=True)
    return dict(name=name, params=n_params, train_s=train_s,
                loss=avg_loss, fd=fd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    args = ap.parse_args()

    out_dir = HERE / "outputs"
    out_dir.mkdir(exist_ok=True)
    results = [run_one(name, build, args.steps, args.data_dir)
               for name, build in CONFIGS]

    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)

    fig, ax = ps.new_axes(6.8, 4.6)
    for family, color in (("unet", ps.SERIES[0]), ("dit", ps.SERIES[1])):
        pts = [r for r in results if r["name"].startswith(family)]
        ax.plot([r["train_s"] for r in pts], [r["fd"] for r in pts],
                color=color, linewidth=2, marker="o", markersize=6)
        for r in pts:
            ax.annotate(r["name"], (r["train_s"], r["fd"]),
                        color=ps.INK_SECONDARY, fontsize=9, fontweight="bold",
                        xytext=(6, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ps.finish(fig, ax,
              "Quality vs training compute at toy scale (600 steps each)",
              "measured training wall-clock (s, log)",
              "feature-space Frechet distance (log, lower is better)",
              out_dir / "quality_vs_compute.png")


if __name__ == "__main__":
    main()
