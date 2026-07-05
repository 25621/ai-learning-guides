"""Compare the trained VP and VE models: samples, training stability, and a
Frechet distance in MNIST-classifier feature space.

Run (after both train_sde.py runs):
    python compare.py
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
sys.path.insert(0, str(HERE.parent / "25-ddpm-on-cifar-10"))
sys.path.insert(0, str(HERE.parent / "26-cosine-vs-linear-schedule"))
import plot_style as ps  # noqa: E402
from feature_net import load_feature_net  # noqa: E402
from fid import frechet_distance, gaussian_stats  # noqa: E402
from unet import UNet  # noqa: E402

from sde import make_sde, sample  # noqa: E402


def load_model(family: str, device: str):
    ckpt = torch.load(HERE / f"checkpoints/{family}.pt", map_location=device,
                      weights_only=True)
    model = UNet().to(device)
    model.load_state_dict(ckpt["ema"])
    model.eval()
    return model, make_sde(family, T=ckpt["T"], device=device)


def plot_losses(out_path: Path):
    fig, ax = ps.new_axes()
    for family, color in (("vp", ps.SERIES[0]), ("ve", ps.SERIES[1])):
        with open(HERE / f"outputs/train_log_{family}.csv") as f:
            rows = list(csv.DictReader(f))
        steps = [int(r["step"]) for r in rows]
        losses = [float(r["loss"]) for r in rows]
        ax.plot(steps, losses, color=color, linewidth=2)
        ax.annotate(family.upper(), (steps[-1], losses[-1]), color=ps.INK_SECONDARY,
                    fontsize=10, fontweight="bold", xytext=(6, 0),
                    textcoords="offset points")
    ax.set_yscale("log")
    ax.set_xlim(right=ax.get_xlim()[1] * 1.08)
    ps.finish(fig, ax, "Same U-Net, two forward processes — eps-prediction MSE",
              "training step", "loss (log scale)", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256, help="samples per family for the metric")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out_dir = HERE / "outputs"
    out_dir.mkdir(exist_ok=True)
    plot_losses(out_dir / "loss_curves.png")

    device = args.device
    samples = {}
    for family in ("vp", "ve"):
        model, sde = load_model(family, device)
        torch.manual_seed(args.seed)
        x = sample(sde, model, (args.n, 1, 28, 28), device=device).clamp(-1, 1)
        samples[family] = x
        save_image(F.interpolate(x[:64], scale_factor=2, mode="nearest"),
                   out_dir / f"samples_{family}.png", nrow=8, normalize=True,
                   value_range=(-1, 1))
        print(f"{family}: sampled {args.n}", flush=True)

    net = load_feature_net(HERE / "checkpoints/feature_net.pt", args.data_dir, device)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    test = datasets.MNIST(args.data_dir, train=False, download=True, transform=tf)
    reals = torch.stack([test[i][0] for i in range(2048)])

    rows = [("family", "frechet_distance")]
    with torch.no_grad():
        stats_real = gaussian_stats(net.features(reals))
        for family, x in samples.items():
            fd = frechet_distance(*stats_real, *gaussian_stats(net.features(x)))
            rows.append((family, f"{fd:.2f}"))
            print(f"{family}: feature-space Frechet distance = {fd:.2f}")
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)


if __name__ == "__main__":
    main()
