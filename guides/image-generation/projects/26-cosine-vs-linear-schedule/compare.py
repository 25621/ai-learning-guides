"""Compare two otherwise-identical DDPMs trained with linear vs cosine noise
schedules: schedule shape, training loss, samples, and a Frechet distance in
MNIST-classifier feature space.

Prerequisites:
    ../24-ddpm-on-mnist/checkpoints/mnist_linear.pt   (project 24's run)
    checkpoints/mnist_cosine.pt                       (train with
        python ../24-ddpm-on-mnist/train.py --schedule cosine --out checkpoints/mnist_cosine.pt)

Run:
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
import plot_style as ps  # noqa: E402
from diffusion import GaussianDiffusion  # noqa: E402
from fid import frechet_distance, gaussian_stats  # noqa: E402
from unet import UNet  # noqa: E402

from feature_net import load_feature_net  # noqa: E402


def plot_alpha_bar(out_path: Path, T: int = 300):
    lin = GaussianDiffusion(T=T, schedule="linear")
    cos = GaussianDiffusion(T=T, schedule="cosine")
    fig, ax = ps.new_axes()
    ax.plot(lin.alpha_bar, color=ps.SERIES[0], linewidth=2)
    ax.plot(cos.alpha_bar, color=ps.SERIES[1], linewidth=2)
    # direct labels next to each curve, in text ink (the line carries the color)
    mid = int(T * 0.62)
    ax.annotate("linear", (mid, lin.alpha_bar[mid].item() + 0.04),
                color=ps.INK_SECONDARY, fontsize=10, fontweight="bold")
    ax.annotate("cosine", (mid, cos.alpha_bar[mid].item() + 0.04),
                color=ps.INK_SECONDARY, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.02)
    ps.finish(fig, ax, "How much signal survives at step t  (alpha-bar)",
              "diffusion step t", "alpha-bar (fraction of x0 variance kept)", out_path)


def plot_losses(linear_log: Path, cosine_log: Path, out_path: Path):
    def read(p):
        with open(p) as f:
            rows = list(csv.DictReader(f))
        return [int(r["step"]) for r in rows], [float(r["loss"]) for r in rows]

    fig, ax = ps.new_axes()
    for (log, label, color) in ((linear_log, "linear", ps.SERIES[0]),
                                (cosine_log, "cosine", ps.SERIES[1])):
        steps, losses = read(log)
        ax.plot(steps, losses, color=color, linewidth=2)
        ax.annotate(label, (steps[-1], losses[-1]), color=ps.INK_SECONDARY,
                    fontsize=10, fontweight="bold", xytext=(6, 0),
                    textcoords="offset points")
    ax.set_yscale("log")
    ax.set_xlim(right=ax.get_xlim()[1] * 1.08)  # room for the end labels
    ps.finish(fig, ax, "Training loss, linear vs cosine schedule (log scale)",
              "training step", "noise-prediction MSE", out_path)


@torch.no_grad()
def sample_from(ckpt_path: Path, n: int, batch: int, device: str, seed: int):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = UNet().to(device)
    model.load_state_dict(ckpt["ema"])
    model.eval()
    diffusion = GaussianDiffusion(T=ckpt["T"], schedule=ckpt["schedule"], device=device)
    torch.manual_seed(seed)
    out = []
    for i in range(0, n, batch):
        x0, _ = diffusion.p_sample_loop(model, (min(batch, n - i), 1, 28, 28), device=device)
        out.append(x0.clamp(-1, 1))
        print(f"  {ckpt['schedule']}: sampled {i + x0.size(0)}/{n}", flush=True)
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    p24 = HERE.parent / "24-ddpm-on-mnist"
    ap.add_argument("--linear-ckpt", default=str(p24 / "checkpoints/mnist_linear.pt"))
    ap.add_argument("--cosine-ckpt", default=str(HERE / "checkpoints/mnist_cosine.pt"))
    ap.add_argument("--linear-log", default=str(p24 / "outputs/train_log.csv"))
    ap.add_argument("--cosine-log", default=str(HERE / "outputs/train_log_cosine.csv"))
    ap.add_argument("--n", type=int, default=256, help="samples per model for the metric")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--out-dir", default=str(HERE / "outputs"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_alpha_bar(out_dir / "alpha_bar.png")
    plot_losses(Path(args.linear_log), Path(args.cosine_log), out_dir / "loss_curves.png")

    print("sampling from both models (full 1000-step DDPM loop)...")
    x_lin = sample_from(Path(args.linear_ckpt), args.n, args.batch, args.device, args.seed)
    x_cos = sample_from(Path(args.cosine_ckpt), args.n, args.batch, args.device, args.seed)
    up = lambda x: F.interpolate(x, scale_factor=2, mode="nearest")  # noqa: E731
    save_image(up(x_lin[:64]), out_dir / "samples_linear.png", nrow=8, normalize=True, value_range=(-1, 1))
    save_image(up(x_cos[:64]), out_dir / "samples_cosine.png", nrow=8, normalize=True, value_range=(-1, 1))

    # Frechet distance in MNIST-classifier feature space, against real test digits.
    net = load_feature_net(HERE / "checkpoints/feature_net.pt", args.data_dir, args.device)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    test = datasets.MNIST(args.data_dir, train=False, download=True, transform=tf)
    reals = torch.stack([test[i][0] for i in range(2048)])

    with torch.no_grad():
        stats_real = gaussian_stats(net.features(reals))
        rows = [("schedule", "frechet_distance")]
        for name, x in (("linear", x_lin), ("cosine", x_cos)):
            fd = frechet_distance(*stats_real, *gaussian_stats(net.features(x)))
            rows.append((name, f"{fd:.2f}"))
            print(f"{name}: feature-space Frechet distance = {fd:.2f}")
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)


if __name__ == "__main__":
    main()
