"""Classifier-guided sampling (Dhariwal & Nichol 2021, pre-CFG).

The UNCONDITIONAL diffusion model from project 24 never saw a label. At each
reverse step we nudge its proposed mean toward images the noisy classifier
scores as the target class:

    mean <- mean + s * Sigma_t * grad_{x_t} log p(y | x_t, t)

s = 0 is ordinary unconditional sampling; larger s trades diversity for
class purity.

Run (after project 24's training and noisy_classifier.py):
    python guided_sampling.py
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
from diffusion import GaussianDiffusion  # noqa: E402
from unet import UNet  # noqa: E402

import plot_style as ps  # noqa: E402
from noisy_classifier import NoisyClassifier  # noqa: E402


def classifier_grad(clf, x_t, t, y):
    """grad_{x_t} log p(y | x_t, t) — the steering signal."""
    x_in = x_t.detach().requires_grad_(True)
    log_probs = F.log_softmax(clf(x_in, t), dim=-1)
    selected = log_probs[torch.arange(y.size(0)), y].sum()
    return torch.autograd.grad(selected, x_in)[0]


@torch.no_grad()
def guided_sample(model, clf, diffusion, y, scale, shape, device, x_init=None):
    x = torch.randn(shape, device=device) if x_init is None else x_init.clone()
    for t in reversed(range(diffusion.T)):
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
        eps = model(x, t_batch)
        mean = diffusion.posterior_mean(x, eps, t)
        if scale > 0:
            with torch.enable_grad():
                grad = classifier_grad(clf, x, t_batch, y)
            mean = mean + scale * diffusion.posterior_var[t] * grad
        if t > 0:
            x = mean + diffusion.posterior_var[t].sqrt() * torch.randn_like(x)
        else:
            x = mean
    return x


def plot_accuracy_vs_t(csv_path: Path, out_path: Path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    ts = [int(r["t"]) for r in rows]
    accs = [float(r["accuracy"]) for r in rows]
    fig, ax = ps.new_axes()
    ax.plot(ts, accs, color=ps.SERIES[0], linewidth=2, marker="o", markersize=5)
    ax.axhline(0.1, color=ps.BASELINE, linewidth=1, linestyle="--")
    ax.annotate("chance (10 classes)", (ts[len(ts) // 2], 0.115),
                color=ps.INK_MUTED, fontsize=9)
    ax.set_ylim(0, 1.02)
    ps.finish(fig, ax, "Noisy-classifier accuracy vs diffusion step",
              "diffusion step t of the noised input", "test accuracy", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddpm-ckpt", default=str(HERE.parent / "24-ddpm-on-mnist/checkpoints/mnist_linear.pt"))
    ap.add_argument("--clf-ckpt", default=str(HERE / "checkpoints/noisy_classifier.pt"))
    ap.add_argument("--target", type=int, default=8, help="class for the scale sweep")
    ap.add_argument("--n", type=int, default=8, help="samples per grid row")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default=str(HERE / "outputs"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    device = args.device
    ckpt = torch.load(args.ddpm_ckpt, map_location=device, weights_only=True)
    model = UNet().to(device)
    model.load_state_dict(ckpt["ema"])
    model.eval()
    diffusion = GaussianDiffusion(T=ckpt["T"], schedule=ckpt["schedule"], device=device)

    clf = NoisyClassifier().to(device)
    clf.load_state_dict(
        torch.load(args.clf_ckpt, map_location=device, weights_only=True)["model"]
    )
    clf.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_accuracy_vs_t(out_dir / "accuracy_vs_t.csv", out_dir / "accuracy_vs_t.png")

    # Scale sweep on one target class, same starting noise per row.
    torch.manual_seed(args.seed)
    shape = (args.n, 1, 28, 28)
    x_T = torch.randn(shape, device=device)
    y = torch.full((args.n,), args.target, device=device, dtype=torch.long)
    scales = (0.0, 1.0, 5.0, 20.0)
    rows, purity_rows = [], [("scale", "fraction_classified_as_target")]
    for s in scales:
        x0 = guided_sample(model, clf, diffusion, y, s, shape, device, x_init=x_T)
        rows.append(x0)
        with torch.no_grad():
            t0 = torch.zeros(args.n, device=device, dtype=torch.long)
            frac = (clf(x0, t0).argmax(1) == y).float().mean().item()
        purity_rows.append((s, f"{frac:.2f}"))
        print(f"scale {s:5.1f}: {frac:.0%} of samples classified as '{args.target}'", flush=True)
    up = lambda x: F.interpolate(x, scale_factor=2, mode="nearest")  # noqa: E731
    save_image(up(torch.cat(rows)), out_dir / "scale_sweep.png", nrow=args.n,
               normalize=True, value_range=(-1, 1))
    with open(out_dir / "purity.csv", "w", newline="") as f:
        csv.writer(f).writerows(purity_rows)

    # One row per class at a moderate scale.
    torch.manual_seed(args.seed + 1)
    class_rows = []
    for c in range(10):
        y_c = torch.full((args.n,), c, device=device, dtype=torch.long)
        class_rows.append(
            guided_sample(model, clf, diffusion, y_c, 5.0, shape, device)
        )
        print(f"sampled class {c}", flush=True)
    save_image(up(torch.cat(class_rows)), out_dir / "all_classes.png", nrow=args.n,
               normalize=True, value_range=(-1, 1))
    print(f"wrote {out_dir}/scale_sweep.png, {out_dir}/all_classes.png")


if __name__ == "__main__":
    main()
