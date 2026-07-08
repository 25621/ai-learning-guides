"""Vanilla DCGAN on MNIST: the non-saturating loss, trained two ways.

`--config stable`   balanced G/D learning rates -> reasonable digits, losses
                     oscillate but don't run away.
`--config collapse` G's learning rate is 10x D's -> D can't keep up, so G
                     stops exploring and settles on the handful of fakes that
                     already fool it: classic runaway-generator mode collapse.

Each config is one short invocation so it fits in small time slices; `--plot`
combines the saved histories into the report figures.

    python train.py --config stable    --data-dir data   # ~3 min on CPU
    python train.py --config collapse  --data-dir data   # ~1.5 min on CPU
    python train.py --plot
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dcgan import Discriminator, Generator, infinite, mnist_loader, weights_init

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"

sys.path.insert(0, str(HERE.parent / "58-caption-ablation"))
from mnist_classifier import load_or_train, read_digits  # noqa: E402

CONFIGS = {
    # nz, lr_g, lr_d, d_steps_per_g, steps
    # "stable": balanced Adam rates, the DCGAN-paper recipe.
    # "collapse": G races 10x faster than D. D can't keep up, so G stops
    # exploring and settles on the handful of fakes that already fool it —
    # the classic runaway-generator mode collapse.
    "stable": dict(nz=100, lr_g=2e-4, lr_d=2e-4, d_steps=1, steps=500),
    "collapse": dict(nz=100, lr_g=2e-3, lr_d=5e-5, d_steps=1, steps=350),
}


def run(cfg_name, data_dir, seed=0):
    cfg = CONFIGS[cfg_name]
    torch.manual_seed(seed)
    loader = infinite(mnist_loader(data_dir, batch_size=128))

    G = Generator(nz=cfg["nz"], nc=1); G.apply(weights_init)
    D = Discriminator(nc=1, norm="batch"); D.apply(weights_init)
    opt_g = torch.optim.Adam(G.parameters(), lr=cfg["lr_g"], betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg["lr_d"], betas=(0.5, 0.999))

    history = []
    t0 = time.time()
    for step in range(cfg["steps"]):
        for _ in range(cfg["d_steps"]):
            x_real, _ = next(loader)
            z = torch.randn(x_real.size(0), cfg["nz"])
            with torch.no_grad():
                x_fake = G(z)
            d_real = D(x_real)
            d_fake = D(x_fake)
            loss_d = (F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
                      + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake)))
            opt_d.zero_grad(); loss_d.backward(); opt_d.step()

        z = torch.randn(128, cfg["nz"])
        x_fake = G(z)
        d_fake = D(x_fake)
        loss_g = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
        opt_g.zero_grad(); loss_g.backward(); opt_g.step()

        if step % 20 == 0 or step == cfg["steps"] - 1:
            history.append((step, loss_g.item(), loss_d.item()))
        if step % 200 == 0:
            print(f"[{cfg_name}] step {step} loss_g={loss_g.item():.3f} loss_d={loss_d.item():.3f} "
                  f"({time.time() - t0:.0f}s)")

    G.eval()
    with torch.no_grad():
        z = torch.randn(64, cfg["nz"])
        samples = G(z).clamp(-1, 1)

    (HERE / "checkpoints").mkdir(exist_ok=True)
    clf = load_or_train(data_dir, HERE / "checkpoints" / "mnist_clf.pt")
    crop = samples[:, :, 2:30, 2:30]  # undo the 32x32 pad back to 28x28 for the classifier
    preds, conf = read_digits(clf, crop)
    counts = np.bincount(preds.numpy(), minlength=10)
    unique_digits = int((counts > 0).sum())

    OUT.mkdir(exist_ok=True)
    np.savez(OUT / f"run_{cfg_name}.npz",
             history=np.array(history), samples=samples.numpy(),
             counts=counts, wall_time=time.time() - t0)
    print(f"[{cfg_name}] done in {time.time() - t0:.0f}s, "
          f"{unique_digits}/10 digit classes present in 64 samples, counts={counts.tolist()}")


def make_plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
    import plot_style as ps

    runs = {name: np.load(OUT / f"run_{name}.npz") for name in CONFIGS}

    # Loss curves, stable vs collapse, side by side.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=100)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax, name in zip(axes, CONFIGS):
        h = runs[name]["history"]
        ax.set_facecolor(ps.SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.plot(h[:, 0], h[:, 1], color=ps.SERIES[0], label="G loss", linewidth=1.2)
        ax.plot(h[:, 0], h[:, 2], color=ps.SERIES[2], label="D loss", linewidth=1.2)
        ax.set_title(name, loc="left", color=ps.INK, fontsize=11)
        ax.set_xlabel("step", color=ps.INK_SECONDARY, fontsize=9)
        ax.tick_params(colors=ps.INK_MUTED, labelsize=8)
        ax.grid(True, color=ps.GRID, linewidth=0.7)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Non-saturating GAN losses: oscillating-but-stable vs a starved generator",
                 fontsize=12, color=ps.INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "loss_curves.png", facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'loss_curves.png'}")

    # Sample grids + per-class histogram.
    fig, axes = plt.subplots(2, 8, figsize=(9, 2.6), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for r, name in enumerate(CONFIGS):
        imgs = runs[name]["samples"][:8]
        for c in range(8):
            img = (imgs[c, 0] + 1) / 2
            axes[r][c].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[r][c].axis("off")
        axes[r][0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9, color=ps.INK)
        axes[r][0].axis("on")
        axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
        for s in axes[r][0].spines.values():
            s.set_visible(False)
    fig.suptitle("Samples: healthy variety (top) vs mode collapse (bottom)", fontsize=11, color=ps.INK)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "samples.png", facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'samples.png'}")

    fig, ax = ps.new_axes(7.2, 4.0)
    width = 0.35
    xs = np.arange(10)
    ax.bar(xs - width / 2, runs["stable"]["counts"], width, color=ps.SERIES[0], label="stable")
    ax.bar(xs + width / 2, runs["collapse"]["counts"], width, color=ps.SERIES[2], label="collapse")
    ax.set_xticks(xs)
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "What digit did the generator actually draw? (64 samples, read by a digit classifier)",
              "predicted digit", "count", OUT / "digit_histogram.png")

    with open(OUT / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "wall_time_s", "unique_digit_classes", "normalized_entropy", "counts"])
        for name in CONFIGS:
            r = runs[name]
            counts = r["counts"].astype(float)
            unique = int((counts > 0).sum())
            p = counts / counts.sum()
            p_nz = p[p > 0]
            entropy = float(-(p_nz * np.log(p_nz)).sum() / np.log(10))  # 1.0 = uniform over 10 digits, 0 = one digit
            w.writerow([name, f"{float(r['wall_time']):.1f}", unique, f"{entropy:.3f}", r["counts"].tolist()])
    print(f"wrote {OUT / 'results.csv'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=list(CONFIGS))
    p.add_argument("--data-dir", default="data")
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    if args.plot:
        make_plots()
    else:
        run(args.config, args.data_dir)
