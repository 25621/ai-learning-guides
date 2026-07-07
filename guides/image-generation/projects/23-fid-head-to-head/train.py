"""FID head-to-head: a DCGAN and a small VAE, same dataset (MNIST), same
number of gradient steps, judged by the same metric. A second DCGAN gets 3x
the steps, to see whether more compute (not more cleverness) closes the gap.

    python train.py --data-dir data             # ~2.7 min on CPU (500-step GAN + VAE)
    python train.py --long-gan --data-dir data   # ~6.7 min on CPU (1500-step GAN)
    python train.py --eval --data-dir data        # ~1 min (Inception feature extraction)
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

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"

sys.path.insert(0, str(HERE.parent / "18-vanilla-gan-on-mnist"))
from dcgan import Generator, Discriminator, infinite, mnist_loader, weights_init  # noqa: E402

sys.path.insert(0, str(HERE.parent / "06-tiny-ae-on-mnist"))
import vae_lib  # noqa: E402

sys.path.insert(0, str(HERE.parent / "04-fid-from-scratch"))
from fid import build_extractor, inception_features, fid as compute_fid  # noqa: E402

STEPS = 500          # matched-step-count comparison with the VAE
STEPS_LONG = 1500    # a 2nd GAN, given 3x the steps, to test whether more compute closes the FID gap
NZ = 100
N_EVAL = 300


def train_gan(data_dir, seed=0, steps=STEPS):
    torch.manual_seed(seed)
    loader = infinite(mnist_loader(data_dir, batch_size=128))
    G = Generator(nz=NZ, nc=1); G.apply(weights_init)
    D = Discriminator(nc=1); D.apply(weights_init)
    opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    history = []
    t0 = time.time()
    for step in range(steps):
        x_real, _ = next(loader)
        z = torch.randn(x_real.size(0), NZ)
        with torch.no_grad():
            x_fake = G(z)
        d_real, d_fake = D(x_real), D(x_fake)
        loss_d = (F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
                  + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake)))
        opt_d.zero_grad(); loss_d.backward(); opt_d.step()

        z = torch.randn(128, NZ)
        d_fake = D(G(z))
        loss_g = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
        opt_g.zero_grad(); loss_g.backward(); opt_g.step()

        if step % 10 == 0:
            history.append((step, loss_g.item(), loss_d.item()))
    wall_time = time.time() - t0
    print(f"GAN trained {steps} steps in {wall_time:.1f}s")

    G.eval()
    with torch.no_grad():
        z = torch.randn(N_EVAL, NZ)
        samples = G(z).clamp(-1, 1)[:, :, 2:30, 2:30]  # 32x32 -> 28x28, undo the pad
        samples = (samples + 1) / 2  # -> [0, 1], matches VAE's output range
    return np.array(history), samples.numpy(), wall_time


def vae_mnist_loader(data_dir, bs=128):
    """Like vae_lib.mnist_loader but num_workers=0 (worker subprocesses have
    destabilized long CPU sessions in this environment)."""
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor()])
    ds = datasets.MNIST(str(data_dir), train=True, download=True, transform=tf)
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0, drop_last=True)


def train_vae(data_dir, seed=0):
    torch.manual_seed(seed)
    loader = vae_lib.infinite(vae_mnist_loader(data_dir, bs=128))
    model = vae_lib.VAE(zdim=16, base=32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    history = []
    t0 = time.time()
    for step in range(STEPS):
        x, _ = next(loader)
        loss, recon, kl = model.loss(x)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 10 == 0:
            history.append((step, loss.item(), recon.item(), kl.item()))
    wall_time = time.time() - t0
    print(f"VAE trained {STEPS} steps in {wall_time:.1f}s")

    model.eval()
    samples = model.sample(N_EVAL)
    return np.array(history), samples.numpy(), wall_time


def get_real(data_dir, n=N_EVAL, seed=1):
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor()])
    ds = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n]
    imgs = torch.stack([ds[i][0] for i in idx])
    return imgs.numpy()


def to_rgb_pm1(x01):
    """[0,1] grayscale (N,1,28,28) -> [-1,1] RGB (N,3,28,28), what the shared
    Inception feature extractor from project 04 expects."""
    x = torch.from_numpy(x01) if isinstance(x01, np.ndarray) else x01
    x = x.repeat(1, 3, 1, 1)
    return x * 2 - 1


def _load_or_empty():
    path = OUT / "train_run.npz"
    return dict(np.load(path)) if path.exists() else {}


def train(data_dir):
    """Matched-step-count comparison: DCGAN and VAE both get STEPS updates."""
    OUT.mkdir(exist_ok=True)
    (HERE / "checkpoints").mkdir(exist_ok=True)
    gan_hist, gan_samples, gan_time = train_gan(data_dir, steps=STEPS)
    vae_hist, vae_samples, vae_time = train_vae(data_dir)
    save = _load_or_empty()
    save.update(gan_hist=gan_hist, gan_samples=gan_samples, gan_time=gan_time,
                vae_hist=vae_hist, vae_samples=vae_samples, vae_time=vae_time)
    print(f"GAN({STEPS} steps): {gan_time:.1f}s | VAE({STEPS} steps): {vae_time:.1f}s | "
          f"VAE is {gan_time / vae_time:.1f}x faster per step budget")
    np.savez(OUT / "train_run.npz", **save)


def train_long_gan(data_dir):
    """A second DCGAN, 3x the steps: does more compute close the FID gap?
    Kept as its own invocation so no single run exceeds a few minutes."""
    OUT.mkdir(exist_ok=True)
    gan_long_hist, gan_long_samples, gan_long_time = train_gan(data_dir, seed=0, steps=STEPS_LONG)
    save = _load_or_empty()
    save.update(gan_long_hist=gan_long_hist, gan_long_samples=gan_long_samples,
                gan_long_time=gan_long_time)
    print(f"GAN({STEPS_LONG} steps): {gan_long_time:.1f}s")
    np.savez(OUT / "train_run.npz", **save)


def evaluate(data_dir):
    d = np.load(OUT / "train_run.npz")
    real = get_real(data_dir)

    print("building Inception feature extractor...")
    extractor = build_extractor()

    feats_real = inception_features(to_rgb_pm1(real), extractor)
    feats_gan = inception_features(to_rgb_pm1(d["gan_samples"]), extractor)
    feats_vae = inception_features(to_rgb_pm1(d["vae_samples"]), extractor)

    fid_gan = compute_fid(feats_real, feats_gan)
    fid_vae = compute_fid(feats_real, feats_vae)
    print(f"FID(real, GAN@{STEPS}steps) = {fid_gan:.2f}")
    print(f"FID(real, VAE@{STEPS}steps) = {fid_vae:.2f}")

    # Stability proxy: variance of the last third of each model's logged loss.
    gan_hist, vae_hist = d["gan_hist"], d["vae_hist"]
    tail = len(gan_hist) // 3
    gan_g_var = float(np.var(gan_hist[-tail:, 1]))
    vae_loss_var = float(np.var(vae_hist[-tail:, 1]))

    save = dict(fid_gan=fid_gan, fid_vae=fid_vae, gan_g_var=gan_g_var, vae_loss_var=vae_loss_var)
    rows = [
        ["DCGAN", f"{fid_gan:.2f}", f"{float(d['gan_time']):.1f}", f"{gan_g_var:.4f}"],
        ["VAE", f"{fid_vae:.2f}", f"{float(d['vae_time']):.1f}", f"{vae_loss_var:.4f}"],
    ]

    if "gan_long_samples" in d.files:
        feats_gan_long = inception_features(to_rgb_pm1(d["gan_long_samples"]), extractor)
        fid_gan_long = compute_fid(feats_real, feats_gan_long)
        print(f"FID(real, GAN@{STEPS_LONG}steps) = {fid_gan_long:.2f}")
        gan_long_g_var = float(np.var(d["gan_long_hist"][-(len(d["gan_long_hist"]) // 3):, 1]))
        save["fid_gan_long"] = fid_gan_long
        rows.append([f"DCGAN ({STEPS_LONG} steps)", f"{fid_gan_long:.2f}",
                     f"{float(d['gan_long_time']):.1f}", f"{gan_long_g_var:.4f}"])

    np.savez(OUT / "eval.npz", **save)

    with open(OUT / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "fid", "wall_time_s", "late_training_loss_variance"])
        w.writerows(rows)
    print(f"wrote {OUT / 'results.csv'}")


def make_plots(data_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
    import plot_style as ps

    d = np.load(OUT / "train_run.npz")
    ev = np.load(OUT / "eval.npz")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=100)
    fig.patch.set_facecolor(ps.SURFACE)
    gan_hist = d["gan_hist"]
    axes[0].set_facecolor(ps.SURFACE)
    for side in ("top", "right"):
        axes[0].spines[side].set_visible(False)
    axes[0].plot(gan_hist[:, 0], gan_hist[:, 1], color=ps.SERIES[0], label="G loss", linewidth=1.1)
    axes[0].plot(gan_hist[:, 0], gan_hist[:, 2], color=ps.SERIES[2], label="D loss", linewidth=1.1)
    axes[0].set_title("DCGAN: oscillating, adversarial", loc="left", fontsize=10, color=ps.INK)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].tick_params(colors=ps.INK_MUTED, labelsize=8)
    axes[0].grid(True, color=ps.GRID, linewidth=0.7)

    vae_hist = d["vae_hist"]
    axes[1].set_facecolor(ps.SURFACE)
    for side in ("top", "right"):
        axes[1].spines[side].set_visible(False)
    axes[1].plot(vae_hist[:, 0], vae_hist[:, 1], color=ps.SERIES[1], label="ELBO loss", linewidth=1.1)
    axes[1].set_title("VAE: monotone, single-objective", loc="left", fontsize=10, color=ps.INK)
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].tick_params(colors=ps.INK_MUTED, labelsize=8)
    axes[1].grid(True, color=ps.GRID, linewidth=0.7)
    fig.suptitle("Same step budget, very different loss behavior", fontsize=12, color=ps.INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "loss_curves.png", facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'loss_curves.png'}")

    has_long = "gan_long_samples" in d.files
    rows_data = [("real", get_real(data_dir, n=8)), ("DCGAN", d["gan_samples"]), ("VAE", d["vae_samples"])]
    if has_long:
        rows_data.append((f"DCGAN\n({STEPS_LONG} steps)", d["gan_long_samples"]))
    fig, axes = plt.subplots(len(rows_data), 8, figsize=(9, 1.2 * len(rows_data)), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for r, (label, imgs) in enumerate(rows_data):
        for c in range(8):
            axes[r][c].imshow(imgs[c, 0], cmap="gray", vmin=0, vmax=1)
            axes[r][c].axis("off")
        axes[r][0].axis("on"); axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
        for s in axes[r][0].spines.values():
            s.set_visible(False)
        axes[r][0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8, color=ps.INK)
    fig.suptitle(f"Real vs DCGAN vs VAE samples (DCGAN also shown at {STEPS_LONG} steps)", fontsize=11, color=ps.INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "samples.png", facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'samples.png'}")

    fig, ax = ps.new_axes(7.0, 4.2)
    models = [f"DCGAN\n({STEPS} steps)", f"VAE\n({STEPS} steps)"]
    fids = [float(ev["fid_gan"]), float(ev["fid_vae"])]
    colors = [ps.SERIES[0], ps.SERIES[1]]
    if "fid_gan_long" in ev.files:
        models.append(f"DCGAN\n({STEPS_LONG} steps)")
        fids.append(float(ev["fid_gan_long"]))
        colors.append(ps.SERIES[2])
    ax.bar(models, fids, color=colors, width=0.5)
    for i, v in enumerate(fids):
        ax.text(i, v + max(fids) * 0.02, f"{v:.1f}", ha="center", fontsize=10, color=ps.INK)
    ps.finish(fig, ax, "FID vs real MNIST test images (lower is better)", "", "FID", OUT / "fid_bar.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--long-gan", action="store_true")
    args = p.parse_args()

    if args.plot:
        make_plots(args.data_dir)
    elif args.eval:
        evaluate(args.data_dir)
    elif args.long_gan:
        train_long_gan(args.data_dir)
    else:
        train(args.data_dir)
