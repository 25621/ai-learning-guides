"""Conditional GAN on CIFAR-10 with a projection discriminator.

The generator gets the class as a one-hot vector concatenated to its noise
(simple label conditioning). The critic instead uses a *projection*
discriminator (Miyato & Koyama, 2018): a class-agnostic realism score plus
`embed(y) . features(x)`, which conditions far more strongly than stapling
the label on as an extra input channel would.

To check conditioning actually works, we train a small independent CIFAR-10
classifier (`cifar_clf.py`) and use it to *read* what class each generated
image looks like — then compare that to the class we asked for.

    python train.py --steps 2000 --data-dir data   # ~8 min on CPU
    python train.py --eval --data-dir data          # ~1.5 min (trains the judge)
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
from torchvision import transforms

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"

sys.path.insert(0, str(HERE.parent / "18-vanilla-gan-on-mnist"))
from dcgan import Generator, ProjectionDiscriminator, weights_init  # noqa: E402
from cifar_clf import CIFAR10Npz, load_or_train, read_classes  # noqa: E402

NZ = 100
BATCH = 128
CLASSES = 10


def loader(data_dir):
    tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])
    ds = CIFAR10Npz(Path(data_dir) / "cifar10_train.npz", tf)
    return torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0, drop_last=True)


def train(steps, data_dir, seed=0):
    torch.manual_seed(seed)
    dl = loader(data_dir)
    it = iter(dl)

    G = Generator(nz=NZ, nc=3, n_classes=CLASSES); G.apply(weights_init)
    D = ProjectionDiscriminator(nc=3, n_classes=CLASSES)
    opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    history = []
    t0 = time.time()
    for step in range(steps):
        try:
            x_real, y_real = next(it)
        except StopIteration:
            it = iter(dl); x_real, y_real = next(it)

        z = torch.randn(x_real.size(0), NZ)
        with torch.no_grad():
            x_fake = G(z, y_real)
        d_real = D(x_real, y_real)
        d_fake = D(x_fake, y_real)
        loss_d = (F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
                  + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake)))
        opt_d.zero_grad(); loss_d.backward(); opt_d.step()

        z = torch.randn(BATCH, NZ)
        y_fake = torch.randint(0, CLASSES, (BATCH,))
        x_fake = G(z, y_fake)
        d_fake = D(x_fake, y_fake)
        loss_g = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
        opt_g.zero_grad(); loss_g.backward(); opt_g.step()

        if step % 20 == 0:
            history.append((step, loss_g.item(), loss_d.item()))
        if step % 100 == 0:
            print(f"step {step} loss_g={loss_g.item():.3f} loss_d={loss_d.item():.3f} ({time.time() - t0:.0f}s)")

    OUT.mkdir(exist_ok=True)
    (HERE / "checkpoints").mkdir(exist_ok=True)
    torch.save(G.state_dict(), HERE / "checkpoints/generator.pt")
    np.save(OUT / "history.npy", np.array(history))
    print(f"done in {time.time() - t0:.0f}s")


@torch.no_grad()
def evaluate(data_dir, n_per_class=30, seed=1):
    torch.manual_seed(seed)
    G = Generator(nz=NZ, nc=3, n_classes=CLASSES)
    G.load_state_dict(torch.load(HERE / "checkpoints/generator.pt"))
    G.eval()

    clf = load_or_train(data_dir, HERE / "checkpoints/cifar_clf.pt", steps=1500)

    confusion = np.zeros((CLASSES, CLASSES), dtype=int)  # [requested, predicted]
    sample_grid = []
    for c in range(CLASSES):
        y = torch.full((n_per_class,), c, dtype=torch.long)
        z = torch.randn(n_per_class, NZ)
        imgs = G(z, y).clamp(-1, 1)
        preds, _ = read_classes(clf, imgs)
        for p in preds.tolist():
            confusion[c, p] += 1
        sample_grid.append(imgs[:8].numpy())

    per_class_acc = confusion.diagonal() / confusion.sum(1)
    overall_acc = confusion.diagonal().sum() / confusion.sum()

    np.savez(OUT / "eval.npz", confusion=confusion, sample_grid=np.array(sample_grid),
             per_class_acc=per_class_acc, overall_acc=overall_acc)
    print(f"overall class-adherence accuracy: {overall_acc:.3f}")
    print("per-class:", np.round(per_class_acc, 2).tolist())


CIFAR_LABELS = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def make_plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
    import plot_style as ps

    hist = np.load(OUT / "history.npy")
    fig, ax = ps.new_axes(7.2, 4.0)
    ax.plot(hist[:, 0], hist[:, 1], color=ps.SERIES[0], label="G loss", linewidth=1.1)
    ax.plot(hist[:, 0], hist[:, 2], color=ps.SERIES[2], label="D loss", linewidth=1.1)
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Conditional GAN training losses", "step", "loss", OUT / "loss_curves.png")

    ev = np.load(OUT / "eval.npz")
    confusion = ev["confusion"]
    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    im = ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(CLASSES)); ax.set_xticklabels(CIFAR_LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(CLASSES)); ax.set_yticklabels(CIFAR_LABELS, fontsize=8)
    ax.set_xlabel("classifier's reading", color=ps.INK_SECONDARY, fontsize=9)
    ax.set_ylabel("class requested from G", color=ps.INK_SECONDARY, fontsize=9)
    for i in range(CLASSES):
        for j in range(CLASSES):
            v = confusion[i, j]
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center",
                        fontsize=7, color="white" if v > confusion.max() / 2 else ps.INK)
    ax.set_title(f"Requested vs recognized class (overall accuracy {float(ev['overall_acc']):.0%})",
                 color=ps.INK, fontsize=11, loc="left", pad=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'confusion_matrix.png'}")

    grid = ev["sample_grid"]  # (10, 8, 3, 32, 32)
    fig, axes = plt.subplots(CLASSES, 8, figsize=(9, 11), dpi=100)
    fig.patch.set_facecolor(ps.SURFACE)
    for c in range(CLASSES):
        for j in range(8):
            img = ((grid[c, j].transpose(1, 2, 0) + 1) / 2).clip(0, 1)
            axes[c][j].imshow(img); axes[c][j].axis("off")
        axes[c][0].axis("on"); axes[c][0].set_xticks([]); axes[c][0].set_yticks([])
        for s in axes[c][0].spines.values():
            s.set_visible(False)
        axes[c][0].set_ylabel(CIFAR_LABELS[c], rotation=0, ha="right", va="center", fontsize=9, color=ps.INK)
    fig.suptitle("Ask for a class, get that class (one row per requested label)", fontsize=12, color=ps.INK)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "class_grid.png", facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'class_grid.png'}")

    with open(OUT / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "adherence_accuracy"])
        for c, acc in zip(CIFAR_LABELS, ev["per_class_acc"]):
            w.writerow([c, f"{acc:.3f}"])
        w.writerow(["overall", f"{float(ev['overall_acc']):.3f}"])
    print(f"wrote {OUT / 'results.csv'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    if args.plot:
        make_plots()
    elif args.eval:
        evaluate(args.data_dir)
    else:
        train(args.steps, args.data_dir)
