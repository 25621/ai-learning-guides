"""Caption ablation: train two identical conditional generators, one on clean
captions and one on noisy 'web alt-text', and measure how well each follows a
prompt.

The only difference between the two runs is the *labels*. The clean model sees
correct captions. The noisy model sees the same images, but a fraction of the
captions name the wrong digit — a stand-in for real web alt-text, which is
frequently wrong or unrelated to the image. We then ask both models to draw
each digit 0-9, read back what they actually drew with a CNN classifier, and
report prompt-adherence.

    python ablation.py --data-dir data      # ~7 min on CPU (two trainings)

Reuses the phase-5/9 conditional DDPM stack via sys.path.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
sys.path.insert(0, str(HERE.parent / "28-class-conditional-ddpm"))

from diffusion import GaussianDiffusion  # noqa: E402
from train import EMA, infinite  # noqa: E402
from unet_conditional import ConditionalUNet  # noqa: E402
from mnist_classifier import load_or_train, read_digits  # noqa: E402

OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"
T = 200


class CaptionedMNIST(Dataset):
    """MNIST where a fixed fraction of captions (labels) are corrupted to a
    random wrong digit — a toy model of noisy web alt-text."""

    def __init__(self, data_dir, noise=0.0, seed=0):
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
        self.ds = datasets.MNIST(str(data_dir), train=True, download=True, transform=tf)
        rng = np.random.default_rng(seed)
        true = np.array(self.ds.targets)
        caption = true.copy()
        flip = rng.random(len(true)) < noise
        caption[flip] = (true[flip] + rng.integers(1, 10, size=flip.sum())) % 10
        self.caption = torch.tensor(caption, dtype=torch.long)
        self.frac_wrong = float(flip.mean())

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, _ = self.ds[i]
        return x, self.caption[i]


def train_model(data_dir, noise, out, steps=1000, seed=0):
    torch.manual_seed(seed)
    ds = CaptionedMNIST(data_dir, noise=noise, seed=seed)
    print(f"  captions wrong: {ds.frac_wrong:.0%}")
    loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=2, drop_last=True)
    config = dict(in_ch=1, image_size=28, attn_resolutions=(7,))
    model = ConditionalUNet(num_classes=10, **config)
    ema = EMA(model, 0.995)
    diffusion = GaussianDiffusion(T=T, schedule="linear", device="cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    batches = infinite(loader)
    import time
    t0 = time.time()
    for step in range(1, steps + 1):
        x0, y = next(batches)
        loss = diffusion.loss(model, x0, model_kwargs={"y": y})
        opt.zero_grad(); loss.backward(); opt.step(); ema.update(model)
        if step % 300 == 0:
            print(f"  step {step}/{steps} | loss {loss.item():.4f} | "
                  f"{step/(time.time()-t0):.1f} it/s", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ema": ema.shadow.state_dict(), "config": config}, out)
    return ema.shadow.eval()


@torch.no_grad()
def sample_grid(model, diffusion, per_class=24, seed=0):
    """Draw `per_class` samples for each digit 0-9. Returns (images, labels)."""
    torch.manual_seed(seed)
    y = torch.arange(10).repeat_interleave(per_class)
    x = torch.randn(len(y), 1, 28, 28)
    x, _ = diffusion.p_sample_loop(model, x.shape, model_kwargs={"y": y}, x_init=x)
    return x, y


def plot_adherence(clean_acc, noisy_acc, clean_per, noisy_per, path):
    import matplotlib
    matplotlib.use("Agg")
    sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
    import plot_style as ps

    fig, ax = ps.new_axes(7.6, 4.2)
    x = np.arange(10)
    w = 0.4
    ax.bar(x - w / 2, clean_per, w, color=ps.SERIES[0], label=f"clean captions ({clean_acc:.0%})")
    ax.bar(x + w / 2, noisy_per, w, color=ps.SERIES[2], label=f"noisy alt-text ({noisy_acc:.0%})")
    ax.set_xticks(x)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Prompt adherence per digit: does the model draw what you asked?",
              "requested digit", "fraction drawn correctly", path)


def plot_samples(imgs, labels, per_class, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols, rows = 10, 8
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 0.7, rows * 0.7), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes.flat:
        ax.axis("off")
    for c in range(10):
        for r in range(rows):
            idx = c * per_class + r
            ax = axes[r][c]
            img = ((imgs[idx, 0].clamp(-1, 1) + 1) * 127.5).byte().numpy()
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            if r == 0:
                ax.set_title(str(c), fontsize=10, color="#0b0b0b")
    fig.suptitle(title, color="#0b0b0b", fontsize=12, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def adherence(imgs, requested, clf):
    pred, _ = read_digits(clf, imgs)
    per = np.array([(pred[requested == c] == c).float().mean().item() for c in range(10)])
    return float((pred == requested).float().mean()), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--noise", type=float, default=0.5)
    ap.add_argument("--per-class", type=int, default=20)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    diffusion = GaussianDiffusion(T=T, schedule="linear", device="cpu")

    clf = load_or_train(args.data_dir, CKPT / "classifier.pt")

    print("training CLEAN-caption model ...")
    clean = train_model(args.data_dir, 0.0, CKPT / "clean.pt", args.steps, seed=0)
    print(f"training NOISY-caption model ({args.noise:.0%} wrong) ...")
    noisy = train_model(args.data_dir, args.noise, CKPT / "noisy.pt", args.steps, seed=0)

    ci, cy = sample_grid(clean, diffusion, args.per_class, seed=1)
    ni, ny = sample_grid(noisy, diffusion, args.per_class, seed=1)

    clean_acc, clean_per = adherence(ci, cy, clf)
    noisy_acc, noisy_per = adherence(ni, ny, clf)
    print(f"\nclean adherence: {clean_acc:.1%} | noisy adherence: {noisy_acc:.1%}")

    plot_adherence(clean_acc, noisy_acc, clean_per, noisy_per, OUT / "adherence.png")
    plot_samples(ci, cy, args.per_class, OUT / "samples_clean.png",
                 f"Clean-caption model — asked for 0-9 across columns ({clean_acc:.0%} correct)")
    plot_samples(ni, ny, args.per_class, OUT / "samples_noisy.png",
                 f"Noisy-caption model — asked for 0-9 across columns ({noisy_acc:.0%} correct)")

    (OUT / "results.csv").write_text(
        "model,overall_adherence\n"
        f"clean,{clean_acc:.4f}\nnoisy,{noisy_acc:.4f}\n")
    print(f"wrote figures + {OUT/'results.csv'}")


if __name__ == "__main__":
    main()
