"""A GenEval-style compositional benchmark on a toy two-slot digit generator.

GenEval doesn't ask "are the images pretty" — it asks "did the model draw the
RIGHT THINGS in the RIGHT NUMBERS". It scores a generator on structured prompt
categories (single object, two objects, counting, position/binding) using an
object detector to check the output against the prompt. Accuracy falls off a
cliff as the prompts get more compositional — that gradient is the whole point.

We reproduce that here. The 'generator' is a 28x56 two-slot diffusion model
conditioned on an unordered pair of digit classes (see compose_model.py). The
'object detector' is the MNIST CNN from project 58, applied to each half of the
canvas. We evaluate three categories of rising difficulty:

    single      {c}      -> exactly one object, class c
    two-same    {c, c}   -> two objects, both class c        (counting)
    two-diff    {a, b}   -> both classes present             (co-occurrence/binding)

    python geneval.py --data-dir data      # ~8 min on CPU

Reuses the phase-5 diffusion stack and project 58's classifier via sys.path.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
sys.path.insert(0, str(HERE.parent / "58-caption-ablation"))

from diffusion import GaussianDiffusion  # noqa: E402
from train import EMA, infinite  # noqa: E402
from compose_model import TwoSlotUNet, EMPTY  # noqa: E402
from mnist_classifier import load_or_train  # noqa: E402

OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"
T = 200


# --------------------------------------------------------------------------- #
# Training data: 28x56 canvases holding one or two digits in the two halves.
# --------------------------------------------------------------------------- #
class TwoDigitCanvas(torch.utils.data.Dataset):
    def __init__(self, data_dir, length=20000, seed=0):
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
        self.ds = datasets.MNIST(str(data_dir), train=True, download=True, transform=tf)
        self.by_class = {c: [] for c in range(10)}
        for i, y in enumerate(self.ds.targets.tolist()):
            self.by_class[y].append(i)
        self.length = length
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.length

    def _digit(self, c):
        i = self.rng.choice(self.by_class[c])
        return self.ds[int(i)][0]

    def __getitem__(self, _):
        canvas = torch.full((1, 28, 56), -1.0)
        n = self.rng.choice([1, 2], p=[0.4, 0.6])
        if n == 1:
            c = int(self.rng.integers(0, 10))
            tokens = [c, EMPTY]
        else:
            a, b = int(self.rng.integers(0, 10)), int(self.rng.integers(0, 10))
            tokens = [a, b]
        self.rng.shuffle(tokens)  # random left/right order
        for slot, tok in enumerate(tokens):
            if tok == EMPTY:
                continue
            canvas[:, :, slot * 28:(slot + 1) * 28] = self._digit(tok)
        return canvas, torch.tensor(tokens, dtype=torch.long)


def train(data_dir, out, steps=1400, seed=0):
    torch.manual_seed(seed)
    model = TwoSlotUNet(in_ch=1, image_size=28)
    ema = EMA(model, 0.995)
    diffusion = GaussianDiffusion(T=T, schedule="linear", device="cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    ds = TwoDigitCanvas(data_dir, seed=seed)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2, drop_last=True)
    batches = infinite(loader)
    print(f"two-slot U-Net params: {sum(p.numel() for p in model.parameters()):,}")
    t0 = time.time()
    for step in range(1, steps + 1):
        x0, pair = next(batches)
        loss = diffusion.loss(model, x0, model_kwargs={"pair": pair})
        opt.zero_grad(); loss.backward(); opt.step(); ema.update(model)
        if step % 200 == 0:
            print(f"  step {step}/{steps} | loss {loss.item():.4f} | "
                  f"{step/(time.time()-t0):.1f} it/s", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ema": ema.shadow.state_dict()}, out)
    return ema.shadow.eval()


# --------------------------------------------------------------------------- #
# The 'object detector': classify each half; a low-ink half is 'empty'.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def detect(canvases, clf, ink_thresh=0.04):
    """Return a list of detected digit multisets (one per canvas)."""
    halves = torch.cat([canvases[:, :, :, :28], canvases[:, :, :, 28:]], dim=0)
    occupied = (halves > 0.0).float().mean(dim=(1, 2, 3)) > ink_thresh
    pred = clf(halves).argmax(dim=1)
    B = canvases.size(0)
    out = []
    for i in range(B):
        objs = []
        for h in (i, i + B):  # left, right halves
            if occupied[h]:
                objs.append(int(pred[h]))
        out.append(sorted(objs))
    return out


# --------------------------------------------------------------------------- #
# The benchmark.
# --------------------------------------------------------------------------- #
def build_prompts(n_per=40, seed=1):
    rng = np.random.default_rng(seed)
    prompts = []
    for _ in range(n_per):
        c = int(rng.integers(0, 10))
        prompts.append(("single", [c, EMPTY], [c]))
    for _ in range(n_per):
        c = int(rng.integers(0, 10))
        prompts.append(("two-same", [c, c], [c, c]))
    for _ in range(n_per):
        a = int(rng.integers(0, 10))
        b = int((a + rng.integers(1, 10)) % 10)
        prompts.append(("two-diff", sorted([a, b]), sorted([a, b])))
    return prompts


@torch.no_grad()
def run_benchmark(model, diffusion, prompts, clf, seed=2):
    torch.manual_seed(seed)
    pairs = torch.tensor([p[1] for p in prompts], dtype=torch.long)
    x = torch.randn(len(prompts), 1, 28, 56)
    canvases, _ = diffusion.p_sample_loop(model, x.shape,
                                          model_kwargs={"pair": pairs}, x_init=x)
    detected = detect(canvases, clf)
    return canvases, detected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--steps", type=int, default=1400)
    ap.add_argument("--n-per", type=int, default=40)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    diffusion = GaussianDiffusion(T=T, schedule="linear", device="cpu")

    clf = load_or_train(args.data_dir,
                        HERE.parent / "58-caption-ablation/checkpoints/classifier.pt")
    print("training two-slot compositional generator ...")
    model = train(args.data_dir, CKPT / "compose.pt", args.steps)

    prompts = build_prompts(args.n_per)
    canvases, detected = run_benchmark(model, diffusion, prompts, clf)

    # Score each category.
    cats = ["single", "two-same", "two-diff"]
    correct = {c: 0 for c in cats}
    total = {c: 0 for c in cats}
    fails = {c: [] for c in cats}
    for i, (cat, _, want) in enumerate(prompts):
        got = detected[i]
        ok = got == sorted(want)
        total[cat] += 1
        correct[cat] += int(ok)
        if not ok and len(fails[cat]) < 4:
            fails[cat].append((i, want, got))
    acc = {c: correct[c] / total[c] for c in cats}

    plot_accuracy(acc, OUT / "accuracy.png")
    plot_samples(canvases, prompts, detected, OUT / "samples.png")

    lines = ["category,accuracy,n"]
    for c in cats:
        lines.append(f"{c},{acc[c]:.3f},{total[c]}")
    (OUT / "results.csv").write_text("\n".join(lines) + "\n")

    print("\nGenEval-style accuracy by category:")
    for c in cats:
        print(f"  {c:9s} {acc[c]:.0%}  ({correct[c]}/{total[c]})")
    print("\nsample failure modes:")
    for c in cats:
        for i, want, got in fails[c][:2]:
            print(f"  {c}: asked {want}, got {got}")
    print(f"\nwrote figures + {OUT/'results.csv'}")


def plot_accuracy(acc, path):
    import matplotlib
    matplotlib.use("Agg")
    sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
    import plot_style as ps
    cats = list(acc.keys()); vals = [acc[c] for c in cats]
    fig, ax = ps.new_axes(6.8, 4.0)
    colors = [ps.SERIES[1], ps.SERIES[0], ps.SERIES[2]]
    ax.bar(range(len(cats)), vals, color=colors, width=0.6)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([f"{c}\n({'1 obj' if c=='single' else '2 obj'})" for c in cats])
    ax.set_ylim(0, 1.05)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center", color=ps.INK_SECONDARY, fontsize=10)
    ps.finish(fig, ax, "Compositional accuracy falls as prompts get harder",
              "prompt category (increasing compositionality →)", "accuracy", path)


def plot_samples(canvases, prompts, detected, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # 3 rows (categories) x 6 columns of examples.
    cats = ["single", "two-same", "two-diff"]
    idx_by_cat = {c: [i for i, p in enumerate(prompts) if p[0] == c] for c in cats}
    fig, axes = plt.subplots(3, 6, figsize=(11, 4.2), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    for r, c in enumerate(cats):
        for k in range(6):
            i = idx_by_cat[c][k]
            ax = axes[r][k]
            img = ((canvases[i, 0].clamp(-1, 1) + 1) * 127.5).byte().numpy()
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            want = sorted(prompts[i][2]); got = detected[i]
            ok = want == got
            ax.set_title(f"ask {want}\ngot {got}", fontsize=7,
                         color="#1baf7a" if ok else "#e34948")
            ax.axis("off")
        axes[r][0].set_ylabel(c, fontsize=9)
    fig.suptitle("Prompt (ask) vs. what the detector found (got) — green = match, red = fail",
                 fontsize=11, color="#0b0b0b")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
