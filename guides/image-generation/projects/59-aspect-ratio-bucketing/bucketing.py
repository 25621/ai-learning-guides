"""Aspect-ratio bucketing vs. the naive 'center-crop everything to a square'.

Web images come in every shape. The lazy fix — center-crop each one to a square
and resize — quietly throws away whatever sits near the edges of tall and wide
images, and it teaches the model that the world is square. Aspect-ratio
bucketing instead sorts images into a few shape buckets, resizes each image to
its bucket's resolution *without* cropping, and forms every mini-batch from a
single bucket (so all tensors in a batch share a shape and still stack).

This script builds a toy variable-aspect dataset from MNIST digits placed
off-center on non-square canvases, runs both strategies, and measures how much
of the original content each one keeps.

    python bucketing.py --data-dir data      # ~10s on CPU
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"

# The bucket menu: a few fixed (H, W) shapes. The extreme 3:1 panoramas are
# where square-cropping does the most damage.
BUCKETS = {"landscape": (24, 72), "portrait": (72, 24), "square": (40, 40)}


def build_variable_aspect(data_dir, n=600, seed=0):
    """Place each 28x28 digit off-center on a non-square canvas so that a
    center square crop is at real risk of clipping it."""
    rng = np.random.default_rng(seed)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    ds = datasets.MNIST(str(data_dir), train=False, download=True, transform=tf)
    idxs = rng.permutation(len(ds))[:n]
    items = []
    shapes = list(BUCKETS.values())
    for k, i in enumerate(idxs):
        x, y = ds[int(i)]                      # (1,28,28) in [-1,1]
        H, W = shapes[k % len(shapes)]
        # Scale the digit to fill the short side, then drop it at a random
        # position along the long side.
        s = min(H, W)
        digit = F.interpolate(x[None], size=(s, s), mode="bilinear",
                              align_corners=False)[0]
        canvas = torch.full((1, H, W), -1.0)   # black background
        top = rng.integers(0, H - s + 1)
        left = rng.integers(0, W - s + 1)
        canvas[:, top:top + s, left:left + s] = digit
        items.append((canvas, int(y)))
    return items


def ink(img):
    return (img > 0.0).float().sum().item()


# --------------------------------------------------------------------------- #
# Strategy A — center-crop to a square (this is where content is lost).
# --------------------------------------------------------------------------- #
def center_crop_window(img):
    """Return the center square sub-image, at native resolution (no resize).
    Everything outside this window is discarded by the crop-everything recipe."""
    _, H, W = img.shape
    s = min(H, W)
    top, left = (H - s) // 2, (W - s) // 2
    return img[:, top:top + s, left:left + s]


def to_square_28(img, size=28):
    return F.interpolate(center_crop_window(img)[None], size=(size, size),
                         mode="bilinear", align_corners=False)[0]


# --------------------------------------------------------------------------- #
# Strategy B — bucketing: resize the WHOLE image to its bucket, nothing cropped.
# --------------------------------------------------------------------------- #
def nearest_bucket(img):
    _, H, W = img.shape
    ar = W / H
    return min(BUCKETS, key=lambda k: abs(BUCKETS[k][1] / BUCKETS[k][0] - ar))


def resize_to_bucket(img):
    name = nearest_bucket(img)
    H, W = BUCKETS[name]
    return F.interpolate(img[None], size=(H, W), mode="bilinear",
                         align_corners=False)[0], name


def letterbox(img, box=48):
    """Fit an image into a box x box frame by shrinking to fit + padding — used
    only to display bucketed images of different shapes side by side."""
    _, H, W = img.shape
    scale = box / max(H, W)
    h, w = max(1, round(H * scale)), max(1, round(W * scale))
    small = F.interpolate(img[None], size=(h, w), mode="bilinear", align_corners=False)[0]
    out = torch.full((1, box, box), -1.0)
    top, left = (box - h) // 2, (box - w) // 2
    out[:, top:top + h, left:left + w] = small
    return out


# --------------------------------------------------------------------------- #
# The bucket sampler: batches that never mix shapes.
# --------------------------------------------------------------------------- #
def bucket_batches(items, batch_size=32):
    groups = defaultdict(list)
    for idx, (img, _) in enumerate(items):
        groups[nearest_bucket(img)].append(idx)
    batches = []
    for name, idxs in groups.items():
        for s in range(0, len(idxs), batch_size):
            batches.append((name, idxs[s:s + batch_size]))
    return groups, batches


def crop_retention(img):
    """Fraction of the digit's ink that survives the center square crop. This
    is measured at native resolution, so it reflects *content lost to cropping*
    only — not the later resize. Bucketing crops nothing, so its retention is
    always 1.0."""
    total = ink(img) + 1e-6
    return ink(center_crop_window(img)) / total


def plot_examples(items, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def show(ax, img, title, color="#0b0b0b"):
        a = ((img[0].clamp(-1, 1) + 1) * 127.5).byte().numpy()
        ax.imshow(a, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=8, color=color)
        ax.set_xticks([]); ax.set_yticks([])

    picks = [items[0], items[1], items[2]]  # one landscape, portrait, square
    fig, axes = plt.subplots(3, 3, figsize=(6.6, 6.2), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    for r, (img, y) in enumerate(picks):
        buck, name = resize_to_bucket(img)
        ret = crop_retention(img)
        show(axes[r][0], letterbox(img), f"original {tuple(img.shape[1:])}")
        show(axes[r][1], to_square_28(img), f"center-crop  ({ret:.0%} kept)",
             "#e34948" if ret < 0.9 else "#0b0b0b")
        show(axes[r][2], letterbox(buck), f"bucket '{name}'  (100% kept)", "#1baf7a")
    fig.suptitle("Same image, two preprocessings. Center-crop (middle) throws away "
                 "off-center content;\nbucketing (right) keeps all of it.",
                 fontsize=10, color="#0b0b0b")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_retention(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
    import plot_style as ps

    cats = list(rows.keys())
    crop = [rows[c] for c in cats]
    fig, ax = ps.new_axes(7.2, 4.0)
    x = np.arange(len(cats)); w = 0.38
    ax.bar(x - w / 2, crop, w, color=ps.SERIES[2], label="center-crop")
    ax.bar(x + w / 2, [1.0] * len(cats), w, color=ps.SERIES[1], label="bucketing")
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylim(0, 1.08)
    for i, v in enumerate(crop):
        ax.text(i - w / 2, v + 0.02, f"{v:.0%}", ha="center",
                color=ps.INK_SECONDARY, fontsize=9)
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Content retained: fraction of digit 'ink' kept by each strategy",
              "image shape", "ink kept / original ink", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--n", type=int, default=600)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    items = build_variable_aspect(args.data_dir, args.n)
    groups, batches = bucket_batches(items, batch_size=32)
    print("bucket sizes:", {k: len(v) for k, v in groups.items()})
    print(f"{len(batches)} shape-homogeneous batches "
          f"(vs. crop: all {len(items)} images forced to one 28x28 shape)")

    shape_name = {(24, 72): "landscape", (72, 24): "portrait", (40, 40): "square"}
    per = defaultdict(list)
    for img, _ in items:
        per[shape_name[tuple(img.shape[1:])]].append(crop_retention(img))
    rows = {c: float(np.mean(v)) for c, v in per.items()}

    plot_examples(items, OUT / "examples.png")
    plot_retention(rows, OUT / "retention.png")

    lines = ["shape,crop_ink_kept,bucket_ink_kept"]
    for c, v in rows.items():
        lines.append(f"{c},{v:.3f},1.000")
    (OUT / "results.csv").write_text("\n".join(lines) + "\n")

    print("\nink retained (center-crop vs bucketing):")
    for c, v in rows.items():
        print(f"  {c:9s} crop {v:.0%}  bucket 100%")
    print(f"wrote figures + {OUT/'results.csv'}")


if __name__ == "__main__":
    main()
