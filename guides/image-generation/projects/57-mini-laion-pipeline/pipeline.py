"""A miniature LAION-style data-curation pipeline on a toy 'web scrape'.

Real pipelines take a billion noisy image-URL/alt-text pairs and boil them down
to a clean training shard. The noise is always the same four problems, and this
script reproduces each on a small MNIST-derived scrape so you can watch the
funnel shrink:

    raw scrape
      -> [dedup]      drop near-duplicate images (perceptual hash)
      -> [aesthetic]  drop low-quality / degraded images
      -> [align]      drop pairs whose caption doesn't match the image (CLIP)
      -> [recaption]  rewrite weak alt-text into rich synthetic captions
      -> clean shard

Everything is CPU-only and finishes in well under a minute after the one-off
classifier training. Run:

    python pipeline.py --data-dir data
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms

from classifier import class_probs, train_classifier

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
DIGIT_WORDS = ["zero", "one", "two", "three", "four",
               "five", "six", "seven", "eight", "nine"]


# --------------------------------------------------------------------------- #
# Build a toy 'web scrape': real pairs plus the four kinds of junk.
# --------------------------------------------------------------------------- #
def build_scrape(data_dir, n_clean=360, n_mismatch=60, n_degraded=45, seed=0):
    rng = np.random.default_rng(seed)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    ds = datasets.MNIST(str(data_dir), train=False, download=True, transform=tf)

    # Disjoint index pools so each junk kind is caught by exactly one stage:
    # mismatch/degraded are their OWN unique images (not copies of a good one),
    # so only the align / aesthetic stage removes them — not dedup.
    perm = rng.permutation(len(ds))
    good_ix = perm[:n_clean]
    mis_ix = perm[n_clean:n_clean + n_mismatch]
    deg_ix = perm[n_clean + n_mismatch:n_clean + n_mismatch + n_degraded]

    def load(i):
        x, y = ds[int(i)]
        return x, int(y)

    records = []
    goods = [load(i) for i in good_ix]
    for x, y in goods:
        records.append(dict(img=x, true=y, claimed=y, kind="good"))

    # (1) near-duplicates: re-uploads of a good image with faint compression
    # noise and a brightness shift — the exact case perceptual hashing catches.
    # (Cropped/shifted re-uploads need embedding-based dedup; out of scope.)
    dup_src = rng.choice(n_clean, size=int(0.15 * n_clean), replace=False)
    for i in dup_src:
        x, y = goods[i]
        dup = (x + 0.02 * torch.randn_like(x) + 0.05).clamp(-1, 1)
        records.append(dict(img=dup, true=y, claimed=y, kind="dup"))

    # (2) caption mismatches: a clean, unique image whose alt-text names the
    # wrong digit — invisible to dedup and aesthetics; only alignment catches it.
    for i in mis_ix:
        x, y = load(i)
        wrong = int((y + rng.integers(1, 10)) % 10)
        records.append(dict(img=x, true=y, claimed=wrong, kind="mismatch"))

    # (3) low-aesthetic: heavy noise / washed-out contrast — looks like junk.
    for i in deg_ix:
        x, y = load(i)
        deg = (0.35 * x + 0.8 * torch.randn_like(x)).clamp(-1, 1)
        records.append(dict(img=deg, true=y, claimed=y, kind="degraded"))

    rng.shuffle(records)
    return records


# --------------------------------------------------------------------------- #
# Stage 1 — perceptual-hash deduplication.
# --------------------------------------------------------------------------- #
def average_hash(img, size=16):
    """aHash: downsample to size x size, threshold at the mean. Near-duplicate
    images collapse to the same (or very close) hash. 16x16 (256 bits) keeps
    distinct digits far apart; a coarse 8x8 hash over-merges same-class digits."""
    x = torch.nn.functional.interpolate(img[None], size=(size, size), mode="area")[0, 0]
    return (x > x.mean()).flatten()


def dedup(records, max_hamming=2):
    kept, hashes = [], []
    for r in records:
        h = average_hash(r["img"])
        if any(int((h ^ hp).sum()) <= max_hamming for hp in hashes):
            continue  # too close to something we already kept
        hashes.append(h)
        kept.append(r)
    return kept


# --------------------------------------------------------------------------- #
# Stage 2 — aesthetic filtering (a cheap sharpness/contrast proxy).
# --------------------------------------------------------------------------- #
def aesthetic_score(img):
    """Stand-in for a learned aesthetic predictor: reward contrast and clean
    edges, both of which collapse under heavy noise / low contrast."""
    contrast = img.std().item()
    gx = (img[:, :, 1:] - img[:, :, :-1]).abs().mean().item()
    gy = (img[:, 1:, :] - img[:, :-1, :]).abs().mean().item()
    # Noise inflates raw edge energy, so penalise total-variation that is
    # high everywhere (a noise signature) relative to contrast.
    tv = gx + gy
    return contrast - 1.6 * tv


def aesthetic_filter(records, thresh):
    return [r for r in records if aesthetic_score(r["img"]) >= thresh]


# --------------------------------------------------------------------------- #
# Stage 3 — CLIP-style caption/image alignment filter.
# --------------------------------------------------------------------------- #
def align_filter(records, clf, thresh=0.5):
    imgs = torch.stack([r["img"] for r in records])
    probs = class_probs(clf, imgs)
    kept = []
    for r, p in zip(records, probs):
        r["align"] = float(p[r["claimed"]])   # P(claimed digit | image)
        if r["align"] >= thresh:
            kept.append(r)
    return kept


# --------------------------------------------------------------------------- #
# Stage 4 — synthetic recaptioning.
# --------------------------------------------------------------------------- #
def boldness(img):
    return (img > 0.0).float().mean().item()  # fraction of 'ink' pixels


def centering(img):
    ys, xs = torch.where(img[0] > 0.0)
    if len(xs) == 0:
        return "empty"
    cx, cy = xs.float().mean().item() / 27, ys.float().mean().item() / 27
    return "centered" if 0.35 < cx < 0.65 and 0.35 < cy < 0.65 else "off-center"


def recaption(records, clf):
    imgs = torch.stack([r["img"] for r in records])
    probs = class_probs(clf, imgs)
    for r, p in zip(records, probs):
        pred = int(p.argmax())
        weight = "bold" if boldness(r["img"]) > 0.18 else "thin"
        place = centering(r["img"])
        r["alt_text"] = f"a photo of the number {DIGIT_WORDS[r['claimed']]}"
        r["synthetic"] = (f"a {weight}, {place} handwritten digit "
                          f"'{pred}' in white on a black background")
    return records


# --------------------------------------------------------------------------- #
# Figures.
# --------------------------------------------------------------------------- #
def to_img(x):
    return ((x[0].clamp(-1, 1) + 1) * 127.5).round().byte().numpy()


def plot_funnel(counts, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
    import plot_style as ps

    stages = list(counts.keys())
    vals = list(counts.values())
    fig, ax = ps.new_axes(7.4, 4.0)
    bars = ax.barh(range(len(stages)), vals, color=ps.SERIES[0], height=0.62)
    bars[-1].set_color(ps.SERIES[1])
    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels(stages)
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.01, i, str(v), va="center",
                color=ps.INK_SECONDARY, fontsize=10)
    ax.set_xlim(0, max(vals) * 1.12)
    ps.finish(fig, ax, "The curation funnel: pairs surviving each stage",
              "pairs remaining", "", path)


def plot_grid(records, path, title, cols=12, rows=4, tag=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = min(cols * rows, len(records))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 0.75, rows * 0.85), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    color = {"good": "#1baf7a", "dup": "#e0873a", "mismatch": "#e34948",
             "degraded": "#8a5cd0"}
    for ax in axes.flat:
        ax.axis("off")
    for k in range(n):
        r = records[k]
        ax = axes.flat[k]
        ax.imshow(to_img(r["img"]), cmap="gray", vmin=0, vmax=255)
        if tag:
            ax.set_title(r["kind"], color=color.get(r["kind"], "#333"),
                         fontsize=7, pad=1)
    fig.suptitle(title, color="#0b0b0b", fontsize=12, x=0.5, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, facecolor="#fcfcfb", bbox_inches="tight")
    import matplotlib.pyplot as plt2
    plt2.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    print("training the stand-in CLIP/VLM classifier ...")
    clf = train_classifier(args.data_dir)

    scrape = build_scrape(args.data_dir)
    counts = {"raw scrape": len(scrape)}
    print(f"raw scrape: {len(scrape)} pairs")

    plot_grid(scrape[:48], OUT / "raw_scrape.png",
              "Raw scrape — good pairs plus four kinds of junk")

    kept = dedup(scrape)
    counts["after dedup"] = len(kept)
    print(f"after dedup: {len(kept)}")

    # Calibrate the aesthetic threshold from the good-image score distribution.
    good_scores = np.array([aesthetic_score(r["img"]) for r in kept if r["kind"] == "good"])
    thresh = float(np.percentile(good_scores, 5))
    kept = aesthetic_filter(kept, thresh)
    counts["after aesthetic"] = len(kept)
    print(f"after aesthetic (thresh={thresh:.3f}): {len(kept)}")

    kept = align_filter(kept, clf, thresh=0.5)
    counts["after align (CLIP)"] = len(kept)
    print(f"after align: {len(kept)}")

    kept = recaption(kept, clf)
    counts["clean shard"] = len(kept)

    plot_funnel(counts, OUT / "funnel.png")
    plot_grid(kept[:48], OUT / "clean_shard.png",
              "Clean shard — deduplicated, high-quality, aligned", tag=False)

    # Composition report: how much of each junk type was removed.
    def frac(recs, k):
        return sum(1 for r in recs if r["kind"] == k)
    lines = ["stage,good,dup,mismatch,degraded,total"]
    for name, recs in [("raw", scrape)]:
        lines.append(f"{name},{frac(recs,'good')},{frac(recs,'dup')},"
                     f"{frac(recs,'mismatch')},{frac(recs,'degraded')},{len(recs)}")
    lines.append(f"clean,{frac(kept,'good')},{frac(kept,'dup')},"
                 f"{frac(kept,'mismatch')},{frac(kept,'degraded')},{len(kept)}")
    (OUT / "composition.csv").write_text("\n".join(lines) + "\n")

    # A handful of alt-text -> synthetic caption rewrites.
    ex = [r for r in kept if r["kind"] == "good"][:6]
    cap_lines = ["| original alt-text | synthetic caption |",
                 "|---|---|"]
    for r in ex:
        cap_lines.append(f"| {r['alt_text']} | {r['synthetic']} |")
    (OUT / "captions.md").write_text("\n".join(cap_lines) + "\n")

    print("\ncomposition:")
    print("\n".join(lines))
    print(f"\nkept {len(kept)}/{len(scrape)} = {len(kept)/len(scrape):.0%} of the raw scrape")
    print(f"wrote {OUT/'composition.csv'} and {OUT/'captions.md'}")


if __name__ == "__main__":
    main()
