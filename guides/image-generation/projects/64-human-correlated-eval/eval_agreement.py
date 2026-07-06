"""How much do raters agree — and does an automatic metric track human taste?

The gold standard for judging image generators is still human preference, but
humans are slow and expensive, so the field leans on (a) LLM-as-judge and
(b) automatic metrics. Both are only useful to the extent they *agree with
humans*. This project runs the exact agreement analysis you would run on a real
rating study — the only thing faked is the raters themselves, because we have
no humans or GPT-4V on a CPU.

Setup, kept deliberately honest:
  - 120 'generated outputs' = real MNIST digits corrupted by varying amounts of
    noise/blur, so each has a genuine, controllable perceptual quality.
  - the hidden 'true quality' of each output is how confidently a strong CNN
    classifier still reads the intended digit (a clean digit reads at ~1.0, a
    noisy blob near chance).
  - three 'human' raters and three 'LLM-judge' raters each score every output
    1-5. Humans are near-unbiased but noisy; the simulated LLM judge is a bit
    more consistent (lower variance) but carries a systematic leniency bias —
    a documented failure mode of LLM judges.
  - an automatic metric ('AutoScore', a CLIPScore/aesthetic stand-in) reads the
    quality signal through its own noise.

We then compute the agreement statistics that matter: quadratic-weighted
Cohen's kappa and Spearman correlation, within and across rater groups, plus
the automatic-metric-vs-human correlation.

    python eval_agreement.py --data-dir data      # ~30s on CPU
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "58-caption-ablation"))
from mnist_classifier import load_or_train  # noqa: E402

OUT = HERE / "outputs"


def make_outputs(data_dir, clf, n=120, seed=0):
    """Real digits at a spread of corruption levels; true quality = classifier
    confidence in the correct label."""
    rng = np.random.default_rng(seed)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    ds = datasets.MNIST(str(data_dir), train=False, download=True, transform=tf)
    idxs = rng.permutation(len(ds))[:n]
    imgs, quality = [], []
    with torch.no_grad():
        for k, i in enumerate(idxs):
            x, y = ds[int(i)]
            sigma = (k / n) * 1.4                      # sweep clean -> very noisy
            xc = (x + sigma * torch.randn_like(x)).clamp(-1, 1)
            if sigma > 0.7:                            # heavy corruption also blurs
                xc = F.avg_pool2d(xc[None], 3, 1, 1)[0]
            p = torch.softmax(clf(xc[None]), dim=1)[0, y].item()
            imgs.append(xc); quality.append(p)
    return torch.stack(imgs), np.array(quality)


def rate(quality, bias, noise, rng, scale=5):
    """Map a latent quality in [0,1] to a 1..scale rating with bias + noise."""
    latent = np.clip(quality + bias + noise * rng.standard_normal(len(quality)), 0, 1)
    return np.clip(np.round(1 + latent * (scale - 1)), 1, scale).astype(int)


def quadratic_weighted_kappa(a, b, scale=5):
    """Cohen's kappa with quadratic weights — the standard ordinal-agreement
    coefficient. 1 = perfect, 0 = chance, <0 = worse than chance."""
    a, b = np.asarray(a) - 1, np.asarray(b) - 1
    O = np.zeros((scale, scale))
    for x, y in zip(a, b):
        O[x, y] += 1
    w = (np.arange(scale)[:, None] - np.arange(scale)[None, :]) ** 2 / (scale - 1) ** 2
    ha, hb = O.sum(1), O.sum(0)
    E = np.outer(ha, hb) / O.sum()
    return 1 - (w * O).sum() / (w * E).sum()


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def mean_pairwise(mat, fn):
    g = mat.shape[0]
    vals = [fn(mat[i], mat[j]) for i in range(g) for j in range(i + 1, g)]
    return float(np.mean(vals))


def cross_group(A, B, fn):
    return float(np.mean([fn(a, b) for a in A for b in B]))


def plot_scatter(human_mean, llm_mean, auto, quality, path):
    import matplotlib
    matplotlib.use("Agg")
    sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
    import plot_style as ps

    fig, ax = ps.new_axes(6.6, 4.6)
    ax.scatter(human_mean, llm_mean, s=26, color=ps.SERIES[0], alpha=0.7,
               edgecolor="white", linewidth=0.4)
    lo, hi = 1, 5
    ax.plot([lo, hi], [lo, hi], color=ps.BASELINE, linewidth=1, linestyle="--")
    r = np.corrcoef(human_mean, llm_mean)[0, 1]
    ax.set_xlim(0.8, 5.2); ax.set_ylim(0.8, 5.2)
    ax.text(0.05, 0.92, f"Pearson r = {r:.2f}", transform=ax.transAxes,
            color=ps.INK_SECONDARY, fontsize=10)
    ps.finish(fig, ax, "Mean human rating vs. mean LLM-judge rating (each dot = one output)",
              "mean human rating (1-5)", "mean LLM-judge rating (1-5)", path)


def plot_bars(stats, path):
    import matplotlib
    matplotlib.use("Agg")
    sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
    import plot_style as ps

    labels = list(stats.keys()); vals = list(stats.values())
    fig, ax = ps.new_axes(7.2, 4.0)
    colors = [ps.SERIES[1] if "human-human" in l else
              ps.SERIES[0] if "human-LLM" in l else ps.SERIES[2] for l in labels]
    ax.bar(range(len(labels)), vals, color=colors, width=0.6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.0)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", color=ps.INK_SECONDARY, fontsize=9)
    ps.finish(fig, ax, "Agreement (quadratic-weighted kappa): the ceiling humans set, "
              "and how close the judges get", "", "kappa", path)


def plot_examples(imgs, quality, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = np.argsort(quality)
    picks = order[np.linspace(0, len(order) - 1, 8).astype(int)]
    fig, axes = plt.subplots(1, 8, figsize=(8, 1.5), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, k in zip(axes, picks):
        a = ((imgs[k, 0].clamp(-1, 1) + 1) * 127.5).byte().numpy()
        ax.imshow(a, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"q={quality[k]:.2f}", fontsize=8, color="#0b0b0b")
        ax.axis("off")
    fig.suptitle("Outputs span clean (high quality) to corrupted (low quality)",
                 fontsize=10, color="#0b0b0b")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(path, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--n", type=int, default=120)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    clf = load_or_train(args.data_dir,
                        HERE.parent / "58-caption-ablation/checkpoints/classifier.pt")
    imgs, quality = make_outputs(args.data_dir, clf, args.n)

    # A panel of 6 humans (near-unbiased, modest noise) that sets the ceiling,
    # 3 LLM judges (lower noise / more self-consistent, but a systematic
    # leniency bias), and AutoScore — a convenient but noisy automatic proxy.
    humans = np.stack([rate(quality, 0.0, 0.09, rng) for _ in range(6)])
    llms = np.stack([rate(quality, 0.15, 0.12, rng) for _ in range(3)])
    auto = np.clip(quality + 0.32 * rng.standard_normal(len(quality)), 0, 1)  # AutoScore

    human_mean = humans.mean(0); llm_mean = llms.mean(0)

    stats = {
        "human-human": mean_pairwise(humans, quadratic_weighted_kappa),
        "LLM-LLM": mean_pairwise(llms, quadratic_weighted_kappa),
        "human-LLM": cross_group(humans, llms, quadratic_weighted_kappa),
    }
    # Correlation (Spearman) of each judge against the human consensus. The fair
    # ceiling is split-half reliability: how well one half of the panel's mean
    # predicts the other half's mean — the best any external judge could hope
    # to do, since even humans don't perfectly agree.
    sp_hh = spearman(humans[:3].mean(0), humans[3:].mean(0))
    sp_llm = spearman(llm_mean, human_mean)
    sp_auto = spearman(auto, human_mean)

    plot_examples(imgs, quality, OUT / "examples.png")
    plot_scatter(human_mean, llm_mean, auto, quality, OUT / "scatter.png")
    plot_bars(stats, OUT / "agreement.png")

    lines = ["pair,quadratic_weighted_kappa"]
    for k, v in stats.items():
        lines.append(f"{k},{v:.3f}")
    lines += ["", "correlation,spearman_vs_human_consensus",
              f"human-human,{sp_hh:.3f}",
              f"LLM-judge,{sp_llm:.3f}",
              f"AutoScore,{sp_auto:.3f}"]
    (OUT / "results.csv").write_text("\n".join(lines) + "\n")

    print("\nquadratic-weighted kappa:")
    for k, v in stats.items():
        print(f"  {k:12s} {v:.3f}")
    print("\nSpearman vs. human consensus:")
    print(f"  human-human (ceiling) {sp_hh:.3f}")
    print(f"  LLM-judge             {sp_llm:.3f}")
    print(f"  AutoScore             {sp_auto:.3f}")
    print(f"\nwrote figures + {OUT/'results.csv'}")


if __name__ == "__main__":
    main()
