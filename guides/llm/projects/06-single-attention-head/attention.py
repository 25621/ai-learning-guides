"""A single attention head from scratch — and why the 1/√d scaling matters.

Attention is a weighted average where each token decides how much to listen to
every earlier token: score every query against every key, softmax the scores into
weights, and mix the values. A causal mask hides the future. We implement it by
hand, check it matches PyTorch's fused `F.scaled_dot_product_attention` to within
floating-point noise, then visualize the causal attention map and demonstrate why
the scores are divided by √d.

    python attention.py      # ~20s on CPU
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-train-a-bpe-from-scratch"))
OUT = HERE / "outputs"


def attention(q, k, v, causal=True):
    """Scaled dot-product attention, written out. Shapes: (B, T, d)."""
    d = q.size(-1)
    scores = q @ k.transpose(-2, -1) / (d ** 0.5)        # (B, T, T) — QKᵀ / √d
    if causal:
        T = q.size(-2)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))  # hide the future
    weights = F.softmax(scores, dim=-1)                   # rows sum to 1
    return weights @ v, weights


def verify():
    torch.manual_seed(0)
    B, T, d = 2, 16, 32
    q, k, v = (torch.randn(B, T, d) for _ in range(3))
    for causal in (False, True):
        mine, _ = attention(q, k, v, causal=causal)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        diff = (mine - ref).abs().max().item()
        print(f"causal={causal!s:5}  max|mine - F.sdpa| = {diff:.2e}  "
              f"{'OK' if diff < 1e-5 else 'MISMATCH'}")
        assert diff < 1e-5
    return B, T, d


def plot_attention_map(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style as ps
    text = "To be, or not"
    torch.manual_seed(1)
    T = len(text)
    d = 24
    x = torch.randn(1, T, d)
    Wq, Wk = torch.randn(d, d) * 0.5, torch.randn(d, d) * 0.5
    q, k = x @ Wq, x @ Wk
    _, w = attention(q, k, x, causal=True)
    W = w[0].detach().numpy()
    fig, ax = plt.subplots(figsize=(5.6, 5.0), dpi=120)
    fig.patch.set_facecolor(ps.SURFACE)
    im = ax.imshow(W, cmap="magma", vmin=0)
    ax.set_xticks(range(T)); ax.set_xticklabels(list(text), fontsize=9)
    ax.set_yticks(range(T)); ax.set_yticklabels(list(text), fontsize=9)
    ax.set_xlabel("key position (attended to)", fontsize=9, color=ps.INK_SECONDARY)
    ax.set_ylabel("query position (attending)", fontsize=9, color=ps.INK_SECONDARY)
    ax.set_title("Causal attention map: upper triangle is masked to zero",
                 fontsize=11, color=ps.INK, loc="left", pad=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_scale_demo(path):
    """Why √d: without it, large d saturates the softmax onto one token."""
    import plot_style as ps
    dims = [4, 8, 16, 32, 64, 128, 256]
    torch.manual_seed(0)
    scaled_max, unscaled_max = [], []
    for d in dims:
        q, k = torch.randn(64, d), torch.randn(64, d)
        s = q @ k.transpose(0, 1)
        unscaled_max.append(F.softmax(s, -1).max(-1).values.mean().item())
        scaled_max.append(F.softmax(s / d ** 0.5, -1).max(-1).values.mean().item())
    fig, ax = ps.new_axes(7.0, 4.2)
    ax.plot(dims, unscaled_max, "-o", color=ps.SERIES[2], label="no √d scaling (saturates)")
    ax.plot(dims, scaled_max, "-o", color=ps.SERIES[0], label="with √d scaling (stable)")
    ax.set_xscale("log", base=2)
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Why divide by √d: unscaled softmax collapses onto one token",
              "head dimension d", "avg. peak attention weight", path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("verifying against F.scaled_dot_product_attention:")
    verify()
    plot_attention_map(OUT / "attention_map.png")
    plot_scale_demo(OUT / "scale_demo.png")

    # numeric record of the saturation effect at d=256
    torch.manual_seed(0)
    q, k = torch.randn(64, 256), torch.randn(64, 256)
    s = q @ k.transpose(0, 1)
    (OUT / "results.csv").write_text(
        "metric,value\n"
        f"verified_vs_pytorch,True\n"
        f"peak_weight_unscaled_d256,{F.softmax(s,-1).max(-1).values.mean():.3f}\n"
        f"peak_weight_scaled_d256,{F.softmax(s/256**0.5,-1).max(-1).values.mean():.3f}\n")
    print(f"wrote figures + {OUT/'results.csv'}")


if __name__ == "__main__":
    main()
