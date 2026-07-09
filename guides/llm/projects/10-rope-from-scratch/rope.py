"""RoPE (Rotary Position Embedding) from scratch — and the proof that it encodes
*relative* position.

RoPE injects position by rotating each token's query and key vectors by an angle
proportional to their position. The magic: because rotations compose, the dot
product ⟨RoPE(q, m), RoPE(k, n)⟩ ends up depending only on the offset (m − n), not
on the absolute positions m and n. We implement RoPE with the half-rotation trick
and verify that property numerically and visually.

    python rope.py      # ~15s on CPU
"""

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-train-a-bpe-from-scratch"))
OUT = HERE / "outputs"


def rope_tables(d, T, base=10000.0):
    """cos/sin lookup of shape (T, d) for the half-rotation form."""
    inv_freq = 1.0 / base ** (torch.arange(0, d, 2).float() / d)   # (d/2,)
    angles = torch.outer(torch.arange(T).float(), inv_freq)         # (T, d/2)
    cos = torch.cos(angles).repeat(1, 2)                            # (T, d)
    sin = torch.sin(angles).repeat(1, 2)
    return cos, sin


def rotate_half(x):
    """The half-rotation trick: [-x2, x1] pairs each dim with its partner d/2 away."""
    d = x.size(-1)
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    """Rotate x (..., d) by the position-dependent angles in cos/sin (..., d)."""
    return x * cos + rotate_half(x) * sin


def verify_relative():
    """⟨RoPE(q,m), RoPE(k,n)⟩ must depend only on (m-n)."""
    torch.manual_seed(0)
    d, T = 64, 256
    cos, sin = rope_tables(d, T)
    q0, k0 = torch.randn(d), torch.randn(d)              # fixed content vectors
    def dot(m, n):
        qm = apply_rope(q0, cos[m], sin[m])
        kn = apply_rope(k0, cos[n], sin[n])
        return (qm @ kn).item()
    # same offset at different absolute positions -> same score
    checks = [((5, 3), (10, 8), (100, 98)), ((0, 0), (50, 50), (200, 200)),
              ((30, 10), (130, 110), (200, 180))]
    worst = 0.0
    for group in checks:
        vals = [dot(m, n) for m, n in group]
        spread = max(vals) - min(vals)
        worst = max(worst, spread)
        print(f"  offset {group[0][0]-group[0][1]:>4}: "
              f"scores {[f'{v:.4f}' for v in vals]}  spread {spread:.2e}")
    print(f"max spread across absolute positions (same offset): {worst:.2e}  "
          f"{'OK — relative only' if worst < 1e-3 else 'FAIL'}")
    assert worst < 1e-3
    return worst


def plot_toeplitz(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style as ps
    d, T = 64, 48
    cos, sin = rope_tables(d, T)
    torch.manual_seed(1)
    q0, k0 = torch.randn(d), torch.randn(d)
    Q = apply_rope(q0.expand(T, d), cos, sin)            # q at every position
    K = apply_rope(k0.expand(T, d), cos, sin)
    M = (Q @ K.T).detach().numpy()                       # (T, T) dot products
    fig, ax = plt.subplots(figsize=(5.6, 5.0), dpi=120)
    fig.patch.set_facecolor(ps.SURFACE)
    im = ax.imshow(M, cmap="coolwarm")
    ax.set_xlabel("key position n", fontsize=9, color=ps.INK_SECONDARY)
    ax.set_ylabel("query position m", fontsize=9, color=ps.INK_SECONDARY)
    ax.set_title("⟨RoPE(q,m), RoPE(k,n)⟩ is constant along diagonals\n"
                 "(depends only on m − n)", fontsize=11, color=ps.INK, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_collapse(path):
    """Same content, several absolute base positions -> one curve vs offset."""
    import plot_style as ps
    d, T = 64, 256
    cos, sin = rope_tables(d, T)
    torch.manual_seed(2)
    q0, k0 = torch.randn(d), torch.randn(d)
    fig, ax = ps.new_axes(7.2, 4.2)
    offsets = torch.arange(0, 64)
    for i, base_pos in enumerate([0, 40, 120]):
        vals = []
        for delta in offsets:
            m = base_pos + delta
            qm = apply_rope(q0, cos[m], sin[m])
            kn = apply_rope(k0, cos[base_pos], sin[base_pos])
            vals.append((qm @ kn).item())
        ax.plot(offsets.numpy(), vals, color=ps.SERIES[i], linewidth=1.6,
                label=f"key at position {base_pos}", alpha=0.8)
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Three key positions, identical curve — RoPE sees only distance",
              "relative offset (m − n)", "attention score ⟨q, k⟩", path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("verifying RoPE encodes relative position only:")
    worst = verify_relative()
    plot_toeplitz(OUT / "toeplitz.png")
    plot_collapse(OUT / "relative_collapse.png")
    (OUT / "results.csv").write_text(
        "metric,value\n"
        f"relative_position_only,True\n"
        f"max_score_spread_same_offset,{worst:.2e}\n")
    print(f"wrote figures + {OUT/'results.csv'}")


if __name__ == "__main__":
    main()
