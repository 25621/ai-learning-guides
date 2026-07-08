"""Multi-head attention with a Grouped-Query (GQA) knob — verified against
PyTorch's `nn.MultiheadAttention`.

Multi-head attention runs several attention heads in parallel, each over its own
slice of the model dimension, then concatenates and projects back. GQA is one
small twist: let several query heads *share* a smaller set of key/value heads,
which shrinks the KV cache that dominates serving memory. We implement both, copy
weights out of `nn.MultiheadAttention` to prove our reshapes are exactly right,
and chart the cache savings.

    python mha.py      # ~20s on CPU
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-train-a-bpe-from-scratch"))
OUT = HERE / "outputs"


class CausalMHA(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads=None, bias=False):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads          # GQA when n_kv_heads < n_heads
        self.d_head = d_model // n_heads
        self.W_q = nn.Linear(d_model, n_heads * self.d_head, bias=bias)
        self.W_k = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=bias)
        self.W_v = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=bias)
        self.W_o = nn.Linear(n_heads * self.d_head, d_model, bias=bias)

    def forward(self, x, return_weights=False):
        B, T, _ = x.shape
        q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        if self.n_kv_heads != self.n_heads:              # broadcast KV heads to Q heads
            rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        if return_weights:
            scores = q @ k.transpose(-2, -1) / self.d_head ** 0.5
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
            w = F.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)
            y = w @ v
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            w = None
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return (self.W_o(y), w) if return_weights else self.W_o(y)

    def kv_cache_bytes(self, seq_len, n_layers, dtype_bytes=2):
        """Bytes to cache K and V for one sequence (the serving-memory bottleneck)."""
        return 2 * self.n_kv_heads * self.d_head * seq_len * n_layers * dtype_bytes


def verify_vs_pytorch():
    """Copy nn.MultiheadAttention's weights into our full-MHA and confirm equality."""
    torch.manual_seed(0)
    B, T, d, h = 2, 12, 32, 4
    ref = nn.MultiheadAttention(d, h, bias=False, batch_first=True)
    mine = CausalMHA(d, h, bias=False)
    wq, wk, wv = ref.in_proj_weight.chunk(3, dim=0)      # (3d, d) -> three (d, d)
    with torch.no_grad():
        mine.W_q.weight.copy_(wq); mine.W_k.weight.copy_(wk); mine.W_v.weight.copy_(wv)
        mine.W_o.weight.copy_(ref.out_proj.weight)
    x = torch.randn(B, T, d)
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
    ref_out, _ = ref(x, x, x, attn_mask=causal, need_weights=False)
    my_out = mine(x)
    diff = (ref_out - my_out).abs().max().item()
    print(f"max|mine - nn.MultiheadAttention| = {diff:.2e}  "
          f"{'OK' if diff < 1e-5 else 'MISMATCH'}")
    assert diff < 1e-5
    return diff


def plot_heads(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style as ps
    text = "the cat sat on"
    torch.manual_seed(3)
    T, d, h = len(text), 32, 4
    mha = CausalMHA(d, h)
    x = torch.randn(1, T, d)
    _, w = mha(x, return_weights=True)                   # (1, h, T, T)
    W = w[0].detach().numpy()
    fig, axes = plt.subplots(1, h, figsize=(11, 3.0), dpi=115)
    fig.patch.set_facecolor(ps.SURFACE)
    for i, ax in enumerate(axes):
        ax.imshow(W[i], cmap="magma", vmin=0)
        ax.set_title(f"head {i+1}", fontsize=10, color=ps.INK)
        ax.set_xticks(range(T)); ax.set_xticklabels(list(text), fontsize=6)
        ax.set_yticks(range(T)); ax.set_yticklabels(list(text), fontsize=6)
    fig.suptitle("Each head attends over its own slice of the model dimension",
                 fontsize=12, color=ps.INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_kv_cache(path):
    import plot_style as ps
    d, h, n_layers, seq = 4096, 32, 32, 4096            # Llama-7B-ish serving config
    configs = [("MHA (32 kv)", 32), ("GQA-8", 8), ("GQA-4", 4), ("MQA (1 kv)", 1)]
    fig, ax = ps.new_axes(7.0, 4.2)
    vals = []
    for name, nkv in configs:
        m = CausalMHA(d, h, n_kv_heads=nkv)
        gb = m.kv_cache_bytes(seq, n_layers) / 1e9
        vals.append(gb)
    bars = ax.bar([c[0] for c in configs], vals,
                  color=[ps.SERIES[0], ps.SERIES[1], ps.SERIES[1], ps.SERIES[2]], width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f} GB", ha="center",
                va="bottom", fontsize=9, color=ps.INK_SECONDARY)
    ps.finish(fig, ax, "KV cache per 4k-token sequence (d=4096, 32 layers, fp16)",
              "attention variant", "KV cache (GB)", path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("verifying against nn.MultiheadAttention:")
    diff = verify_vs_pytorch()
    plot_heads(OUT / "per_head_maps.png")
    plot_kv_cache(OUT / "kv_cache.png")

    d, h, n_layers, seq = 4096, 32, 32, 4096
    rows = ["config,n_kv_heads,kv_cache_GB_4k,shrink_vs_mha"]
    base = CausalMHA(d, h, n_kv_heads=32).kv_cache_bytes(seq, n_layers)
    for name, nkv in [("MHA", 32), ("GQA-8", 8), ("GQA-4", 4), ("MQA", 1)]:
        b = CausalMHA(d, h, n_kv_heads=nkv).kv_cache_bytes(seq, n_layers)
        rows.append(f"{name},{nkv},{b/1e9:.3f},{base/b:.0f}x")
    (OUT / "results.csv").write_text("\n".join(rows) + "\n")
    print(f"verified reshapes (diff {diff:.1e}); KV-cache: MHA {base/1e9:.2f} GB -> "
          f"MQA {CausalMHA(d,h,n_kv_heads=1).kv_cache_bytes(seq,n_layers)/1e9:.2f} GB (32x smaller)")
    print(f"wrote figures + {OUT/'results.csv'}")


if __name__ == "__main__":
    main()
