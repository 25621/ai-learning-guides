"""Activation checkpointing: trade compute for memory, and measure the exact deal.

The forward pass saves intermediate activations so the backward pass can reuse
them. Activation checkpointing throws most of them away and *recomputes* them
during the backward instead. That saves memory (fewer tensors kept alive) at the
cost of time (an extra partial forward).

On CPU we can't call `torch.cuda.max_memory_allocated`, so we measure the saved
activations *exactly* with `torch.autograd.graph.saved_tensors_hooks`: every
tensor stashed for the backward pass is intercepted and its bytes summed. We do
this with and without checkpointing, across a sweep of sequence lengths, and also
time the step so the compute cost is visible.

    python checkpointing.py       # ~1 min on CPU

Reuses the GPT skeleton from project 08.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint as ckpt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "08-nanogpt-reproduction"))
sys.path.insert(0, str(HERE.parent / "01-train-a-bpe-from-scratch"))
from model import Config, GPT   # noqa: E402

OUT = HERE / "outputs"


class CheckpointGPT(GPT):
    """Same GPT, but each transformer block is wrapped in activation checkpointing.

    Only the block *inputs* are kept for the backward pass; everything inside the
    block (attention scores, MLP hidden states) is recomputed on the way back.
    """
    def __init__(self, cfg, use_checkpoint):
        super().__init__(cfg)
        self.use_checkpoint = use_checkpoint

    def forward(self, idx, targets=None):
        import torch.nn.functional as F
        B, T = idx.shape
        x = self.tok(idx)
        rope = self._rope_for(T, idx.device) if self.cfg.pos == "rope" else None
        for blk in self.blocks:
            if self.use_checkpoint:
                x = ckpt.checkpoint(blk, x, rope, use_reentrant=False)
            else:
                x = blk(x, rope)
        logits = self.head(self.norm_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def saved_activation_bytes(model, x, y):
    """Sum the bytes of every tensor saved for the backward pass (= activation memory)."""
    total = [0]
    seen = set()

    def pack(t):
        key = t.data_ptr()
        if key not in seen:
            seen.add(key)
            total[0] += t.numel() * t.element_size()
        return t

    def unpack(t):
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        _, loss = model(x, y)
    loss.backward()
    model.zero_grad(set_to_none=True)
    return total[0]


def time_step(model, x, y, reps=5):
    for _ in range(2):
        model.zero_grad(set_to_none=True); _, l = model(x, y); l.backward()
    t0 = time.time()
    for _ in range(reps):
        model.zero_grad(set_to_none=True); _, l = model(x, y); l.backward()
    return (time.time() - t0) / reps


def main():
    torch.manual_seed(0); torch.set_num_threads(12)
    OUT.mkdir(parents=True, exist_ok=True)
    batch = 8
    seqs = [128, 256, 512, 768]
    rows = []
    for T in seqs:
        cfg = Config(vocab_size=65, n_layer=6, n_head=8, n_embd=256, block_size=T)
        x = torch.randint(0, 65, (batch, T)); y = torch.randint(0, 65, (batch, T))
        m_off = CheckpointGPT(cfg, use_checkpoint=False)
        m_on = CheckpointGPT(cfg, use_checkpoint=True)
        m_on.load_state_dict(m_off.state_dict())
        a_off = saved_activation_bytes(m_off, x, y)
        a_on = saved_activation_bytes(m_on, x, y)
        t_off = time_step(m_off, x, y)
        t_on = time_step(m_on, x, y)
        rows.append((T, a_off, a_on, t_off, t_on))
        print(f"seq {T:4d} | activations off {a_off/1e6:7.1f} MB  on {a_on/1e6:6.1f} MB "
              f"({a_off/a_on:4.1f}x less) | step off {t_off*1e3:6.0f} ms  on {t_on*1e3:6.0f} ms "
              f"(+{100*(t_on/t_off-1):.0f}%)")

    rows = np.array(rows, dtype=float)
    plot(rows)
    with open(OUT / "results.csv", "w") as f:
        f.write("seq_len,act_mb_off,act_mb_on,mem_saving_x,step_ms_off,step_ms_on,time_overhead_pct\n")
        for T, ao, an, to, tn in rows:
            f.write(f"{int(T)},{ao/1e6:.1f},{an/1e6:.1f},{ao/an:.2f},"
                    f"{to*1e3:.0f},{tn*1e3:.0f},{100*(tn/to-1):.0f}\n")
    print(f"wrote {OUT/'checkpointing.png'} + results.csv")


def plot(rows):
    import plot_style as ps
    import matplotlib.pyplot as plt
    seqs = rows[:, 0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.2), dpi=110)
    for ax in (ax1, ax2):
        ax.set_facecolor(ps.SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(ps.BASELINE)
        ax.tick_params(colors=ps.INK_MUTED, labelsize=9)
        ax.grid(True, color=ps.GRID, linewidth=0.8); ax.set_axisbelow(True)
    fig.patch.set_facecolor(ps.SURFACE)

    ax1.plot(seqs, rows[:, 1] / 1e6, "-o", color=ps.SERIES[2], label="off (store all)")
    ax1.plot(seqs, rows[:, 2] / 1e6, "-o", color=ps.SERIES[1], label="on (checkpoint)")
    ax1.set_title("Activation memory saved", color=ps.INK, fontsize=11, loc="left", pad=10)
    ax1.set_xlabel("sequence length", color=ps.INK_SECONDARY, fontsize=10)
    ax1.set_ylabel("saved activations (MB)", color=ps.INK_SECONDARY, fontsize=10)
    ax1.legend(frameon=False, fontsize=9)

    ax2.plot(seqs, rows[:, 3] * 1e3, "-o", color=ps.SERIES[2], label="off")
    ax2.plot(seqs, rows[:, 4] * 1e3, "-o", color=ps.SERIES[1], label="on (+recompute)")
    ax2.set_title("Step time cost of recomputation", color=ps.INK, fontsize=11, loc="left", pad=10)
    ax2.set_xlabel("sequence length", color=ps.INK_SECONDARY, fontsize=10)
    ax2.set_ylabel("step time (ms)", color=ps.INK_SECONDARY, fontsize=10)
    ax2.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT / "checkpointing.png", facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
