"""Provoke codebook collapse with an oversized codebook, then cure it.

A plain VQ-VAE given a codebook far bigger than it needs tends to *collapse*: it
leans on a handful of entries and leaves the rest dead, capping how much detail
it can store. This project trains three tokenizers with the SAME oversized
1024-entry codebook and watches the usage fill in as we add fixes:

  1. plain VQ (loss-based)           -> collapses, few codes used
  2. + EMA codebook updates          -> better, but dead codes stay dead
  3. + EMA + dead-code re-init        -> nearly the whole codebook comes alive

'Perplexity' (effective number of codes in use) is the headline number; the
usage histograms make it visible.

Each config is trained in its own short invocation (robust to time limits):

    python collapse.py --data-dir data --config plain
    python collapse.py --data-dir data --config ema
    python collapse.py --data-dir data --config ema_reinit
    python collapse.py --plot            # combine the three into figures + csv

Reuses the shared encoder/decoder and quantizers from project 12 via sys.path.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "12-vq-vae-on-cifar-10"))
sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
from vq_lib import (Encoder, Decoder, VectorQuantizer, VectorQuantizerEMA,  # noqa: E402
                    perplexity, cifar_loader, infinite, cifar_tensor)

OUT = HERE / "outputs"
K, D = 1024, 32
CONFIGS = {
    "plain": ("plain VQ", lambda: VectorQuantizer(K=K, D=D)),
    "ema": ("EMA", lambda: VectorQuantizerEMA(K=K, D=D, reinit=False)),
    "ema_reinit": ("EMA + reinit", lambda: VectorQuantizerEMA(K=K, D=D, reinit=True, dead_after=64)),
}


class VQVAE(torch.nn.Module):
    def __init__(self, vq):
        super().__init__()
        self.enc = Encoder(D=D, ch=64)
        self.vq = vq
        self.dec = Decoder(D=D, ch=64)

    def forward(self, x):
        z_q, vq_loss, idx = self.vq(self.enc(x))
        return self.dec(z_q), vq_loss, idx


def run_config(key, data_dir, steps):
    name, make_vq = CONFIGS[key]
    torch.manual_seed(0)
    model = VQVAE(make_vq())
    loader = infinite(cifar_loader(data_dir, 128, True))
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    curve = []
    t0 = time.time()
    for step in range(1, steps + 1):
        (x,) = next(loader)
        xhat, vq_loss, idx = model(x)
        loss = F.mse_loss(xhat, x) + vq_loss
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            pplx, _ = perplexity(idx, K)
            curve.append((step, pplx))
        if step % 300 == 0:
            print(f"  [{name}] step {step}/{steps} | recon {F.mse_loss(xhat,x).item():.4f} | "
                  f"perplexity {curve[-1][1]:.0f}/{K} | {step/(time.time()-t0):.1f} it/s", flush=True)
    model.eval()
    test = cifar_tensor(data_dir, train=False, n=1000)
    with torch.no_grad():
        xhat, _, idx = model(test)
        recon = F.mse_loss(xhat, test).item()
    pplx, counts = perplexity(idx, K)
    used = int((counts > 0).sum())
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"_{key}.npz", name=name, recon=recon, pplx=pplx, used=used,
             counts=counts.numpy(), curve=np.array(curve))
    print(f"[{name}] recon {recon:.4f} | perplexity {pplx:.0f}/{K} | codes used {used}/{K}", flush=True)


def make_plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style as ps
    data = {k: np.load(OUT / f"_{k}.npz", allow_pickle=True) for k in CONFIGS}

    # perplexity curves
    fig, ax = ps.new_axes(7.0, 4.2)
    for i, k in enumerate(CONFIGS):
        c = data[k]["curve"]
        ax.plot(c[:, 0], c[:, 1], color=ps.SERIES[i % 3], label=str(data[k]["name"]))
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, f"Codes actually used (perplexity) during training — of {K}",
              "training step", "perplexity (effective codes)", OUT / "perplexity_curves.png")

    # usage histograms
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.4), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, k in zip(axes, CONFIGS):
        counts = data[k]["counts"]; order = np.argsort(counts)[::-1]
        ax.bar(range(K), counts[order] / counts.sum() * 100, color=ps.SERIES[0], width=1.0)
        ax.set_title(f"{data[k]['name']}\n{int(data[k]['used'])}/{K} codes, pplx {float(data[k]['pplx']):.0f}",
                     fontsize=9, color="#0b0b0b")
        ax.set_xlim(0, K); ax.set_xlabel("codebook entry (sorted)", fontsize=8)
        ax.set_facecolor(ps.SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("% of tokens", fontsize=9)
    fig.suptitle("Codebook usage: collapse (left) → cured (right)", fontsize=12, color="#0b0b0b")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "usage_histograms.png", facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)

    lines = ["config,recon_mse,perplexity,codes_used,codebook"]
    for k in CONFIGS:
        d = data[k]
        lines.append(f"{d['name']},{float(d['recon']):.4f},{float(d['pplx']):.1f},{int(d['used'])},{K}")
    (OUT / "results.csv").write_text("\n".join(lines) + "\n")
    print("wrote figures + results.csv")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--config", choices=list(CONFIGS))
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    if args.plot:
        make_plots()
    elif args.config:
        run_config(args.config, args.data_dir, args.steps)
    else:
        raise SystemExit("pass --config <name> or --plot")


if __name__ == "__main__":
    main()
