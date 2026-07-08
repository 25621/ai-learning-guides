"""FSQ (Finite Scalar Quantization) vs. a learned VQ codebook.

FSQ makes discrete image tokens WITHOUT training a codebook: the encoder emits a
few channels, each is squashed to a bounded range and *rounded to a fixed grid*
of levels (straight-through for gradients). The implicit vocabulary is the
product of the per-channel level counts — here levels [8,8,4] give 8·8·4 = 256
"codes", matched to the VQ-VAE's 256-entry codebook. With no codebook to train,
FSQ cannot suffer codebook collapse.

We train a VQ-VAE and an FSQ auto-encoder identically and compare them on
reconstruction quality (MSE and reconstruction-FID). Each is trained in its own
invocation (robust to time limits):

    python fsq.py --data-dir data --config vq
    python fsq.py --data-dir data --config fsq
    python fsq.py --data-dir data --plot        # Inception FID + figures

Reuses the encoder/decoder + VQ from project 12 and the FID from project 04.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "12-vq-vae-on-cifar-10"))
sys.path.insert(0, str(HERE.parent / "04-fid-from-scratch"))
sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
from vq_lib import Encoder, Decoder, VectorQuantizerEMA, cifar_loader, infinite, cifar_tensor  # noqa: E402

OUT = HERE / "outputs"
LEVELS = [8, 8, 4]           # implicit codebook = 256, to match VQ K=256
K = 256


class FSQ(nn.Module):
    """Bound each channel to its level range and round (straight-through)."""
    def __init__(self, levels=LEVELS):
        super().__init__()
        self.register_buffer("levels", torch.tensor(levels))
        self.dim = len(levels)

    def forward(self, z):
        half = ((self.levels.float() - 1) / 2).view(1, -1, 1, 1)
        zb = torch.tanh(z) * half
        zq = torch.round(zb)
        return zb + (zq - zb).detach(), torch.tensor(0.0)

    @torch.no_grad()
    def indices(self, z):
        half = ((self.levels.float() - 1) / 2).view(1, -1, 1, 1)
        codes = (torch.round(torch.tanh(z) * half) + half).long()
        idx = torch.zeros(z.size(0), z.size(2), z.size(3), dtype=torch.long)
        radix = 1
        for d in range(self.dim):
            idx = idx + codes[:, d] * radix
            radix *= int(self.levels[d])
        return idx


class Tokenizer(nn.Module):
    def __init__(self, kind, ch=64):
        super().__init__()
        self.kind = kind
        Dz = 32 if kind == "vq" else len(LEVELS)
        self.enc = Encoder(D=Dz, ch=ch)
        self.quant = VectorQuantizerEMA(K=K, D=Dz, reinit=True, dead_after=128) if kind == "vq" else FSQ()
        self.dec = Decoder(D=Dz, ch=ch)

    def forward(self, x):
        out = self.quant(self.enc(x))       # VQ returns (zq, loss, idx); FSQ (zq, loss)
        return self.dec(out[0]), out[1]

    @torch.no_grad()
    def code_indices(self, x):
        z = self.enc(x)
        if self.kind == "vq":
            _, _, idx = self.quant(z)
            return idx
        return self.quant.indices(z)


def run_config(kind, data_dir, steps, fid_n=512):
    torch.manual_seed(0)
    model = Tokenizer(kind)
    loader = infinite(cifar_loader(data_dir, 128, True))
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    t0 = time.time()
    for step in range(1, steps + 1):
        (x,) = next(loader)
        xhat, qloss = model(x)
        loss = F.mse_loss(xhat, x) + qloss
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 400 == 0:
            print(f"  [{kind}] step {step}/{steps} | recon {F.mse_loss(xhat,x).item():.4f} | "
                  f"{step/(time.time()-t0):.1f} it/s", flush=True)
    model.eval()
    real = cifar_tensor(data_dir, train=False, n=fid_n)
    with torch.no_grad():
        recon, _ = model(real)
        idx = model.code_indices(real)
    mse = F.mse_loss(recon, real).item()
    used = int(torch.bincount(idx.reshape(-1), minlength=K).gt(0).sum())
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"_{kind}.npz", recon=recon.numpy(), mse=mse, used=used)
    print(f"[{kind}] recon MSE {mse:.4f} | codes used {used}/{K}", flush=True)


def to_img(x):
    return ((np.clip(x, -1, 1) + 1) / 2).transpose(0, 2, 3, 1)


def make_plots(data_dir, fid_n=512):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style as ps
    from fid import build_extractor, inception_features, fid as frechet

    vq = np.load(OUT / "_vq.npz"); fsq = np.load(OUT / "_fsq.npz")
    real = cifar_tensor(data_dir, train=False, n=fid_n)
    print("computing reconstruction FID (Inception) ...")
    ext = build_extractor()
    fr = inception_features(real, ext)
    stats = {}
    for name, d in [("VQ-VAE", vq), ("FSQ", fsq)]:
        fv = frechet(fr, inception_features(torch.tensor(d["recon"]), ext))
        stats[name] = dict(mse=float(d["mse"]), fid=fv, used=int(d["used"]))
        print(f"  {name}: recon MSE {stats[name]['mse']:.4f} | recon FID {fv:.1f} | "
              f"codes used {stats[name]['used']}/{K}")

    # reconstruction strip
    fig, axes = plt.subplots(3, 10, figsize=(10, 3.2), dpi=115)
    fig.patch.set_facecolor("#fcfcfb")
    rows = [("input", to_img(real[:10].numpy())),
            ("VQ-VAE", to_img(vq["recon"][:10])), ("FSQ", to_img(fsq["recon"][:10]))]
    for r, (name, imgs) in enumerate(rows):
        for c in range(10):
            axes[r][c].imshow(imgs[c]); axes[r][c].axis("off")
        fig.text(0.02, 1 - (r + 0.5) / 3 * 0.9 - 0.02, name, ha="left", fontsize=10, color="#0b0b0b")
    fig.suptitle("Reconstructions: learned VQ codebook vs. codebook-free FSQ",
                 fontsize=12, color="#0b0b0b")
    fig.tight_layout(rect=[0.08, 0, 1, 0.93])
    fig.savefig(OUT / "reconstructions.png", facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)

    # bars
    fig, ax = ps.new_axes(6.4, 4.0)
    xs = np.arange(2); w = 0.35
    ax.bar(xs - w / 2, [stats["VQ-VAE"]["mse"], stats["FSQ"]["mse"]], w, color=ps.SERIES[0], label="recon MSE")
    ax.set_ylabel("recon MSE", color=ps.SERIES[0], fontsize=10)
    ax.set_xticks(xs); ax.set_xticklabels(["VQ-VAE", "FSQ"])
    ax2 = ax.twinx()
    ax2.bar(xs + w / 2, [stats["VQ-VAE"]["fid"], stats["FSQ"]["fid"]], w, color=ps.SERIES[2], label="recon FID")
    ax2.set_ylabel("recon FID", color=ps.SERIES[2], fontsize=10); ax2.spines["top"].set_visible(False)
    ps.finish(fig, ax, "VQ vs FSQ: reconstruction MSE and FID (lower is better)",
              "", "recon MSE", OUT / "vq_vs_fsq.png")

    lines = ["tokenizer,recon_mse,recon_fid,codes_used,vocab"]
    for name in ["VQ-VAE", "FSQ"]:
        s = stats[name]; lines.append(f"{name},{s['mse']:.4f},{s['fid']:.2f},{s['used']},{K}")
    (OUT / "results.csv").write_text("\n".join(lines) + "\n")
    print("wrote figures + results.csv\n" + "\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--config", choices=["vq", "fsq"])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    if args.plot:
        make_plots(args.data_dir)
    elif args.config:
        run_config(args.config, args.data_dir, args.steps)
    else:
        raise SystemExit("pass --config vq|fsq or --plot")


if __name__ == "__main__":
    main()
