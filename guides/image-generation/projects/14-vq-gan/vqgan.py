"""VQ-GAN: make VQ reconstructions crisp with a perceptual loss + patch critic.

A VQ-VAE trained on pixel error alone rebuilds blurry images — averaging over
plausible textures is the safest way to lower per-pixel error. VQ-GAN adds two
signals that punish blur:

  - a PERCEPTUAL loss: match images in the feature space of a pretrained VGG, so
    the loss rewards correct textures/edges rather than exact pixels;
  - a PATCH DISCRIMINATOR: a small GAN critic that judges whether each local
    patch looks real, forcing the decoder to commit to sharp detail.

We train the SAME encoder/decoder/codebook twice — once with pixel loss only
(plain VQ-VAE), once with the full VQ-GAN objective — and compare reconstructions.
Each is trained in its own invocation (robust to time limits):

    python vqgan.py --data-dir data --config plain
    python vqgan.py --data-dir data --config vqgan
    python vqgan.py --plot

Reuses the shared encoder/decoder/quantizer from project 12 via sys.path.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "12-vq-vae-on-cifar-10"))
sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
from vq_lib import Encoder, Decoder, VectorQuantizerEMA, cifar_loader, infinite, cifar_tensor  # noqa: E402

OUT = HERE / "outputs"
K, D = 256, 32


class VQAE(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.enc = Encoder(D=D, ch=ch)
        self.vq = VectorQuantizerEMA(K=K, D=D, reinit=True, dead_after=128)
        self.dec = Decoder(D=D, ch=ch)

    def forward(self, x):
        z_q, vq_loss, idx = self.vq(self.enc(x))
        return self.dec(z_q), vq_loss


class Perceptual(nn.Module):
    """Shallow VGG16 feature-space L1 distance (a lightweight LPIPS stand-in)."""
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:9].eval()
        for p in vgg.parameters():
            p.requires_grad_(False)
        self.vgg = vgg
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        xn = ((x + 1) / 2 - self.mean) / self.std
        yn = ((y + 1) / 2 - self.mean) / self.std
        return F.l1_loss(self.vgg(xn), self.vgg(yn))


class PatchDiscriminator(nn.Module):
    """Small PatchGAN: a grid of real/fake logits over local patches."""
    def __init__(self, ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, ch, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(ch, ch * 2, 4, 2, 1), nn.GroupNorm(8, ch * 2), nn.LeakyReLU(0.2),
            nn.Conv2d(ch * 2, 1, 3, 1, 1))
    def forward(self, x):
        return self.net(x)


def to_img(x):
    return ((x.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).numpy()


def sharpness(x):
    return float((x[:, :, 1:] - x[:, :, :-1]).abs().mean() + (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean())


def run_config(key, data_dir, steps):
    torch.manual_seed(0)
    model = VQAE()
    bs = 64 if key == "vqgan" else 128
    loader = infinite(cifar_loader(data_dir, bs, True))
    optG = torch.optim.Adam(model.parameters(), lr=2e-4)
    perceptual = Perceptual() if key == "vqgan" else None
    D_net = PatchDiscriminator() if key == "vqgan" else None
    optD = torch.optim.Adam(D_net.parameters(), lr=2e-4, betas=(0.5, 0.9)) if key == "vqgan" else None
    t0 = time.time()
    for step in range(1, steps + 1):
        (x,) = next(loader)
        xhat, vq = model(x)
        if key == "plain":
            loss = F.mse_loss(xhat, x) + vq
            optG.zero_grad(); loss.backward(); optG.step()
        else:
            if step > steps // 4:                          # let the AE warm up first
                optD.zero_grad()
                (torch.relu(1 - D_net(x)).mean() + torch.relu(1 + D_net(xhat.detach())).mean()).backward()
                optD.step()
            optG.zero_grad()
            g = F.l1_loss(xhat, x) + perceptual(x, xhat) + vq
            if step > steps // 4:
                g = g - 0.2 * D_net(xhat).mean()
            g.backward(); optG.step()
        if step % 300 == 0:
            print(f"  [{key}] step {step}/{steps} | l1 {F.l1_loss(xhat,x).item():.4f} | "
                  f"{step/(time.time()-t0):.1f} it/s", flush=True)
    model.eval()
    test = cifar_tensor(data_dir, train=False, n=10)
    with torch.no_grad():
        recon, _ = model(test)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"_{key}.npz", recon=recon.numpy(), input=test.numpy(),
             sharp=sharpness(recon))
    print(f"[{key}] sharpness {sharpness(recon):.4f}", flush=True)


def make_plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plain = np.load(OUT / "_plain.npz"); vqgan = np.load(OUT / "_vqgan.npz")
    x = torch.tensor(plain["input"])
    rows = [("input", to_img(x)),
            ("VQ-VAE (pixel loss)", to_img(torch.tensor(plain["recon"]))),
            ("VQ-GAN (perc + patch-GAN)", to_img(torch.tensor(vqgan["recon"])))]
    fig, axes = plt.subplots(3, 10, figsize=(10, 3.2), dpi=115)
    fig.patch.set_facecolor("#fcfcfb")
    for r, (name, imgs) in enumerate(rows):
        for c in range(10):
            axes[r][c].imshow(imgs[c]); axes[r][c].axis("off")
        fig.text(0.005, 1 - (r + 0.5) / 3 * 0.9 - 0.02, name, ha="left", fontsize=9, color="#0b0b0b")
    fig.suptitle("Same encoder/decoder/codebook — pixel loss vs. VQ-GAN objective",
                 fontsize=12, color="#0b0b0b")
    fig.tight_layout(rect=[0.16, 0, 1, 0.93])
    fig.savefig(OUT / "reconstructions.png", facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    sh_in = sharpness(x)
    (OUT / "results.csv").write_text(
        "model,sharpness_grad\n"
        f"input,{sh_in:.4f}\nVQ-VAE (pixel),{float(plain['sharp']):.4f}\n"
        f"VQ-GAN,{float(vqgan['sharp']):.4f}\n")
    print(f"wrote figures | sharpness: input {sh_in:.3f} | VQ-VAE {float(plain['sharp']):.3f} | "
          f"VQ-GAN {float(vqgan['sharp']):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--config", choices=["plain", "vqgan"])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    if args.plot:
        make_plots()
    elif args.config:
        run_config(args.config, args.data_dir, args.steps)
    else:
        raise SystemExit("pass --config plain|vqgan or --plot")


if __name__ == "__main__":
    main()
