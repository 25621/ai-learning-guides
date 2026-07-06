"""A conditional VAE on MNIST: generate the digit you ask for.

A plain VAE dreams up random digits; you cannot request a '7'. A conditional VAE
feeds the class label into BOTH the encoder and the decoder, so the model no
longer has to store 'which digit is this' in the latent — the label already says
so. The latent is then free to encode only *style* (slant, thickness), and at
generation time you hand the decoder the label you want plus a random style code
and reliably get that digit.

We verify it two ways: a labelled sample grid (each row = one requested digit),
and an automatic check that reads the generated digits back with the Phase-10
MNIST classifier.

    python cvae.py --data-dir data      # ~3 min on CPU

Reuses project 58's MNIST classifier as the automatic reader.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "58-caption-ablation"))
sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))

OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"
ZDIM = 16
NCLS = 10


class CVAE(nn.Module):
    """Class-conditional VAE. The label rides in as an embedding concatenated to
    the encoder's features and to the decoder's latent."""

    def __init__(self, zdim=ZDIM, base=32, emb=16):
        super().__init__()
        self.zdim = zdim
        self.label = nn.Embedding(NCLS, emb)
        self.enc_body = nn.Sequential(
            nn.Conv2d(1, base, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(base, base * 2, 4, 2, 1), nn.ReLU(), nn.Flatten())
        self.enc_fc = nn.Linear(base * 2 * 7 * 7 + emb, zdim * 2)
        self.dec_fc = nn.Linear(zdim + emb, base * 2 * 7 * 7)
        self.base = base
        self.dec_body = nn.Sequential(
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(base, 1, 4, 2, 1), nn.Sigmoid())

    def encode(self, x, y):
        h = torch.cat([self.enc_body(x), self.label(y)], dim=1)
        return self.enc_fc(h).chunk(2, dim=1)

    def decode(self, z, y):
        h = self.dec_fc(torch.cat([z, self.label(y)], dim=1)).view(-1, self.base * 2, 7, 7)
        return self.dec_body(h)

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
        return self.decode(z, y), mu, logvar

    def loss(self, x, y, beta=1.0):
        xhat, mu, logvar = self(x, y)
        recon = F.binary_cross_entropy(xhat, x, reduction="none").sum((1, 2, 3)).mean()
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()
        return recon + beta * kl, recon, kl

    @torch.no_grad()
    def sample(self, y):
        z = torch.randn(len(y), self.zdim)
        return self.decode(z, y)


def mnist_loader(data_dir, bs=128, train=True):
    tf = transforms.Compose([transforms.ToTensor()])
    ds = datasets.MNIST(str(data_dir), train=train, download=True, transform=tf)
    return DataLoader(ds, batch_size=bs, shuffle=train, num_workers=2, drop_last=train)


def train(model, data_dir, steps, lr=1e-3):
    loader = mnist_loader(data_dir, 128, True)
    it = iter(loader)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    for step in range(1, steps + 1):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader); x, y = next(it)
        loss, recon, kl = model.loss(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 400 == 0:
            print(f"  step {step}/{steps} | recon {recon.item():.1f} | kl {kl.item():.1f} | "
                  f"{step/(time.time()-t0):.1f} it/s", flush=True)


def plot_grid(model, path, per_class=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    torch.manual_seed(1)
    fig, axes = plt.subplots(NCLS, per_class, figsize=(per_class, NCLS), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    with torch.no_grad():
        for c in range(NCLS):
            y = torch.full((per_class,), c, dtype=torch.long)
            imgs = model.sample(y)
            for k in range(per_class):
                axes[c][k].imshow(imgs[k, 0], cmap="gray", vmin=0, vmax=1)
                axes[c][k].set_xticks([]); axes[c][k].set_yticks([])
            axes[c][0].set_ylabel(f"“{c}”", rotation=0, labelpad=16,
                                  fontsize=11, va="center", color="#0b0b0b")
    fig.suptitle("Conditional VAE: each row is one requested digit (random style per column)",
                 fontsize=12, color="#0b0b0b")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--steps", type=int, default=2500)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    from mnist_classifier import load_or_train, read_digits
    clf = load_or_train(args.data_dir, HERE.parent / "58-caption-ablation/checkpoints/classifier.pt")

    model = CVAE()
    print(f"CVAE params: {sum(p.numel() for p in model.parameters()):,}\ntraining ...")
    train(model, args.data_dir, args.steps)
    model.eval()
    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, CKPT / "cvae.pt")

    # Adherence: ask for each digit 50 times, read back with the classifier.
    torch.manual_seed(2)
    per = 50
    y = torch.arange(NCLS).repeat_interleave(per)
    with torch.no_grad():
        imgs = model.sample(y)
        # The VAE decodes to [0,1]; project 58's classifier expects the MNIST
        # normalization to [-1,1], so rescale before reading the digits.
        pred, _ = read_digits(clf, imgs * 2 - 1)
    adherence = (pred == y).float().mean().item()
    per_class = [(pred[y == c] == c).float().mean().item() for c in range(NCLS)]

    plot_grid(model, OUT / "conditional_grid.png")
    (OUT / "results.csv").write_text(
        "digit,adherence\n" + "\n".join(f"{c},{a:.3f}" for c, a in enumerate(per_class))
        + f"\noverall,{adherence:.3f}\n")
    print(f"\nlabel adherence (classifier agrees with request): {adherence:.1%}")
    print(f"wrote figures + {OUT/'results.csv'}")


if __name__ == "__main__":
    main()
