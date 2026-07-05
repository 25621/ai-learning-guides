"""Train a classifier on NOISY images — the extra ingredient classifier
guidance needs.

An ordinary classifier only ever sees clean images, but during guided
sampling it will be asked "what digit is this?" at every noise level from
pure static to nearly clean. So we train it exactly like the diffusion model
is trained: sample a random t, noise the image with q_sample, and ask for the
label anyway. The timestep enters through FiLM on each conv block so the
classifier knows how much noise to expect.

Run:
    python noisy_classifier.py
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
from diffusion import GaussianDiffusion  # noqa: E402
from unet import sinusoidal_embedding  # noqa: E402


class FiLMBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, temb_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.norm = nn.GroupNorm(8, out_ch)
        self.temb_proj = nn.Linear(temb_dim, out_ch * 2)

    def forward(self, x, temb):
        h = self.norm(self.conv(x))
        scale, shift = self.temb_proj(temb).chunk(2, dim=1)
        return F.silu(h * (1 + scale[:, :, None, None]) + shift[:, :, None, None])


class NoisyClassifier(nn.Module):
    """Small time-conditioned CNN: p(y | x_t, t)."""

    def __init__(self, temb_dim: int = 64, num_classes: int = 10):
        super().__init__()
        self.temb_dim = temb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(temb_dim, temb_dim), nn.SiLU(), nn.Linear(temb_dim, temb_dim)
        )
        self.blocks = nn.ModuleList(
            [
                FiLMBlock(1, 32, 2, temb_dim),   # 28 -> 14
                FiLMBlock(32, 64, 2, temb_dim),  # 14 -> 7
                FiLMBlock(64, 64, 1, temb_dim),
            ]
        )
        self.head = nn.Linear(64, num_classes)

    def forward(self, x, t):
        temb = self.time_mlp(sinusoidal_embedding(t, self.temb_dim))
        for block in self.blocks:
            x = block(x, temb)
        return self.head(x.mean(dim=(2, 3)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--schedule", default="linear")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--out", default=str(HERE / "checkpoints/noisy_classifier.pt"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    train_ds = datasets.MNIST(args.data_dir, train=True, download=True, transform=tf)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    clf = NoisyClassifier().to(device)
    diffusion = GaussianDiffusion(T=args.T, schedule=args.schedule, device=device)
    opt = torch.optim.AdamW(clf.parameters(), lr=args.lr)

    def batches():
        while True:
            yield from loader

    it = batches()
    for step in range(1, args.steps + 1):
        x0, y = next(it)
        x0, y = x0.to(device), y.to(device)
        t = torch.randint(0, args.T, (x0.size(0),), device=device)
        x_t = diffusion.q_sample(x0, t, torch.randn_like(x0))
        loss = F.cross_entropy(clf(x_t, t), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"step {step}/{args.steps} | loss {loss.item():.3f}", flush=True)

    # Accuracy as a function of noise level, on 2000 held-out digits.
    test_ds = datasets.MNIST(args.data_dir, train=False, download=True, transform=tf)
    x_test = torch.stack([test_ds[i][0] for i in range(2000)]).to(device)
    y_test = torch.tensor([test_ds[i][1] for i in range(2000)], device=device)
    clf.eval()
    acc_rows = [("t", "accuracy")]
    t_grid = [int(f * (args.T - 1)) for f in (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)]
    with torch.no_grad():
        for t_val in t_grid:
            t = torch.full((x_test.size(0),), t_val, device=device, dtype=torch.long)
            x_t = diffusion.q_sample(x_test, t, torch.randn_like(x_test))
            acc = (clf(x_t, t).argmax(1) == y_test).float().mean().item()
            acc_rows.append((t_val, f"{acc:.4f}"))
            print(f"accuracy at t={t_val}: {acc:.4f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": clf.state_dict(), "T": args.T, "schedule": args.schedule}, args.out)
    import csv

    acc_path = HERE / "outputs/accuracy_vs_t.csv"
    acc_path.parent.mkdir(parents=True, exist_ok=True)
    with open(acc_path, "w", newline="") as f:
        csv.writer(f).writerows(acc_rows)
    print(f"saved {args.out} and {acc_path}")


if __name__ == "__main__":
    main()
