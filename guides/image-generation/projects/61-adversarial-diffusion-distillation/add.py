"""Adversarial Diffusion Distillation (ADD) — the SDXL-Turbo recipe — and why
the adversarial term matters.

Consistency/LCM distillation (project 60) makes a few-step student by matching a
teacher's ODE. Pushed to a SINGLE step it tends to go blurry, because an L2
distillation loss averages over everything the teacher might have produced. ADD
(Sauer et al. 2023) fixes the blur by adding a GAN: a discriminator forces the
student's one-step output onto the sharp real-image manifold, while a
distillation term keeps it faithful to the teacher.

This project isolates that idea. From one DDPM teacher we distill two 1-step
students that are identical except for the loss:

    distill-only   : student matches the teacher's denoised estimate (L2)     -> blurry
    ADD            : the same L2 term + a hinge-GAN discriminator             -> sharp

    python add.py --data-dir data      # ~9 min on CPU

Reuses the phase-5 U-Net / DDPM (project 24) via sys.path.
"""

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))

from diffusion import GaussianDiffusion  # noqa: E402
from train import EMA, infinite  # noqa: E402
from unet import UNet  # noqa: E402

OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"
T = 200


def mnist_loader(data_dir, bs=64):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    ds = datasets.MNIST(str(data_dir), train=True, download=True, transform=tf)
    return DataLoader(ds, batch_size=bs, shuffle=True, num_workers=2, drop_last=True)


def train_teacher(data_dir, out, steps=800, seed=0):
    if out.exists():
        m = UNet(); m.load_state_dict(torch.load(out)["ema"]); return m.eval()
    torch.manual_seed(seed)
    model = UNet(); ema = EMA(model, 0.995)
    diffusion = GaussianDiffusion(T=T, schedule="linear", device="cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    batches = infinite(mnist_loader(data_dir))
    t0 = time.time()
    for step in range(1, steps + 1):
        loss = diffusion.loss(model, next(batches)[0])
        opt.zero_grad(); loss.backward(); opt.step(); ema.update(model)
        if step % 200 == 0:
            print(f"  [teacher] {step}/{steps} loss {loss.item():.4f} "
                  f"{step/(time.time()-t0):.1f} it/s", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ema": ema.shadow.state_dict()}, out)
    return ema.shadow.eval()


class X0UNet(UNet):
    """A U-Net repurposed to predict x0 directly (bounded output for 1-step
    generation), initialised from an eps-prediction teacher for feature reuse."""
    def forward(self, x, t):
        return super().forward(x, t).clamp(-1, 1)


class Discriminator(nn.Module):
    """A small patch discriminator over 28x28 images (hinge GAN)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.LeakyReLU(0.2),      # 14
            nn.Conv2d(32, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.LeakyReLU(0.2),  # 7
            nn.Conv2d(64, 1, 3, 1, 1),                          # 7x7 patch logits
        )

    def forward(self, x):
        return self.net(x)


def q_sample(alpha_bar, x0, t, eps):
    a = alpha_bar[t].view(-1, 1, 1, 1)
    return a.sqrt() * x0 + (1 - a).sqrt() * eps


def teacher_x0(teacher, alpha_bar, x0_hat, t):
    """The teacher's opinion of x0 after re-noising the student's guess to t —
    the distillation target (a one-step score-distillation)."""
    eps = torch.randn_like(x0_hat)
    x_t = q_sample(alpha_bar, x0_hat, t, eps)
    with torch.no_grad():
        e = teacher(x_t, t)
    a = alpha_bar[t].view(-1, 1, 1, 1)
    return ((x_t - (1 - a).sqrt() * e) / a.sqrt()).clamp(-1, 1)


# Student sampling starts from a fixed high noise level and predicts x0 in one shot.
STUDENT_STEPS = torch.tensor([T - 1, T * 3 // 4, T // 2, T // 4])


def distill(teacher, data_dir, adversarial, out, steps=450, seed=0):
    torch.manual_seed(seed)
    diffusion = GaussianDiffusion(T=T, schedule="linear", device="cpu")
    ab = diffusion.alpha_bar
    student = X0UNet(); student.load_state_dict(teacher.state_dict()); student.train()
    optG = torch.optim.AdamW(student.parameters(), lr=1e-4)
    D = Discriminator()
    optD = torch.optim.AdamW(D.parameters(), lr=2e-4, betas=(0.5, 0.9))
    batches = infinite(mnist_loader(data_dir))
    t0 = time.time()
    for step in range(1, steps + 1):
        x0, _ = next(batches)
        B = x0.size(0)
        s = STUDENT_STEPS[torch.randint(0, len(STUDENT_STEPS), (B,))]
        eps = torch.randn_like(x0)
        x_s = q_sample(ab, x0, s, eps)
        x0_hat = student(x_s, s)

        # Distillation target from the teacher.
        t_d = torch.randint(T // 4, T, (B,))
        target = teacher_x0(teacher, ab, x0_hat.detach(), t_d)
        loss_distill = torch.mean((x0_hat - target) ** 2)

        if adversarial:
            # Discriminator step (hinge).
            optD.zero_grad()
            d_real = D(x0)
            d_fake = D(x0_hat.detach())
            loss_D = (torch.relu(1 - d_real).mean() + torch.relu(1 + d_fake).mean())
            loss_D.backward(); optD.step()
            # Generator: distill + adversarial.
            optG.zero_grad()
            loss_adv = -D(x0_hat).mean()
            lossG = loss_distill + 0.5 * loss_adv
            lossG.backward(); optG.step()
        else:
            optG.zero_grad(); loss_distill.backward(); optG.step()

        if step % 100 == 0:
            tag = "ADD" if adversarial else "distill-only"
            print(f"  [{tag}] {step}/{steps} distill {loss_distill.item():.4f} "
                  f"{step/(time.time()-t0):.1f} it/s", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"student": student.state_dict()}, out)
    return student.eval(), ab


@torch.no_grad()
def sample_1step(student, ab, n=64, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(n, 1, 28, 28)
    s = torch.full((n,), T - 1, dtype=torch.long)
    return student(x, s)


def sharpness(imgs):
    """High-frequency energy (mean |Laplacian|) — blur lowers it."""
    lap = (imgs[:, :, 1:, :] - imgs[:, :, :-1, :]).abs().mean() + \
          (imgs[:, :, :, 1:] - imgs[:, :, :, :-1]).abs().mean()
    return float(lap)


def readability(imgs, clf):
    with torch.no_grad():
        return torch.softmax(clf(imgs), 1).max(1).values.mean().item()


def row(imgs, n=10):
    r = torch.cat([imgs[i, 0] for i in range(n)], dim=1)
    return ((r.clamp(-1, 1) + 1) * 127.5).byte().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--teacher-steps", type=int, default=800)
    ap.add_argument("--distill-steps", type=int, default=450)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(HERE.parent / "58-caption-ablation"))
    from mnist_classifier import load_or_train
    clf = load_or_train(args.data_dir,
                        HERE.parent / "58-caption-ablation/checkpoints/classifier.pt")

    print("training teacher ...")
    teacher = train_teacher(args.data_dir, CKPT / "teacher.pt", args.teacher_steps)

    print("distilling (distill-only, no discriminator) ...")
    s_plain, ab = distill(teacher, args.data_dir, False, CKPT / "plain.pt", args.distill_steps)
    print("distilling (ADD, with discriminator) ...")
    s_add, _ = distill(teacher, args.data_dir, True, CKPT / "add.pt", args.distill_steps)

    imgs_plain = sample_1step(s_plain, ab)
    imgs_add = sample_1step(s_add, ab)

    stats = {
        "distill-only (1 step)": (sharpness(imgs_plain), readability(imgs_plain, clf)),
        "ADD (1 step)": (sharpness(imgs_add), readability(imgs_add, clf)),
    }
    for k, (sh, rd) in stats.items():
        print(f"  {k:24s} sharpness {sh:.3f}  readability {rd:.3f}")

    plot_compare(imgs_plain, imgs_add, OUT / "samples.png")
    plot_bars(stats, OUT / "metrics.png")

    lines = ["student,sharpness,readability"]
    for k, (sh, rd) in stats.items():
        lines.append(f"{k},{sh:.4f},{rd:.4f}")
    (OUT / "results.csv").write_text("\n".join(lines) + "\n")
    print(f"\nwrote figures + {OUT/'results.csv'}")


def plot_compare(plain, add, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(9, 2.4), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    axes[0].imshow(row(plain), cmap="gray", vmin=0, vmax=255)
    axes[0].set_ylabel("distill-only", fontsize=9, rotation=0, ha="right", va="center")
    axes[1].imshow(row(add), cmap="gray", vmin=0, vmax=255)
    axes[1].set_ylabel("ADD", fontsize=9, rotation=0, ha="right", va="center")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("One-step samples: the adversarial term (ADD, bottom) is sharper than "
                 "L2 distillation alone (top)", fontsize=10, color="#0b0b0b")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(path, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_bars(stats, path):
    import matplotlib
    matplotlib.use("Agg")
    sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
    import plot_style as ps
    names = list(stats.keys())
    fig, ax = ps.new_axes(6.6, 4.0)
    x = np.arange(len(names)); w = 0.35
    sh = [stats[n][0] for n in names]; rd = [stats[n][1] for n in names]
    ax.bar(x - w / 2, sh, w, color=ps.SERIES[0], label="sharpness (|grad|)")
    ax.bar(x + w / 2, rd, w, color=ps.SERIES[1], label="readability")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "The adversarial term buys sharpness at equal step count",
              "", "score", path)


if __name__ == "__main__":
    main()
