"""Consistency distillation: turn a 50-step diffusion teacher into a 1-4 step
student.

A diffusion model samples by crawling down a long ODE trajectory from noise to
image. A consistency model (Song et al. 2023) instead learns a function
f(x_t, t) that jumps *directly* to the trajectory's endpoint x_0 from anywhere
on it. If f is truly consistent — same answer from every point on one
trajectory — then a single evaluation from pure noise produces an image, and a
few re-noise/re-predict steps clean up what one step misses. This is the idea
behind Latent Consistency Models and the few-step 'Turbo' generators.

We distill it from a pretrained DDPM teacher. For each training pair we noise a
real image to a high level, take ONE deterministic DDIM step down with the
teacher, and train the student to give the same x_0 estimate at both the high
and the (teacher-advanced) lower level — with the low-level answer coming from
an EMA 'target' copy of the student. A boundary-preserving EDM parameterization
guarantees f(x, t->0) = x, which anchors the whole scheme.

    python consistency.py --data-dir data      # ~9 min on CPU

Reuses the phase-5 U-Net / DDPM (project 24) via sys.path.
"""

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
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
SIGMA_DATA = 0.5


def mnist_loader(data_dir, bs=64):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    ds = datasets.MNIST(str(data_dir), train=True, download=True, transform=tf)
    return DataLoader(ds, batch_size=bs, shuffle=True, num_workers=2, drop_last=True)


# --------------------------------------------------------------------------- #
# Teacher: a plain unconditional DDPM.
# --------------------------------------------------------------------------- #
def train_teacher(data_dir, out, steps=900, seed=0):
    if out.exists():
        m = UNet(); m.load_state_dict(torch.load(out)["ema"]); return m.eval()
    torch.manual_seed(seed)
    model = UNet(); ema = EMA(model, 0.995)
    diffusion = GaussianDiffusion(T=T, schedule="linear", device="cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    batches = infinite(mnist_loader(data_dir))
    t0 = time.time()
    for step in range(1, steps + 1):
        x0, _ = next(batches)
        loss = diffusion.loss(model, x0)
        opt.zero_grad(); loss.backward(); opt.step(); ema.update(model)
        if step % 200 == 0:
            print(f"  [teacher] {step}/{steps} loss {loss.item():.4f} "
                  f"{step/(time.time()-t0):.1f} it/s", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ema": ema.shadow.state_dict()}, out)
    return ema.shadow.eval()


# --------------------------------------------------------------------------- #
# The consistency parameterization (EDM skip/out over the VP schedule).
# --------------------------------------------------------------------------- #
class Consistency:
    def __init__(self, alpha_bar):
        self.alpha_bar = alpha_bar
        self.sigma = ((1 - alpha_bar) / alpha_bar).sqrt()   # VP -> sigma
        self.sigma_min = float(self.sigma[0])

    def coeffs(self, t):
        a = self.alpha_bar[t].view(-1, 1, 1, 1)
        sig = self.sigma[t].view(-1, 1, 1, 1)
        s0 = self.sigma_min
        c_skip = SIGMA_DATA ** 2 / ((sig - s0) ** 2 + SIGMA_DATA ** 2)
        c_out = SIGMA_DATA * (sig - s0) / (SIGMA_DATA ** 2 + sig ** 2).sqrt()
        c_in = 1.0 / (sig ** 2 + SIGMA_DATA ** 2).sqrt()
        return a, sig, c_skip, c_out, c_in

    def f(self, net, x_vp, t):
        """Student estimate of x_0 from a VP-noised x_t at integer step t."""
        a, sig, c_skip, c_out, c_in = self.coeffs(t)
        y = x_vp / a.sqrt()                       # = x0 + sigma * eps
        F = net(c_in * y, t)
        return c_skip * y + c_out * F

    @torch.no_grad()
    def teacher_step(self, teacher, x_hi, t_hi, t_lo):
        """One deterministic DDIM step from t_hi down to t_lo using the teacher."""
        a_hi = self.alpha_bar[t_hi].view(-1, 1, 1, 1)
        a_lo = self.alpha_bar[t_lo].view(-1, 1, 1, 1)
        eps = teacher(x_hi, t_hi)
        x0 = (x_hi - (1 - a_hi).sqrt() * eps) / a_hi.sqrt()
        return a_lo.sqrt() * x0 + (1 - a_lo).sqrt() * eps

    def q_sample(self, x0, t, eps):
        a = self.alpha_bar[t].view(-1, 1, 1, 1)
        return a.sqrt() * x0 + (1 - a).sqrt() * eps


# --------------------------------------------------------------------------- #
# Consistency distillation.
# --------------------------------------------------------------------------- #
def distill(teacher, data_dir, out, steps=650, N=18, seed=0):
    torch.manual_seed(seed)
    diffusion = GaussianDiffusion(T=T, schedule="linear", device="cpu")
    cm = Consistency(diffusion.alpha_bar)
    # Student initialised from the teacher — standard and much faster to
    # converge. (The teacher is an EMA copy with grad disabled, so re-enable
    # grad on the student we actually optimise.)
    student = copy.deepcopy(teacher).train()
    for p in student.parameters():
        p.requires_grad_(True)
    target = copy.deepcopy(teacher).eval()
    for p in target.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(student.parameters(), lr=2e-4)
    batches = infinite(mnist_loader(data_dir))
    grid = torch.linspace(0, T - 1, N).round().long()   # time discretisation

    t0 = time.time()
    for step in range(1, steps + 1):
        x0, _ = next(batches)
        B = x0.size(0)
        n = torch.randint(0, N - 1, (B,))
        t_hi = grid[n + 1]
        t_lo = grid[n]
        eps = torch.randn_like(x0)
        x_hi = cm.q_sample(x0, t_hi, eps)
        x_lo = cm.teacher_step(teacher, x_hi, t_hi, t_lo)

        pred_hi = cm.f(student, x_hi, t_hi)
        with torch.no_grad():
            pred_lo = cm.f(target, x_lo, t_lo)
        # Pseudo-Huber loss (Song 2023, 'improved' CM) — robust and smooth.
        c = 0.03
        loss = torch.mean(torch.sqrt((pred_hi - pred_lo) ** 2 + c ** 2) - c)

        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():                       # EMA target update
            for tp, sp in zip(target.parameters(), student.parameters()):
                tp.lerp_(sp, 1.0 - 0.95)
        if step % 100 == 0:
            print(f"  [distill] {step}/{steps} loss {loss.item():.4f} "
                  f"{step/(time.time()-t0):.1f} it/s", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"student": student.state_dict()}, out)
    return student.eval(), cm


# --------------------------------------------------------------------------- #
# Sampling.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def consistency_sample(net, cm, n_steps, shape, seed=0):
    """Multistep consistency sampling: predict x0 from pure noise, then refine
    by re-noising to *progressively lower* levels and predicting again. n_steps=1
    is a single network evaluation. Re-noising to lower (not high) levels each
    round is what makes extra steps monotonically improve quality instead of
    re-injecting too much noise."""
    torch.manual_seed(seed)
    x = torch.randn(shape)                      # x_{T-1} ~ N(0, ~1)
    if n_steps == 1:
        levels = [T - 1]
    else:
        tail = torch.linspace(T // 2, T // 12, n_steps - 1).round().long().tolist()
        levels = [T - 1] + tail
    x0 = None
    for i, lv in enumerate(levels):
        t = torch.full((shape[0],), int(lv), dtype=torch.long)
        x0 = cm.f(net, x, t).clamp(-1, 1)
        if i < len(levels) - 1:
            a = cm.alpha_bar[levels[i + 1]]
            x = a.sqrt() * x0 + (1 - a).sqrt() * torch.randn_like(x0)
    return x0


@torch.no_grad()
def teacher_ddim(teacher, cm, n_steps, shape, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(shape)
    ts = list(np.linspace(0, T - 1, n_steps).astype(int))
    for i in reversed(range(len(ts))):
        t = ts[i]
        a_t = cm.alpha_bar[t]
        a_prev = cm.alpha_bar[ts[i - 1]] if i > 0 else torch.tensor(1.0)
        tb = torch.full((shape[0],), t, dtype=torch.long)
        eps = teacher(x, tb)
        x0 = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-1, 1)
        x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
    return x


def readability(imgs, clf):
    with torch.no_grad():
        return torch.softmax(clf(imgs), 1).max(1).values.mean().item()


def grid_img(imgs, n=8):
    imgs = imgs[:n]
    row = torch.cat([imgs[i, 0] for i in range(len(imgs))], dim=1)
    return ((row.clamp(-1, 1) + 1) * 127.5).byte().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--teacher-steps", type=int, default=1000)
    ap.add_argument("--distill-steps", type=int, default=450)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(HERE.parent / "58-caption-ablation"))
    from mnist_classifier import load_or_train
    clf = load_or_train(args.data_dir,
                        HERE.parent / "58-caption-ablation/checkpoints/classifier.pt")

    print("training teacher ...")
    teacher = train_teacher(args.data_dir, CKPT / "teacher.pt", args.teacher_steps)
    if (CKPT / "student.pt").exists():
        print("loading cached distilled student ...")
        student = copy.deepcopy(teacher)
        student.load_state_dict(torch.load(CKPT / "student.pt")["student"])
        student.eval()
        cm = Consistency(GaussianDiffusion(T=T, schedule="linear", device="cpu").alpha_bar)
    else:
        print("consistency-distilling student ...")
        student, cm = distill(teacher, args.data_dir, CKPT / "student.pt", args.distill_steps)

    shape = (64, 1, 28, 28)
    runs = {
        "teacher DDIM (50 steps)": (teacher_ddim(teacher, cm, 50, shape), 50),
        "student (1 step)": (consistency_sample(student, cm, 1, shape), 1),
        "student (2 steps)": (consistency_sample(student, cm, 2, shape), 2),
        "student (4 steps)": (consistency_sample(student, cm, 4, shape), 4),
    }

    scores = {}
    for name, (imgs, nfe) in runs.items():
        scores[name] = (readability(imgs, clf), nfe)
        print(f"  {name:26s} readability {scores[name][0]:.3f}  (NFE={nfe})")

    plot_samples(runs, OUT / "samples.png")
    plot_tradeoff(scores, OUT / "tradeoff.png")

    lines = ["sampler,nfe,readability"]
    for name, (s, nfe) in scores.items():
        lines.append(f"{name},{nfe},{s:.3f}")
    (OUT / "results.csv").write_text("\n".join(lines) + "\n")
    print(f"\nwrote figures + {OUT/'results.csv'}")


def plot_samples(runs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(runs), 1, figsize=(8, 1.05 * len(runs)), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, (name, (imgs, nfe)) in zip(axes, runs.items()):
        ax.imshow(grid_img(imgs), cmap="gray", vmin=0, vmax=255)
        ax.set_ylabel(name, fontsize=8, rotation=0, ha="right", va="center")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Same model family, fewer steps: teacher (50) vs consistency student (1/2/4)",
                 fontsize=11, color="#0b0b0b")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_tradeoff(scores, path):
    import matplotlib
    matplotlib.use("Agg")
    sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
    import plot_style as ps
    names = list(scores.keys())
    nfe = [scores[n][1] for n in names]
    read = [scores[n][0] for n in names]
    fig, ax = ps.new_axes(6.8, 4.2)
    ax.scatter(nfe, read, s=60, color=ps.SERIES[0], zorder=3)
    for n, x, y in zip(names, nfe, read):
        ax.annotate(n.replace("student ", "").replace(" DDIM (50 steps)", ""),
                    (x, y), textcoords="offset points", xytext=(6, 4),
                    fontsize=8, color=ps.INK_SECONDARY)
    ax.set_xscale("log")
    ax.set_xticks(nfe); ax.set_xticklabels(nfe)
    ax.set_ylim(0.5, 1.0)   # absolute scale so small gaps read as small
    ps.finish(fig, ax, "Quality vs. number of function evaluations (NFE): "
              "1-2 steps nearly match the 50-step teacher",
              "network evaluations per image (log)", "sample readability", path)


if __name__ == "__main__":
    main()
