"""Learned position table vs 2D RoPE: same DiT, same budget, same seed.

Two questions:
1. In-distribution quality — does RoPE at least match learned positions?
2. Resolution extrapolation — sample at 36x36 after training at 28x28.
   The learned table must be bilinearly interpolated (the standard ViT
   patch); RoPE just rotates by the larger indices.

Run (trains both models, ~8 min total on CPU):
    python compare_positions.py
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
sys.path.insert(0, str(HERE.parent / "43-implement-dit-s-2"))
import plot_style as ps  # noqa: E402
from diffusion import GaussianDiffusion  # noqa: E402
from dit import DIT_MINI, DiT  # noqa: E402
from train import EMA, infinite, mnist_loader  # noqa: E402

from rope2d import RoPE2D  # noqa: E402


def build(variant: str) -> DiT:
    if variant == "learned":
        return DiT(**DIT_MINI, learned_pos=True)
    grid = DIT_MINI["image_size"] // DIT_MINI["patch"]
    # A large grid up front: RoPE angles for positions the model may only
    # ever see at sampling time cost nothing.
    rope = RoPE2D(DIT_MINI["dim"] // DIT_MINI["heads"], grid_h=16, grid_w=16)

    class CroppedRoPE:
        """Restrict the precomputed table to the actual token grid."""

        def __init__(self, rope, gh, gw, full=16):
            idx = (torch.arange(gh)[:, None] * full + torch.arange(gw)[None, :]).flatten()
            self.cos, self.sin = rope.cos[:, :, idx], rope.sin[:, :, idx]

        def __call__(self, x):
            x1, x2 = x[..., 0::2], x[..., 1::2]
            out = torch.empty_like(x)
            out[..., 0::2] = x1 * self.cos - x2 * self.sin
            out[..., 1::2] = x1 * self.sin + x2 * self.cos
            return out

    model = DiT(**DIT_MINI, learned_pos=False, rope=None)
    model._rope_full = rope  # kept for resampling at other resolutions
    model._set_grid = lambda gh, gw: [
        setattr(b.attn, "rope", CroppedRoPE(rope, gh, gw)) for b in model.blocks
    ]
    model._set_grid(grid, grid)
    return model


def train(model, steps, data_dir, seed=0, log_rows=None):
    torch.manual_seed(seed)
    ema = EMA(model, 0.995)
    diffusion = GaussianDiffusion(T=300, schedule="linear")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batches = infinite(mnist_loader(data_dir, 64))
    t0 = time.time()
    for step in range(1, steps + 1):
        x0, y = next(batches)
        loss = diffusion.loss(model, x0, model_kwargs={"y": y})
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema.update(model)
        if step % 50 == 0:
            if log_rows is not None:
                log_rows.append((step, f"{loss.item():.4f}"))
            print(f"  step {step}/{steps} | loss {loss.item():.4f} "
                  f"| {step / (time.time() - t0):.2f} it/s", flush=True)
    return ema.shadow, diffusion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    args = ap.parse_args()

    out_dir = HERE / "outputs"
    out_dir.mkdir(exist_ok=True)
    curves = {}
    samples28, samples36 = {}, {}

    for variant in ("learned", "rope"):
        print(f"=== training {variant} ===", flush=True)
        model = build(variant)
        rows = []
        shadow, diffusion = train(model, args.steps, args.data_dir, log_rows=rows)
        curves[variant] = rows
        shadow.eval()

        torch.manual_seed(7)
        y = torch.arange(10).repeat(4)
        x, _ = diffusion.p_sample_loop(shadow, (40, 1, 28, 28), device="cpu",
                                       model_kwargs={"y": y})
        samples28[variant] = x

        # Extrapolation: a 36x36 canvas (9x9 tokens vs the 7x7 trained grid).
        if variant == "rope":
            # Rebuild with rope tables for the larger grid, same weights.
            grid_rope = build("rope")
            grid_rope.load_state_dict(shadow.state_dict())
            grid_rope._set_grid(9, 9)
            target = grid_rope
        else:
            target = shadow  # learned table gets bilinear interpolation
        target.eval()
        torch.manual_seed(7)
        x36, _ = diffusion.p_sample_loop(target, (10, 1, 36, 36), device="cpu",
                                         model_kwargs={"y": torch.arange(10)})
        samples36[variant] = x36

    # Figures.
    fig, ax = ps.new_axes()
    for variant, color in (("learned", ps.SERIES[0]), ("rope", ps.SERIES[1])):
        steps = [int(r[0]) for r in curves[variant]]
        losses = [float(r[1]) for r in curves[variant]]
        ax.plot(steps, losses, color=color, linewidth=2)
        ax.annotate(variant, (steps[-1], losses[-1]), color=ps.INK_SECONDARY,
                    fontsize=10, fontweight="bold", xytext=(6, 0),
                    textcoords="offset points")
    ax.set_yscale("log")
    ax.set_xlim(right=ax.get_xlim()[1] * 1.1)
    ps.finish(fig, ax, "Same DiT, two position encodings — training loss",
              "training step", "noise-prediction MSE", out_dir / "loss_curves.png")

    for name, sam in (("samples28", samples28), ("samples36", samples36)):
        grid = torch.cat([sam["learned"], sam["rope"]])
        nrow = 10
        save_image(F.interpolate(grid, scale_factor=2, mode="nearest"),
                   out_dir / f"{name}.png", nrow=nrow, normalize=True,
                   value_range=(-1, 1))
    with open(out_dir / "loss_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("variant", "step", "loss"))
        for variant, rows in curves.items():
            for r in rows:
                w.writerow((variant, *r))
    print(f"wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
