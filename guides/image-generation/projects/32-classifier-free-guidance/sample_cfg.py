"""Classifier-free guidance at inference: run the model twice per step and
extrapolate away from the unconditional prediction.

    eps_cfg = eps_uncond + s * (eps_cond - eps_uncond)

s = 1 is plain conditional sampling; s = 0 is unconditional. The sweep here
(1 -> 12) maps the fidelity/diversity trade-off, scored by a small MNIST
classifier: "purity" = fraction classified as the requested digit,
"diversity" = mean pairwise distance in the classifier's feature space.

Run (after train_cfg.py):
    python sample_cfg.py
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
sys.path.insert(0, str(HERE.parent / "26-cosine-vs-linear-schedule"))
sys.path.insert(0, str(HERE.parent / "28-class-conditional-ddpm"))
import plot_style as ps  # noqa: E402
from diffusion import GaussianDiffusion  # noqa: E402
from feature_net import load_feature_net  # noqa: E402
from unet_conditional import ConditionalUNet  # noqa: E402

from train_cfg import NULL_CLASS  # noqa: E402


class CFGModel:
    """Wraps the trained model as eps(x, t) with a per-sample guidance scale,
    so project 24's sampling loop can drive it unchanged."""

    def __init__(self, model, y: torch.Tensor, scale: torch.Tensor):
        self.model, self.y = model, y
        self.scale = scale.view(-1, 1, 1, 1)
        self.null = torch.full_like(y, NULL_CLASS)

    def __call__(self, x, t):
        eps_cond = self.model(x, t, self.y)
        eps_uncond = self.model(x, t, self.null)
        return eps_uncond + self.scale * (eps_cond - eps_uncond)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE / "checkpoints/cfg_mnist.pt"))
    ap.add_argument("--target", type=int, default=3)
    ap.add_argument("--n", type=int, default=8, help="samples per scale in the grid")
    ap.add_argument("--n-metric", type=int, default=32,
                    help="samples per scale for purity/diversity (needs more than the grid)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--out-dir", default=str(HERE / "outputs"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    device = args.device
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    model = ConditionalUNet(num_classes=11).to(device)
    model.load_state_dict(ckpt["ema"])
    model.eval()
    diffusion = GaussianDiffusion(T=ckpt["T"], schedule=ckpt["schedule"], device=device)

    scales = (1.0, 3.0, 5.0, 8.0, 12.0)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # One batched run: all scales share the same starting noise rows.
    torch.manual_seed(args.seed)
    x_T = torch.randn(args.n, 1, 28, 28, device=device).repeat(len(scales), 1, 1, 1)
    y = torch.full((len(scales) * args.n,), args.target, device=device, dtype=torch.long)
    scale_vec = torch.tensor(scales, device=device).repeat_interleave(args.n)
    cfg = CFGModel(model, y, scale_vec)
    x0, _ = diffusion.p_sample_loop(cfg, x_T.shape, device=device, x_init=x_T)
    save_image(F.interpolate(x0, scale_factor=2, mode="nearest"),
               out_dir / "scale_sweep.png", nrow=args.n, normalize=True,
               value_range=(-1, 1))

    # Score each scale with an independent classifier, on a larger batch than
    # the display grid so the curves are not sampling noise.
    torch.manual_seed(args.seed + 2)
    m = args.n_metric
    y_m = torch.full((len(scales) * m,), args.target, device=device, dtype=torch.long)
    scale_m = torch.tensor(scales, device=device).repeat_interleave(m)
    x_m, _ = diffusion.p_sample_loop(CFGModel(model, y_m, scale_m),
                                     (len(scales) * m, 1, 28, 28), device=device)

    net = load_feature_net(HERE / "checkpoints/feature_net.pt", args.data_dir, device)
    rows = [("scale", "purity", "diversity")]
    purities, diversities = [], []
    with torch.no_grad():
        for i, s in enumerate(scales):
            batch = x_m[i * m : (i + 1) * m]
            feats = net.features(batch)
            purity = (net(batch).argmax(1) == args.target).float().mean().item()
            diversity = torch.cdist(feats, feats).mean().item()
            rows.append((s, f"{purity:.2f}", f"{diversity:.2f}"))
            purities.append(purity)
            diversities.append(diversity)
            print(f"scale {s:5.1f} | purity {purity:.0%} | diversity {diversity:.2f}")
    with open(out_dir / "sweep_metrics.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)

    # A grid of all ten classes at a moderate scale.
    torch.manual_seed(args.seed + 1)
    y_all = torch.arange(10, device=device).repeat(8)
    s_all = torch.full((80,), 3.0, device=device)
    cfg_all = CFGModel(model, y_all, s_all)
    x_all, _ = diffusion.p_sample_loop(cfg_all, (80, 1, 28, 28), device=device)
    save_image(F.interpolate(x_all, scale_factor=2, mode="nearest"),
               out_dir / "class_grid_s3.png", nrow=10, normalize=True,
               value_range=(-1, 1))

    # Purity and diversity vs scale, one small chart each (different units).
    for name, vals in (("purity", purities), ("diversity", diversities)):
        fig, ax = ps.new_axes(5.4, 3.4)
        ax.plot(scales, vals, color=ps.SERIES[0], linewidth=2, marker="o", markersize=5)
        ax.set_xticks(scales)
        ps.finish(fig, ax, f"CFG scale vs {name}", "guidance scale s", name,
                  out_dir / f"{name}_vs_scale.png")


if __name__ == "__main__":
    main()
