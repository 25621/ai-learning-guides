"""Per-class sample grid from the trained mini-DiT.

Run (after train_dit.py):
    python sample_dit.py
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
from diffusion import GaussianDiffusion  # noqa: E402

from dit import DiT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE / "checkpoints/dit_mini.pt"))
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--out", default=str(HERE / "outputs/class_grid.png"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    device = "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    model = DiT(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["ema"])
    model.eval()
    diffusion = GaussianDiffusion(T=ckpt["T"], schedule=ckpt["schedule"], device=device)

    torch.manual_seed(args.seed)
    y = torch.arange(10, device=device).repeat(args.rows)
    x0, _ = diffusion.p_sample_loop(model, (y.size(0), 1, 28, 28), device=device,
                                    model_kwargs={"y": y})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_image(F.interpolate(x0, scale_factor=2, mode="nearest"), args.out,
               nrow=10, normalize=True, value_range=(-1, 1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
