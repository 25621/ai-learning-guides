"""Sample a labeled grid from the class-conditional DDPM: one column per
class, several samples per column.

Run:
    python sample_classes.py --ckpt checkpoints/conditional.pt
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

from unet_conditional import ConditionalUNet  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE / "checkpoints/conditional.pt"))
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(HERE / "outputs/class_grid.png"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=True)
    model = ConditionalUNet(num_classes=10, **ckpt["config"]).to(args.device)
    model.load_state_dict(ckpt["ema"])
    model.eval()
    diffusion = GaussianDiffusion(T=ckpt["T"], schedule=ckpt["schedule"], device=args.device)

    # Row-major grid: every row asks for classes 0..9 in order.
    y = torch.arange(10, device=args.device).repeat(args.rows)
    in_ch = ckpt["config"]["in_ch"]
    size = ckpt["config"]["image_size"]
    x0, _ = diffusion.p_sample_loop(
        model, (y.size(0), in_ch, size, size), device=args.device, model_kwargs={"y": y}
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    x0 = F.interpolate(x0, scale_factor=2, mode="nearest")
    save_image(x0, args.out, nrow=10, normalize=True, value_range=(-1, 1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
