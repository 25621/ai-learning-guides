"""Sample a grid from a trained CIFAR-10 DDPM checkpoint.

Run:
    python sample_cifar.py --ckpt checkpoints/cifar_smoke.pt
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "24-ddpm-on-mnist"))
from diffusion import GaussianDiffusion  # noqa: E402
from unet import UNet  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/cifar_smoke.pt")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="outputs/samples.png")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=True)
    model = UNet(**ckpt["config"]).to(args.device)
    model.load_state_dict(ckpt["ema"])
    model.eval()

    diffusion = GaussianDiffusion(T=ckpt["T"], schedule=ckpt["schedule"], device=args.device)
    x0, _ = diffusion.p_sample_loop(model, (args.n, 3, 32, 32), device=args.device)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    x0 = F.interpolate(x0, scale_factor=2, mode="nearest")
    save_image(x0, args.out, nrow=8, normalize=True, value_range=(-1, 1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
