"""Invert real MNIST digits into diffusion noise, then denoise with a changed
prompt to edit them while keeping their layout.

Two figures:
  recon.png   real (row 1) vs invert->denoise with the SAME label (row 2).
              The gap is inversion drift — the reconstruction is close but not
              pixel-perfect, exactly what null-text inversion later fixes.
  edit.png    real 4s (row 1) -> inverted, then denoised as class 9 (row 2).
              The stroke thickness, slant and position carry over; only the
              identity the prompt describes changes.

    python edit.py                   # ~1 min after a cond base exists

Run a cond base first (see README).
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
sys.path.insert(0, str(HERE.parent / "28-class-conditional-ddpm"))

from diffusion import GaussianDiffusion  # noqa: E402
from unet_conditional import ConditionalUNet  # noqa: E402
from ddim_invert import ddim_denoise, ddim_invert  # noqa: E402

DATA_DIR = HERE / "data"
STEPS = 100


def reals_of_class(data_dir, cls, n):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    ds = datasets.MNIST(str(data_dir), train=False, download=True, transform=tf)
    xs, i = [], 0
    while len(xs) < n:
        x, y = ds[i]
        if y == cls:
            xs.append(x)
        i += 1
    return torch.stack(xs)


def one_per_class(data_dir):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    ds = datasets.MNIST(str(data_dir), train=False, download=True, transform=tf)
    got = {}
    for i in range(len(ds)):
        x, y = ds[i]
        if y not in got:
            got[y] = x
        if len(got) == 10:
            break
    return torch.stack([got[c] for c in range(10)]), torch.arange(10)


def save(x, path, nrow):
    save_image(F.interpolate(x, scale_factor=2, mode="nearest"), path,
               nrow=nrow, normalize=True, value_range=(-1, 1))
    print(f"wrote {path}")


def main(data_dir=DATA_DIR):
    out = HERE / "outputs"; out.mkdir(exist_ok=True)
    ckpt = torch.load(HERE / "checkpoints/cond_base.pt", weights_only=True)
    diffusion = GaussianDiffusion(T=ckpt["T"], schedule="linear", device="cpu")
    model = ConditionalUNet(num_classes=ckpt["num_classes"], **ckpt["config"])
    model.load_state_dict(ckpt["ema"]); model.eval()

    # evenly spaced timestep subsequence, walked up (invert) then down (denoise)
    ts = torch.linspace(0, diffusion.T - 1, STEPS).round().long().tolist()
    ab = diffusion.alpha_bar

    # --- reconstruction: invert then denoise with the true label ---
    reals, ys = one_per_class(data_dir)
    xT = ddim_invert(model, reals, ts, ab, model_kwargs={"y": ys})
    recon = ddim_denoise(model, xT, ts, ab, model_kwargs={"y": ys})
    save(torch.cat([reals, recon.clamp(-1, 1)]), out / "recon.png", nrow=10)

    # --- edit: invert 4s as class 4, denoise as class 9 ---
    fours = reals_of_class(data_dir, 4, 10)
    y_src = torch.full((10,), 4)
    y_tgt = torch.full((10,), 9)
    xT4 = ddim_invert(model, fours, ts, ab, model_kwargs={"y": y_src})
    edited = ddim_denoise(model, xT4, ts, ab, model_kwargs={"y": y_tgt})
    save(torch.cat([fours, edited.clamp(-1, 1)]), out / "edit.png", nrow=10)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    args = ap.parse_args()
    main(Path(args.data_dir))
