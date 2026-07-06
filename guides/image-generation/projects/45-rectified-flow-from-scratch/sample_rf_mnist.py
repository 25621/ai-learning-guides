"""Few-step Euler sampling from the MNIST rectified-flow model: one grid
row per step count (1, 2, 4, 8, 16), same starting noise.

Run (after train_rf_mnist.py):
    python sample_rf_mnist.py
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))

from rf import euler_sample  # noqa: E402
from train_rf_mnist import VelocityUNet  # noqa: E402


def main():
    device = "cpu"
    ckpt = torch.load(HERE / "checkpoints/rf_mnist.pt", map_location=device,
                      weights_only=True)
    model = VelocityUNet().to(device)
    model.load_state_dict(ckpt["ema"])
    model.eval()

    torch.manual_seed(7)
    x_init = torch.randn(8, 1, 28, 28)
    rows = []
    for n in (1, 2, 4, 8, 16):
        x, _ = euler_sample(model, x_init, n)
        rows.append(x.clamp(-1, 1))
        print(f"{n} steps done", flush=True)

    out_dir = HERE / "outputs"
    out_dir.mkdir(exist_ok=True)
    save_image(F.interpolate(torch.cat(rows), scale_factor=2, mode="nearest"),
               out_dir / "few_step_mnist.png", nrow=8, normalize=True,
               value_range=(-1, 1))
    print(f"wrote {out_dir}/few_step_mnist.png (rows: 1, 2, 4, 8, 16 steps)")


if __name__ == "__main__":
    main()
