"""Train a from-scratch ControlNet on MNIST: condition an edge map, learn to
fill it in. The base U-Net stays frozen; only the control branch + zero-convs
learn.

Outputs:
  init_identity.png   proof the zero-convs make the branch a no-op at step 0:
                      frozen-base sample vs controlled-model sample are equal
  control_result.png  row 1: test digits / row 2: their edge maps (the control
                      signal) / row 3: images generated FROM those edges

    python train_controlnet.py       # ~4 min after a base exists

Make a base first:
    python ../50-lora-fine-tune/train_base.py --out checkpoints/base_mnist.pt
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))

from diffusion import GaussianDiffusion  # noqa: E402
from train import infinite  # noqa: E402
from unet import UNet  # noqa: E402
from controlnet import ControlledUNet, edge_map  # noqa: E402

DATA_DIR = HERE / "data"
STEPS = 1200


def loader(data_dir, bs):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    ds = datasets.MNIST(str(data_dir), train=True, download=True, transform=tf)
    return DataLoader(ds, batch_size=bs, shuffle=True, num_workers=2, drop_last=True)


def save(x, path, nrow):
    save_image(F.interpolate(x, scale_factor=2, mode="nearest"), path,
               nrow=nrow, normalize=True, value_range=(-1, 1))
    print(f"wrote {path}")


def main(data_dir=DATA_DIR):
    out = HERE / "outputs"; out.mkdir(exist_ok=True)
    torch.manual_seed(0)
    ckpt = torch.load(HERE / "checkpoints/base_mnist.pt", weights_only=True)
    T = ckpt["T"]
    diffusion = GaussianDiffusion(T=T, schedule="linear", device="cpu")
    base = UNet(); base.load_state_dict(ckpt["ema"]); base.eval()
    model = ControlledUNet(base)

    n_ctrl = sum(p.numel() for p in model.control_parameters())
    print(f"control-branch params: {n_ctrl:,} (base frozen)")

    # zero-conv identity check at init, saved as a figure
    test_ds = datasets.MNIST(str(data_dir), train=False, download=True,
                             transform=transforms.Compose(
                                 [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]))
    test = torch.stack([test_ds[i][0] for i in range(8)])
    hints = edge_map(test)
    g = torch.Generator().manual_seed(3)
    x_init = torch.randn(8, 1, 28, 28, generator=g)
    # Seed the global RNG identically before each loop so both draw the SAME
    # per-step sampling noise — then any difference between the rows is the
    # model, not the sampler. At init the zero-convs make the control branch a
    # no-op, so the two rows come out identical (proof the branch starts inert).
    with torch.no_grad():
        torch.manual_seed(100)
        base_s, _ = diffusion.p_sample_loop(base, (8, 1, 28, 28), x_init=x_init)
        torch.manual_seed(100)
        ctrl_s, _ = diffusion.p_sample_loop(
            model, (8, 1, 28, 28), x_init=x_init, model_kwargs={"hint": hints})
    save(torch.cat([base_s.clamp(-1, 1), ctrl_s.clamp(-1, 1)]),
         out / "init_identity.png", nrow=8)

    opt = torch.optim.AdamW(model.control_parameters(), lr=3e-4)
    batches = infinite(loader(data_dir, 64))
    t0 = time.time()
    for step in range(1, STEPS + 1):
        x0, _ = next(batches)
        loss = diffusion.loss(model, x0, model_kwargs={"hint": edge_map(x0)})
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 150 == 0:
            print(f"step {step}/{STEPS} | loss {loss.item():.4f} | "
                  f"{step / (time.time() - t0):.1f} it/s", flush=True)

    with torch.no_grad():
        gen, _ = diffusion.p_sample_loop(
            model, (8, 1, 28, 28), x_init=x_init, model_kwargs={"hint": hints})
    save(torch.cat([test, hints, gen.clamp(-1, 1)]), out / "control_result.png", nrow=8)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    args = ap.parse_args()
    main(Path(args.data_dir))
