"""Fine-tune a LoRA on ~20 images of a custom subject, on top of the frozen
MNIST base. The 'subject' the base has never seen is a FashionMNIST *bag* — a
solid blob that looks nothing like a thin digit stroke, so it is obvious when
the adapter has taught the base something new.

Produces, in outputs/:
  subject.png          the 20 training images
  base_vs_lora.png     row 1: frozen base (digits) / row 2: +LoRA (bags)
  lora_steps.png       samples after 150 / 400 / 1200 LoRA steps (under->over)
  params.txt           trainable-parameter accounting

    python train_lora.py            # ~2 min after train_base.py

Run train_base.py first.
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))

from diffusion import GaussianDiffusion  # noqa: E402
from unet import UNet  # noqa: E402
from lora import inject_lora, lora_parameters  # noqa: E402

DATA_DIR = HERE / "data"
SUBJECT_CLASS = 8   # FashionMNIST 'Bag'
N_SUBJECT = 20
RANK = 4


def subject_images(data_dir, n=N_SUBJECT):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    ds = datasets.FashionMNIST(str(data_dir), train=True, download=True, transform=tf)
    xs, i = [], 0
    while len(xs) < n:
        x, y = ds[i]
        if y == SUBJECT_CLASS:
            xs.append(x)
        i += 1
    return torch.stack(xs)  # (n, 1, 28, 28)


@torch.no_grad()
def sample_grid(model, diffusion, n=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    x_init = torch.randn(n, 1, 28, 28, generator=g)
    x0, _ = diffusion.p_sample_loop(model, (n, 1, 28, 28), x_init=x_init)
    return x0.clamp(-1, 1)


def save(x, path, nrow=8):
    save_image(F.interpolate(x, scale_factor=2, mode="nearest"), path,
               nrow=nrow, normalize=True, value_range=(-1, 1))
    print(f"wrote {path}")


def main(data_dir=DATA_DIR):
    out = HERE / "outputs"; out.mkdir(exist_ok=True)
    ckpt = torch.load(HERE / "checkpoints/base_mnist.pt", weights_only=True)
    T = ckpt["T"]
    diffusion = GaussianDiffusion(T=T, schedule="linear", device="cpu")

    # --- base samples (before any adaptation) ---
    base = UNet(); base.load_state_dict(ckpt["ema"]); base.eval()
    base_samples = sample_grid(base, diffusion, seed=1)

    subject = subject_images(data_dir)
    save(subject, out / "subject.png", nrow=10)

    # --- inject LoRA and fine-tune only the adapters ---
    model = UNet(); model.load_state_dict(ckpt["ema"])
    adapters = inject_lora(model, rank=RANK, alpha=RANK)
    model.train()

    n_base = sum(p.numel() for p in ckpt["ema"].values())
    n_lora = sum(p.numel() for p in lora_parameters(adapters))
    acct = (f"frozen base parameters : {n_base:,}\n"
            f"trainable LoRA params  : {n_lora:,}  (rank {RANK})\n"
            f"LoRA as % of base      : {100 * n_lora / n_base:.2f}%\n")
    (out / "params.txt").write_text(acct)
    print(acct)

    opt = torch.optim.AdamW(lora_parameters(adapters), lr=1e-3)
    snapshots = {}
    snap_at = (150, 400, 1200)
    t0 = time.time()
    for step in range(1, max(snap_at) + 1):
        idx = torch.randint(0, subject.size(0), (16,))
        loss = diffusion.loss(model, subject[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if step in snap_at:
            model.eval()
            snapshots[step] = sample_grid(model, diffusion, seed=1)
            model.train()
            print(f"step {step} | loss {loss.item():.4f} | "
                  f"{step / (time.time() - t0):.1f} it/s", flush=True)

    save(torch.cat([base_samples, snapshots[snap_at[-1]]]),
         out / "base_vs_lora.png", nrow=8)
    save(torch.cat([snapshots[s] for s in snap_at]),
         out / "lora_steps.png", nrow=8)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    args = ap.parse_args()
    main(Path(args.data_dir))
