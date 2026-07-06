"""Style LoRA: the same low-rank trick as project 50, but the target is a
*look* rather than a *subject*. The style here is a 'bold marker' render — a
morphological dilation that thickens every stroke into a heavy weight.

The experiment that proves it learned a style and not a set of pictures:
we only ever show the LoRA styled versions of digits 0-4, then ask it to draw
digits 5-9. If those held-out classes come out styled too, the adapter
captured *how* to render, not *what* was in the training set — generalization,
the line between a useful style LoRA and an overfit one.

Outputs:
  style_examples.png   real digits 0-4 -> their bold-styled targets
  base_vs_style.png    held-out classes 5-9: frozen base / +style LoRA
  params.txt           trainable LoRA parameter count

    python style_lora.py             # ~3 min after a cond base exists
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
sys.path.insert(0, str(HERE.parent / "28-class-conditional-ddpm"))
sys.path.insert(0, str(HERE.parent / "50-lora-fine-tune"))

from diffusion import GaussianDiffusion  # noqa: E402
from unet_conditional import ConditionalUNet  # noqa: E402
from lora import inject_lora, lora_parameters  # noqa: E402

DATA_DIR = HERE / "data"
TRAIN_CLASSES = (0, 1, 2, 3, 4)   # style is only ever shown on these
HELD_OUT = (5, 6, 7, 8, 9)        # ... and tested on these
RANK = 4
STEPS = 1200


def bold_style(x):
    """A 'bold marker' look: morphological dilation with a 3x3 element (in
    [0,1], back to [-1,1]). Thickens every stroke by about a pixel — clearly
    heavier than the base's hand, but still legible, so it is easy to see the
    style has transferred to an unseen class."""
    p = (x + 1) / 2
    return F.max_pool2d(p, 3, 1, 1).clamp(0, 1) * 2 - 1


def styled_loader(data_dir, batch_size):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    ds = datasets.MNIST(str(data_dir), train=True, download=True, transform=tf)
    keep = [i for i, (_, y) in enumerate(zip(ds.data, ds.targets)) if int(y) in TRAIN_CLASSES]
    xs = torch.stack([ds[i][0] for i in keep[:3000]])
    ys = torch.tensor([ds[i][1] for i in keep[:3000]])
    return bold_style(xs), ys


def save(x, path, nrow):
    save_image(F.interpolate(x, scale_factor=2, mode="nearest"), path,
               nrow=nrow, normalize=True, value_range=(-1, 1))
    print(f"wrote {path}")


@torch.no_grad()
def sample(model, diffusion, classes, per=2, seed=0):
    y = torch.tensor(list(classes)).repeat(per)
    g = torch.Generator().manual_seed(seed)
    x_init = torch.randn(y.size(0), 1, 28, 28, generator=g)
    x0, _ = diffusion.p_sample_loop(model, (y.size(0), 1, 28, 28),
                                    x_init=x_init, model_kwargs={"y": y})
    return x0.clamp(-1, 1)


def main(data_dir=DATA_DIR):
    out = HERE / "outputs"; out.mkdir(exist_ok=True)
    ckpt = torch.load(HERE / "checkpoints/cond_base.pt", weights_only=True)
    diffusion = GaussianDiffusion(T=ckpt["T"], schedule="linear", device="cpu")

    styled_x, styled_y = styled_loader(data_dir, 64)

    # a preview of the target style on the training classes
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    raw = datasets.MNIST(str(data_dir), train=True, transform=tf)
    ex = torch.stack([next(raw[i][0] for i in range(len(raw)) if raw[i][1] == c)
                      for c in TRAIN_CLASSES])
    save(torch.cat([ex, bold_style(ex)]), out / "style_examples.png", nrow=5)

    # base samples on the HELD-OUT classes (before styling)
    base = ConditionalUNet(num_classes=ckpt["num_classes"], **ckpt["config"])
    base.load_state_dict(ckpt["ema"]); base.eval()
    base_samples = sample(base, diffusion, HELD_OUT, per=2, seed=1)

    # inject + train the style LoRA on styled 0-4 only
    model = ConditionalUNet(num_classes=ckpt["num_classes"], **ckpt["config"])
    model.load_state_dict(ckpt["ema"])
    adapters = inject_lora(model, rank=RANK, alpha=RANK)
    model.train()
    opt = torch.optim.AdamW(lora_parameters(adapters), lr=1e-3)

    n_lora = sum(p.numel() for p in lora_parameters(adapters))
    (out / "params.txt").write_text(
        f"trainable style-LoRA params: {n_lora:,} (rank {RANK})\n"
        f"trained on styled digits {TRAIN_CLASSES}, tested on {HELD_OUT}.\n")

    t0 = time.time()
    for step in range(1, STEPS + 1):
        idx = torch.randint(0, styled_x.size(0), (32,))
        loss = diffusion.loss(model, styled_x[idx], model_kwargs={"y": styled_y[idx]})
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 150 == 0:
            print(f"step {step}/{STEPS} | loss {loss.item():.4f} | "
                  f"{step / (time.time() - t0):.1f} it/s", flush=True)

    model.eval()
    style_samples = sample(model, diffusion, HELD_OUT, per=2, seed=1)
    save(torch.cat([base_samples, style_samples]), out / "base_vs_style.png", nrow=5)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    args = ap.parse_args()
    main(Path(args.data_dir))
