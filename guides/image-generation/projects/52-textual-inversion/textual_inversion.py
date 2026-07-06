"""Textual Inversion at MNIST scale: freeze the ENTIRE class-conditional base
and learn one new thing only — a single embedding vector for a new token V that
points at the subject. Nothing in the U-Net moves; we optimize `temb_dim`
numbers (here 128) and nothing else. That vector is the whole artifact — a few
hundred bytes, the smallest personalization there is.

Mechanism: the base has a class-embedding table (one row per digit). A normal
prompt looks a row up and adds it to the time embedding. Our new token has no
row; instead we splice a learnable vector in wherever the label equals V, and
backprop reaches only that vector.

Outputs:
  subject.png        the 20 training images
  ti_result.png      row 1: real bags / row 2: samples for the learned token V
  digits_intact.png  prompts 0..9 — untouched, because the model never changed
  params.txt         artifact size vs LoRA (project 50) and DreamBooth (51)

    python textual_inversion.py      # ~2 min after a cond base exists
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms
from torchvision.utils import save_image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
sys.path.insert(0, str(HERE.parent / "28-class-conditional-ddpm"))

from diffusion import GaussianDiffusion  # noqa: E402
from unet_conditional import ConditionalUNet  # noqa: E402

DATA_DIR = HERE / "data"
SUBJECT_CLASS = 8      # FashionMNIST 'Bag'
N_SUBJECT = 20
V = 10                 # new token id (one past the 10 real digit classes)
STEPS = 800


class InvertedToken(nn.Module):
    """Frozen base + one trainable token vector spliced in at label == V."""

    def __init__(self, base: ConditionalUNet, init: torch.Tensor):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.token = nn.Parameter(init.clone())  # the ONLY trainable tensor

    def forward(self, x, t, y):
        is_v = (y == V)[:, None].float()
        rows = self.base.label_emb(y.clamp(max=V - 1))     # safe lookup
        emb = rows * (1 - is_v) + self.token[None] * is_v  # splice token at V
        return self.base.forward_with_temb(x, self.base.time_embedding(t) + emb)


def load_fashion(data_dir, cls, n):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    ds = datasets.FashionMNIST(str(data_dir), train=True, download=True, transform=tf)
    xs, i = [], 0
    while len(xs) < n:
        x, y = ds[i]
        if y == cls:
            xs.append(x)
        i += 1
    return torch.stack(xs)


def save(x, path, nrow):
    save_image(F.interpolate(x, scale_factor=2, mode="nearest"), path,
               nrow=nrow, normalize=True, value_range=(-1, 1))
    print(f"wrote {path}")


@torch.no_grad()
def sample(model, diffusion, y, seed):
    g = torch.Generator().manual_seed(seed)
    x_init = torch.randn(y.size(0), 1, 28, 28, generator=g)
    x0, _ = diffusion.p_sample_loop(model, (y.size(0), 1, 28, 28),
                                    x_init=x_init, model_kwargs={"y": y})
    return x0.clamp(-1, 1)


def main(data_dir=DATA_DIR):
    out = HERE / "outputs"; out.mkdir(exist_ok=True)
    ckpt = torch.load(HERE / "checkpoints/cond_base.pt", weights_only=True)
    diffusion = GaussianDiffusion(T=ckpt["T"], schedule="linear", device="cpu")

    base = ConditionalUNet(num_classes=ckpt["num_classes"], **ckpt["config"])
    base.load_state_dict(ckpt["ema"]); base.eval()

    subject = load_fashion(data_dir, SUBJECT_CLASS, N_SUBJECT)
    save(subject, out / "subject.png", nrow=10)

    init = base.label_emb.weight.mean(0).detach()
    model = InvertedToken(base, init)
    opt = torch.optim.AdamW([model.token], lr=5e-2)  # one vector -> big LR ok

    t0 = time.time()
    for step in range(1, STEPS + 1):
        idx = torch.randint(0, subject.size(0), (16,))
        y = torch.full((16,), V)
        loss = diffusion.loss(model, subject[idx], model_kwargs={"y": y})
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            print(f"step {step}/{STEPS} | loss {loss.item():.4f} | "
                  f"{step / (time.time() - t0):.1f} it/s", flush=True)

    torch.save({"token": model.token.detach()}, HERE / "checkpoints/token.pt")

    v_samples = sample(model, diffusion, torch.full((10,), V), seed=5)
    save(torch.cat([subject[:10], v_samples]), out / "ti_result.png", nrow=10)
    digits = sample(model, diffusion, torch.arange(10), seed=6)
    save(digits, out / "digits_intact.png", nrow=10)

    n_token = model.token.numel()
    (out / "params.txt").write_text(
        f"Textual Inversion trains ONE vector: {n_token} parameters "
        f"({n_token * 4} bytes as float32).\n"
        f"Compare: LoRA ~ tens of thousands (project 50); "
        f"DreamBooth ~ the entire {sum(p.numel() for p in ckpt['ema'].values()):,}"
        f"-param model (project 51).\n")
    print((out / "params.txt").read_text())


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    args = ap.parse_args()
    main(Path(args.data_dir))
