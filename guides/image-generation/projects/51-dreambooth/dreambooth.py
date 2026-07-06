"""DreamBooth at MNIST scale: full fine-tune of the WHOLE class-conditional
base on a handful of subject images, binding the subject to a fresh trigger
token V (a new class id = 10). The digit classes 0-9 play the role of the
text prompts; V is 'a photo of [V] bag'.

The point of the project is the trade-off, so we train two ways and compare:

  * WITH prior preservation: every step mixes the 5 subject images (labeled V)
    with a batch of generic base-distribution images (labeled with their real
    class). The extra term is DreamBooth's prior-preservation loss; it stops
    the five bags from bulldozing everything the model knew about digits.
  * WITHOUT it: train only on the 5 bags. The model learns the bag and
    *forgets how to draw digits* — catastrophic forgetting, made visible.

Outputs:
  subject.png              the 5 training images
  db_with_prior.png        prompts 0..9 then V — digits survive, V is a bag
  db_no_prior.png          prompts 0..9 then V — digits have collapsed
  params.txt               full-model trainable count (vs LoRA in project 50)

    python dreambooth.py             # ~4 min after train_cond_base.py
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

from diffusion import GaussianDiffusion  # noqa: E402
from unet_conditional import ConditionalUNet  # noqa: E402

DATA_DIR = HERE / "data"
SUBJECT_CLASS = 8      # FashionMNIST 'Bag'
V = 10                 # the new trigger token id
N_SUBJECT = 5
STEPS = 600
PRIOR_WEIGHT = 1.0


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


def load_prior(data_dir, n=512):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    ds = datasets.MNIST(str(data_dir), train=True, download=True, transform=tf)
    xs = torch.stack([ds[i][0] for i in range(n)])
    ys = torch.tensor([ds[i][1] for i in range(n)])
    return xs, ys


def expand_to_V(ckpt):
    """Load a 10-class base into an 11-class model; init token V as the mean
    of the existing class embeddings."""
    model = ConditionalUNet(num_classes=V + 1, **ckpt["config"])
    base_sd = ckpt["ema"]
    own = model.state_dict()
    emb = base_sd["label_emb.weight"]
    new_emb = own["label_emb.weight"].clone()
    new_emb[:V] = emb
    new_emb[V] = emb.mean(0)
    base_sd = {**base_sd, "label_emb.weight": new_emb}
    model.load_state_dict(base_sd)
    return model


@torch.no_grad()
def prompt_grid(model, diffusion, seed=3, per=1):
    ys = torch.arange(V + 1).repeat(per)  # rows of 0..V, one full prompt set each
    g = torch.Generator().manual_seed(seed)
    x_init = torch.randn(ys.size(0), 1, 28, 28, generator=g)
    x0, _ = diffusion.p_sample_loop(model, (ys.size(0), 1, 28, 28),
                                    x_init=x_init, model_kwargs={"y": ys})
    return x0.clamp(-1, 1)


def finetune(base_ckpt, diffusion, subject, prior, use_prior):
    model = expand_to_V(base_ckpt)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    prior_x, prior_y = prior
    t0 = time.time()
    for step in range(1, STEPS + 1):
        idx = torch.randint(0, subject.size(0), (8,))
        y_v = torch.full((8,), V)
        loss = diffusion.loss(model, subject[idx], model_kwargs={"y": y_v})
        if use_prior:
            j = torch.randint(0, prior_x.size(0), (8,))
            loss = loss + PRIOR_WEIGHT * diffusion.loss(
                model, prior_x[j], model_kwargs={"y": prior_y[j]})
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 150 == 0:
            print(f"  [{'prior' if use_prior else 'no-prior'}] step {step}/{STEPS} "
                  f"| loss {loss.item():.4f} | {step / (time.time() - t0):.1f} it/s",
                  flush=True)
    model.eval()
    return model


def save(x, path, nrow):
    save_image(F.interpolate(x, scale_factor=2, mode="nearest"), path,
               nrow=nrow, normalize=True, value_range=(-1, 1))
    print(f"wrote {path}")


def main(data_dir=DATA_DIR):
    out = HERE / "outputs"; out.mkdir(exist_ok=True)
    ckpt = torch.load(HERE / "checkpoints/cond_base.pt", weights_only=True)
    diffusion = GaussianDiffusion(T=ckpt["T"], schedule="linear", device="cpu")

    subject = load_fashion(data_dir, SUBJECT_CLASS, N_SUBJECT)
    prior = load_prior(data_dir)
    save(subject, out / "subject.png", nrow=N_SUBJECT)

    n_full = sum(p.numel() for p in ckpt["ema"].values())
    (out / "params.txt").write_text(
        f"DreamBooth updates EVERY weight: {n_full:,} trainable parameters.\n"
        f"The saved artifact is the full model — the opposite of LoRA's few %.\n")

    print("fine-tuning WITH prior preservation ...")
    m_prior = finetune(ckpt, diffusion, subject, prior, use_prior=True)
    save(prompt_grid(m_prior, diffusion, per=3),
         out / "db_with_prior.png", nrow=V + 1)

    print("fine-tuning WITHOUT prior preservation ...")
    m_no = finetune(ckpt, diffusion, subject, prior, use_prior=False)
    save(prompt_grid(m_no, diffusion, per=3),
         out / "db_no_prior.png", nrow=V + 1)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    args = ap.parse_args()
    main(Path(args.data_dir))
