"""Build the synthetic instruction-edit dataset on the fly and train the
one-pass InstructPix2Pix editor.

Outputs:
  data_triples.png    top row: originals; each row below: the procedural target
                      for one instruction — this IS the training data
  editor_edits.png    top row: held-out test originals; each row below: the
                      trained editor's single-loop edit for that instruction

    python train_editor.py           # ~4 min on CPU
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
from train import EMA  # noqa: E402
from editor import INSTRUCTIONS, InstructEditor, apply_edit  # noqa: E402

DATA_DIR = HERE / "data"
STEPS = 1400
T = 200


def load_digits(data_dir, train, n=None):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    ds = datasets.MNIST(str(data_dir), train=train, download=True, transform=tf)
    xs = torch.stack([ds[i][0] for i in range(n or len(ds))])
    return xs


def save(x, path, nrow):
    save_image(F.interpolate(x, scale_factor=2, mode="nearest"), path,
               nrow=nrow, normalize=True, value_range=(-1, 1))
    print(f"wrote {path}")


@torch.no_grad()
def edit(model, diffusion, original, instr_id, seed=0):
    n = original.size(0)
    instr = torch.full((n,), instr_id)
    g = torch.Generator().manual_seed(seed)
    x_init = torch.randn(n, 1, 28, 28, generator=g)
    x0, _ = diffusion.p_sample_loop(
        model, (n, 1, 28, 28), x_init=x_init,
        model_kwargs={"original": original, "instr": instr})
    return x0.clamp(-1, 1)


def main(data_dir=DATA_DIR):
    out = HERE / "outputs"; out.mkdir(exist_ok=True)
    torch.manual_seed(0)
    pool = load_digits(data_dir, train=True, n=5000)
    diffusion = GaussianDiffusion(T=T, schedule="linear", device="cpu")
    model = InstructEditor()
    ema = EMA(model, 0.995)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    print(f"editor params: {sum(p.numel() for p in model.parameters()):,}")

    # show the synthetic data: originals + procedural target per instruction
    demo = pool[:10]
    rows = [demo] + [apply_edit(demo, i) for i in range(len(INSTRUCTIONS))]
    save(torch.cat(rows), out / "data_triples.png", nrow=10)

    t0 = time.time()
    for step in range(1, STEPS + 1):
        idx = torch.randint(0, pool.size(0), (48,))
        original = pool[idx]
        instr_id = step % len(INSTRUCTIONS)        # cycle through instructions
        edited = apply_edit(original, instr_id)
        instr = torch.full((48,), instr_id)
        loss = diffusion.loss(model, edited,
                              model_kwargs={"original": original, "instr": instr})
        opt.zero_grad(); loss.backward(); opt.step(); ema.update(model)
        if step % 200 == 0:
            print(f"step {step}/{STEPS} | loss {loss.item():.4f} | "
                  f"{step / (time.time() - t0):.1f} it/s", flush=True)

    ema.shadow.eval()
    test = load_digits(data_dir, train=False, n=10)
    rows = [test] + [edit(ema.shadow, diffusion, test, i, seed=i)
                     for i in range(len(INSTRUCTIONS))]
    save(torch.cat(rows), out / "editor_edits.png", nrow=10)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    args = ap.parse_args()
    main(Path(args.data_dir))
