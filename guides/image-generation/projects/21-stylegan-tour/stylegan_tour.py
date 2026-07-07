"""Inference tour of a pretrained StyleGAN2 (FFHQ, 1024x1024, config-f).

No training here — this is a 30M-parameter model trained by NVIDIA on 8
GPUs for weeks; the point is to *use* it and see what the W vs W+ latent
spaces actually control. Weights are the community "rosinality format"
checkpoint (the same one JoJoGAN/StyleGAN-NADA build on), fetched once and
cached in checkpoints/ (gitignored, ~130MB).

Three experiments:
  1. Unconditional samples with the truncation trick (single w per image,
     broadcast to all 18 layers — the "W" space).
  2. A truncation sweep on one z, showing the diversity/quality trade-off.
  3. Style mixing: two seeds, swapped at different layer depths (the "W+"
     space — each of the 18 layers can carry a *different* code), showing
     which layers control coarse structure (pose, face shape) vs fine
     detail (color, texture).

    python stylegan_tour.py   # ~2 min on CPU (24 forward passes @ ~4.3s each)
"""

import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
CKPT_DIR = HERE / "checkpoints"
CKPT_PATH = CKPT_DIR / "stylegan2-ffhq-config-f.pt"
CKPT_URL = ("https://huggingface.co/akhaliq/jojogan-stylegan2-ffhq-config-f/"
            "resolve/main/stylegan2-ffhq-config-f.pt")

sys.path.insert(0, str(HERE))
from sg2_model import Generator  # noqa: E402

SIZE = 1024
STYLE_DIM = 512
N_MLP = 8


def load_generator():
    CKPT_DIR.mkdir(exist_ok=True)
    if not CKPT_PATH.exists():
        print(f"downloading {CKPT_URL} (~130MB)...")
        urllib.request.urlretrieve(CKPT_URL, CKPT_PATH)
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    G = Generator(SIZE, STYLE_DIM, N_MLP, channel_multiplier=2)
    G.load_state_dict(ckpt["g_ema"])
    G.eval()
    for p in G.parameters():
        p.requires_grad_(False)
    return G, ckpt["latent_avg"]


def to_uint8(img):
    return ((img.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).numpy()


@torch.no_grad()
def run():
    torch.set_num_threads(12)
    torch.manual_seed(0)
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    G, latent_avg = load_generator()
    print(f"loaded generator in {time.time() - t0:.0f}s")

    # 1) Unconditional samples, truncation psi=0.7 (the usual "good quality,
    # still diverse" sweet spot).
    n_samples = 8
    z = torch.randn(n_samples, STYLE_DIM)
    imgs = []
    for i in range(n_samples):
        img, _ = G([z[i:i + 1]], truncation=0.7, truncation_latent=latent_avg.unsqueeze(0))
        imgs.append(to_uint8(img)[0])
    np.save(OUT / "samples.npy", np.stack(imgs))
    print(f"1/3 samples done ({time.time() - t0:.0f}s)")

    # 2) Truncation sweep on one seed.
    z_fixed = torch.randn(1, STYLE_DIM)
    psis = [0.0, 0.25, 0.5, 0.75, 1.0]
    trunc_imgs = []
    for psi in psis:
        img, _ = G([z_fixed], truncation=psi, truncation_latent=latent_avg.unsqueeze(0))
        trunc_imgs.append(to_uint8(img)[0])
    np.save(OUT / "truncation.npy", np.stack(trunc_imgs))
    np.save(OUT / "truncation_psis.npy", np.array(psis))
    print(f"2/3 truncation sweep done ({time.time() - t0:.0f}s)")

    # 3) Style mixing across crossover depths: source A supplies layers
    # [0, inject_index), source B supplies [inject_index, 18). Layers run
    # coarse -> fine, so a low inject_index gives B almost total control
    # (only A's coarsest pose/shape survives) and a high inject_index gives
    # A total control (only B's finest color/texture survives).
    # Hand-picked for a visibly distinct pair (young woman vs older bearded
    # man) so the coarse/fine crossover is unambiguous; drawn at truncation=1
    # (no truncation) since the truncation trick pulls faces toward the mean
    # and would mute exactly the identity contrast we want to show.
    torch.manual_seed(42)
    candidates = torch.randn(6, STYLE_DIM)
    z_a, z_b = candidates[0:1], candidates[5:6]
    img_a, _ = G([z_a], truncation=1.0)
    img_b, _ = G([z_b], truncation=1.0)

    crossovers = [2, 4, 8, 12, 16]
    mixed_imgs = [to_uint8(img_a)[0]]
    for idx in crossovers:
        img_mix, _ = G([z_a, z_b], truncation=1.0, inject_index=idx)
        mixed_imgs.append(to_uint8(img_mix)[0])
    mixed_imgs.append(to_uint8(img_b)[0])
    np.save(OUT / "style_mixing.npy", np.stack(mixed_imgs))
    np.save(OUT / "style_mixing_crossovers.npy", np.array([0] + crossovers + [18]))
    print(f"3/3 style mixing done ({time.time() - t0:.0f}s)")

    print(f"total: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    run()
