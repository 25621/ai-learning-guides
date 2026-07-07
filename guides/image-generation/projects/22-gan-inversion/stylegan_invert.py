"""Optimization-based GAN inversion at production scale: pretrained
StyleGAN2-FFHQ (1024x1024), inverting into the W+ space.

We don't have a real photo to invert against (no license to embed one, and no
internet-connected face dataset available offline), so the "real" target is
itself a StyleGAN2 sample generated from a known z — a standard sanity check
in the inversion literature (if you can't even re-find your own generator's
outputs, you have no chance on genuinely out-of-distribution photos). The
target used a single shared w (plain W, broadcast to all 18 layers); we
deliberately invert into the *more expressive* W+ (an independent code per
layer) to show W+ can match — or beat — the truth it's searching for.

Each Adam step here costs a full 1024x1024 forward + backward through a
30M-parameter network (~4.5s on this CPU), so the step budget is small by
deep-learning standards but still enough to see clear convergence.

    python stylegan_invert.py   # ~5 min on CPU (70 optimization steps)
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
STYLEGAN_DIR = HERE.parent / "21-stylegan-tour"

sys.path.insert(0, str(STYLEGAN_DIR))
from sg2_model import Generator  # noqa: E402

STEPS = 70
LOG_EVERY = 10


def to_uint8(img):
    return ((img.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).numpy()


def load_generator():
    ckpt_path = STYLEGAN_DIR / "checkpoints" / "stylegan2-ffhq-config-f.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    G = Generator(1024, 512, 8, channel_multiplier=2)
    G.load_state_dict(ckpt["g_ema"])
    G.eval()
    for p in G.parameters():
        p.requires_grad_(False)
    return G, ckpt["latent_avg"]


def run():
    torch.set_num_threads(12)
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    G, latent_avg = load_generator()

    # The "real" target: a single-w StyleGAN2 sample (truncation=0.9, hand-picked
    # seed/index for a clean face — untruncated sampling occasionally produces
    # artifacted outliers that would make a confusing target image).
    torch.manual_seed(10)
    candidates = torch.randn(6, 512)
    z_true = candidates[2:3]
    with torch.no_grad():
        w_true = G.get_latent(z_true)              # (1, 512) — the true w
        target, _ = G([z_true], truncation=0.9, truncation_latent=latent_avg.unsqueeze(0),
                       randomize_noise=False)
    target = target.detach()
    w_true_plus = w_true.unsqueeze(1).repeat(1, G.n_latent, 1)  # what W+ equal to the truth looks like

    # Optimize a full W+ code (one independent 512-d vector per of the 18
    # layers) initialized at the dataset average face.
    w_hat = latent_avg.view(1, 1, -1).repeat(1, G.n_latent, 1).clone().requires_grad_(True)
    opt = torch.optim.Adam([w_hat], lr=0.05)

    history = []
    snapshots = []
    for step in range(STEPS):
        opt.zero_grad()
        img, _ = G([w_hat], input_is_latent=True, randomize_noise=False)
        loss = F.mse_loss(img, target)
        loss.backward()
        opt.step()
        history.append((step, loss.item()))
        if step % LOG_EVERY == 0 or step == STEPS - 1:
            print(f"step {step} mse={loss.item():.4f} ({time.time() - t0:.0f}s)")
            with torch.no_grad():
                snapshots.append(to_uint8(img)[0])

    with torch.no_grad():
        final_img, _ = G([w_hat], input_is_latent=True, randomize_noise=False)
        w_mse = F.mse_loss(w_hat, w_true_plus).item()
        pixel_mse = F.mse_loss(final_img, target).item()
        psnr = 10 * np.log10(4.0 / pixel_mse)  # pixels in [-1,1], range^2 = 4

    np.savez(OUT / "stylegan_invert.npz",
              history=np.array(history), target=to_uint8(target)[0],
              final=to_uint8(final_img)[0], snapshots=np.stack(snapshots),
              w_mse=w_mse, pixel_mse=pixel_mse, psnr=psnr, wall_time=time.time() - t0)
    print(f"done in {time.time() - t0:.0f}s | pixel MSE {pixel_mse:.4f} | PSNR {psnr:.1f}dB | "
          f"w+ vs true-w MSE {w_mse:.4f}")


if __name__ == "__main__":
    run()
