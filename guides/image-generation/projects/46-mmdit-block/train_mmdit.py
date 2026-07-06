"""Train the mini-MMDiT with flow matching (the SD3/Flux pairing), then
sample a per-class grid and visualize the joint attention.

Run:
    python train_mmdit.py            # ~5 min on a multicore CPU
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
sys.path.insert(0, str(HERE.parent / "45-rectified-flow-from-scratch"))
import plot_style as ps  # noqa: E402
from rf import euler_sample, rf_loss  # noqa: E402
from train import EMA, infinite, mnist_loader  # noqa: E402

from mmdit import MiniMMDiT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--out", default=str(HERE / "checkpoints/mmdit.pt"))
    ap.add_argument("--log", default=str(HERE / "outputs/train_log.csv"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    model = MiniMMDiT().to(device)
    print(f"MMDiT params: {sum(p.numel() for p in model.parameters()):,}")
    ema = EMA(model, 0.995)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    batches = infinite(mnist_loader(args.data_dir, args.batch_size))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    rows = [("step", "loss")]
    running, t0 = 0.0, time.time()
    for step in range(1, args.steps + 1):
        x0, y = next(batches)
        loss = rf_loss(model, x0.to(device), model_kwargs={"y": y.to(device)})
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema.update(model)
        running += loss.item()
        if step % 50 == 0:
            rows.append((step, f"{running / 50:.4f}"))
            print(f"step {step:5d}/{args.steps} | loss {running / 50:.4f} "
                  f"| {step / (time.time() - t0):.2f} it/s", flush=True)
            running = 0.0

    torch.save({"ema": ema.shadow.state_dict()}, args.out)
    with open(args.log, "w", newline="") as f:
        csv.writer(f).writerows(rows)

    # --- per-class grid, 16 Euler steps ----------------------------------
    shadow = ema.shadow
    shadow.eval()
    out_dir = HERE / "outputs"
    torch.manual_seed(7)
    y = torch.arange(10).repeat(8)
    x, _ = euler_sample(shadow, torch.randn(80, 1, 28, 28), 16,
                        model_kwargs={"y": y})
    save_image(F.interpolate(x.clamp(-1, 1), scale_factor=2, mode="nearest"),
               out_dir / "class_grid.png", nrow=10, normalize=True,
               value_range=(-1, 1))

    # --- joint-attention visualization ------------------------------------
    # How much does each image patch attend to the class ("text") tokens,
    # midway through generation? High mass where the digit's strokes are.
    torch.manual_seed(9)
    y1 = torch.tensor([3] * 4)
    x_mid, _ = euler_sample(shadow, torch.randn(4, 1, 28, 28), 8,
                            model_kwargs={"y": y1})
    with torch.no_grad():
        t_mid = torch.full((4,), 0.5)
        x_noised = 0.5 * x_mid + 0.5 * torch.randn_like(x_mid)
        _, attn = shadow(x_noised, t_mid, y1, return_attn=True)
    # attn: (B, heads, N_img+N_txt, N_img+N_txt). Image-query -> txt-key mass.
    n_img = shadow.grid**2
    img_to_txt = attn[:, :, :n_img, n_img:].sum(-1).mean(1)  # (B, N_img)
    maps = img_to_txt.reshape(-1, 1, shadow.grid, shadow.grid)

    fig_imgs = torch.cat([
        F.interpolate(x_mid.clamp(-1, 1), scale_factor=2, mode="nearest"),
        F.interpolate(2 * (maps / maps.amax(dim=(2, 3), keepdim=True)) - 1,
                      scale_factor=8, mode="nearest"),
    ])
    save_image(fig_imgs, out_dir / "attention_to_text.png", nrow=4,
               normalize=True, value_range=(-1, 1))
    print(f"wrote {out_dir}/class_grid.png and attention_to_text.png")


if __name__ == "__main__":
    main()
