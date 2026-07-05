"""Train a conditional DDPM with label dropout — the CFG training recipe.

The only change versus project 28's conditional training: with probability
10% the true label is replaced by a reserved NULL class (index 10), so one
network learns both p(x | y) and p(x). That single line is what makes
classifier-free guidance possible at inference.

Run:
    python train_cfg.py            # ~3 min on a multicore CPU
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
sys.path.insert(0, str(HERE.parent / "28-class-conditional-ddpm"))
from diffusion import GaussianDiffusion  # noqa: E402
from train import EMA, infinite, mnist_loader  # noqa: E402
from unet_conditional import ConditionalUNet  # noqa: E402

NULL_CLASS = 10  # embedding index reserved for "no condition"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--schedule", default="linear")
    ap.add_argument("--p-uncond", type=float, default=0.1, help="label dropout rate")
    ap.add_argument("--ema-decay", type=float, default=0.995)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--out", default=str(HERE / "checkpoints/cfg_mnist.pt"))
    ap.add_argument("--log", default=str(HERE / "outputs/train_log.csv"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    model = ConditionalUNet(num_classes=11).to(device)  # 10 digits + NULL
    ema = EMA(model, args.ema_decay)
    diffusion = GaussianDiffusion(T=args.T, schedule=args.schedule, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    batches = infinite(mnist_loader(args.data_dir, args.batch_size))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    log_rows = [("step", "loss")]
    running, t0 = 0.0, time.time()

    for step in range(1, args.steps + 1):
        x0, y = next(batches)
        x0, y = x0.to(device), y.to(device)
        # THE line: sometimes hide the label so the model also learns p(x).
        drop = torch.rand(y.shape, device=device) < args.p_uncond
        y = torch.where(drop, torch.full_like(y, NULL_CLASS), y)

        loss = diffusion.loss(model, x0, model_kwargs={"y": y})
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema.update(model)

        running += loss.item()
        if step % 50 == 0:
            log_rows.append((step, f"{running / 50:.4f}"))
            print(f"step {step:5d}/{args.steps} | loss {running / 50:.4f} "
                  f"| {step / (time.time() - t0):.2f} it/s", flush=True)
            running = 0.0

    torch.save({"ema": ema.shadow.state_dict(), "T": args.T,
                "schedule": args.schedule, "p_uncond": args.p_uncond}, args.out)
    with open(args.log, "w", newline="") as f:
        csv.writer(f).writerows(log_rows)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
