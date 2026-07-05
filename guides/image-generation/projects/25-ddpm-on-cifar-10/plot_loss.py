"""Plot the CIFAR-10 training loss curve from train_cifar.py's CSV log.

Run:
    python plot_loss.py
"""

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
import plot_style as ps  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(HERE / "outputs/train_log.csv"))
    ap.add_argument("--out", default=str(HERE / "outputs/loss_curve.png"))
    args = ap.parse_args()

    with open(args.log) as f:
        rows = list(csv.DictReader(f))
    steps = [int(r["step"]) for r in rows]
    losses = [float(r["loss"]) for r in rows]

    fig, ax = ps.new_axes()
    ax.plot(steps, losses, color=ps.SERIES[0], linewidth=2)
    ax.set_yscale("log")
    ps.finish(fig, ax, "DDPM on CIFAR-10 — smoke-run loss (log scale)",
              "training step", "noise-prediction MSE", args.out)


if __name__ == "__main__":
    main()
