"""Re-flow (Liu et al., 2022): make a rectified flow's paths actually straight.

Vanilla flow matching pairs each data point with a RANDOM noise point every
batch, so the learned marginal velocity field still has to bend where paths
of different pairs would cross — that is why project 45's paths curve near
the origin and 1-step sampling collapses.

Re-flow fixes the pairing: run the trained model once to produce
(noise, sample) couples, then train a second model on THOSE fixed couples.
Each noise point now has one consistent destination, crossings vanish, and
the optimal paths become genuinely straight — which is what 1-step sampling
needs.

Run (after project 45's train_rf_toy.py):
    python reflow.py            # ~2 min on CPU
"""

import csv
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
sys.path.insert(0, str(HERE.parent / "30-score-matching-from-scratch"))
sys.path.insert(0, str(HERE.parent / "45-rectified-flow-from-scratch"))
import plot_style as ps  # noqa: E402
from rf import euler_sample  # noqa: E402
from toy_data import DATASETS  # noqa: E402
from train_rf_toy import (  # noqa: E402
    DATASET, VelocityNet, plot_fewstep, plot_paths, straightness, train,
)


@torch.no_grad()
def make_couples(model, n: int = 20000, steps: int = 60, seed: int = 5):
    """(noise, sample) pairs from the trained flow — the re-flow dataset."""
    torch.manual_seed(seed)
    eps = torch.randn(n, 2)
    x0, _ = euler_sample(model, eps, steps)
    return x0, eps


def main():
    out_dir = HERE / "outputs"
    out_dir.mkdir(exist_ok=True)

    # Round 1: project 45's model (retrained here for self-containedness).
    ckpt_path = HERE.parent / "45-rectified-flow-from-scratch/checkpoints/rf_toy.pt"
    model1 = VelocityNet()
    if ckpt_path.exists():
        model1.load_state_dict(torch.load(ckpt_path, weights_only=True)["model"])
    else:
        model1 = train()

    print("generating (noise, sample) couples from round-1 model...", flush=True)
    couples = make_couples(model1)

    print("training round-2 model on fixed couples (re-flow)...", flush=True)
    model2 = train(pairs=couples, seed=1)

    # --- measurements ---------------------------------------------------
    rows = [("model", "straightness (1 = straight)", "1-step W-proxy")]
    data_ref = DATASETS[DATASET](4000, generator=torch.Generator().manual_seed(9))

    def one_step_quality(model) -> float:
        """Crude distributional distance for 1-step samples: energy distance
        between sample set and data (adequate to rank two models in 2D)."""
        torch.manual_seed(11)
        x, _ = euler_sample(model, torch.randn(4000, 2), 1)
        d_xy = torch.cdist(x, data_ref).mean()
        d_xx = torch.cdist(x, x).mean()
        d_yy = torch.cdist(data_ref, data_ref).mean()
        return float(2 * d_xy - d_xx - d_yy)

    for name, model in (("round 1 (random pairing)", model1),
                        ("round 2 (re-flowed)", model2)):
        torch.manual_seed(3)
        _, traj = euler_sample(model, torch.randn(200, 2), 60,
                               return_trajectory=True)
        s = straightness(traj)
        q = one_step_quality(model)
        rows.append((name, f"{s:.3f}", f"{q:.4f}"))
        print(f"{name}: straightness {s:.3f} | 1-step energy distance {q:.4f}")
        tag = "before" if model is model1 else "after"
        plot_paths(traj, f"Sampling paths {tag} re-flow",
                   out_dir / f"paths_{tag}.png")

    with open(out_dir / "metrics.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)

    plot_fewstep(model2, out_dir / "few_step_after_reflow.png")


if __name__ == "__main__":
    main()
