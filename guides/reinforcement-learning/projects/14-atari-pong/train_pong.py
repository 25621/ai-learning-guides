"""DQN on pixels: the Atari recipe, on a Pong that fits in ten minutes.

Three experiments, all on MiniPong (Pong against a wall, 20x20 pixels, +1 per
rally returned, -1 and done on a miss, episode caps at 10 rallies):

  1. the pipeline itself   -- what the network actually sees, drawn from REAL
                              ALE Pong frames if `ale_py` is installed
  2. frame stacking        -- k = 1 vs k = 4. A single frame cannot say which way
                              the ball is moving; four of them can. This is the
                              experiment that justifies the wrapper.
  3. reward clipping       -- give the env a rare jackpot (every 5th rally pays
                              20 instead of 1) and train with and without
                              `sign(r)`. Both are scored on the plain game.

                              Note the identity that makes this cheap: clipping
                              the jackpot game to sign(r) turns every rally back
                              into +1 and every miss into -1 -- which IS the plain
                              game, transition for transition. So the clipped arm
                              needs no runs of its own; the plain stack-4 runs are
                              already exactly it. That equivalence is the whole
                              point of clipping, so the script leans on it rather
                              than spending compute rediscovering it.

Runtime: ~5 min on 12 CPU cores (9 training runs in parallel, 1 thread each).
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-build-a-gridworld"))
sys.path.insert(0, str(HERE.parent / "13-add-a-replay-buffer"))

import matplotlib.pyplot as plt  # noqa: E402
from dqn_lib import Config, DQNAgent, ReplayBuffer, train_dqn  # noqa: E402
from plot_style import INK, INK_MUTED, INK_SECONDARY, SERIES, SURFACE, finish, new_axes  # noqa: E402
from pong_lib import GRID, MAX_RALLIES, ConvQNet, make_minipong  # noqa: E402

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

TOTAL_STEPS = 50_000
SEEDS = 3
JACKPOT = 19.0          # every 5th rally pays 1 + 19 = 20, the rest pay 1

BASE = Config(
    total_steps=TOTAL_STEPS,
    gamma=0.99,
    lr=1e-3,
    batch_size=32,
    buffer_size=20_000,
    learning_starts=1_000,
    train_freq=2,          # one gradient step per two env steps: half the compute,
    target_freq=500,       # and a lower replay ratio, which is if anything steadier
    eps_start=1.0,
    eps_end=0.05,
    eps_decay_frac=0.3,
    eval_every=5_000,
    eval_episodes=10,
)


def run(job):
    """One MiniPong run. Always evaluated on the PLAIN game (no jackpot, no clipping),
    so every variant is scored by the only thing that matters: rallies returned."""
    name, seed, stack, clip, rally_bonus = job
    torch.set_num_threads(1)

    cfg = Config(**{**BASE.__dict__, "seed": seed})
    env = make_minipong(seed=seed, stack=stack, clip=clip, rally_bonus=rally_bonus)
    eval_env = make_minipong(seed=seed + 1000, stack=stack)   # plain game, plain scoring
    agent = DQNAgent(lambda: ConvQNet(in_frames=stack), env.n_actions, cfg)
    buffer = ReplayBuffer(cfg.buffer_size, env.obs_shape, obs_dtype=np.uint8)

    hist = train_dqn(env, agent, buffer, cfg, eval_env=eval_env)
    return {
        "name": name, "seed": seed,
        "eval_step": np.array(hist["eval_step"]),
        "eval_return": np.array(hist["eval_return"]),
        "q_pred": np.array(hist["q_pred"]),
        "step": np.array(hist["step"]),
    }


# --------------------------------------------------------------------------
# Figure 1: what the agent actually sees
# --------------------------------------------------------------------------

def plot_atari_pipeline():
    """Real ALE Pong: raw console frame -> preprocessed 84x84 -> the 4-frame stack."""
    try:
        from pong_lib import AtariPipeline
        pipe = AtariPipeline(seed=0)
    except Exception as exc:                       # noqa: BLE001
        print(f"skipping the ALE figure ({exc.__class__.__name__}: {exc}).")
        print("  `pip install ale_py` to see what real Pong looks like through the wrappers.")
        return False

    pipe.reset()
    rng = np.random.default_rng(0)
    for _ in range(40):                            # let the ball get into play
        pipe.step(int(rng.integers(pipe.n_actions)))
    raw = pipe.raw_frame
    stack = pipe.frames

    fig = plt.figure(figsize=(11.0, 3.4), dpi=110)
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(1, 6, width_ratios=[1.35, 1.35, 1, 1, 1, 1], wspace=0.18)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(raw)
    ax.set_title("raw ALE frame\n210x160x3 RGB", fontsize=9, color=INK, loc="left")
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(stack[-1], cmap="gray", vmin=0, vmax=1)
    ax.set_title("preprocessed\n84x84 gray, cropped", fontsize=9, color=INK, loc="left")
    ax.axis("off")

    for i in range(4):
        ax = fig.add_subplot(gs[0, 2 + i])
        ax.imshow(stack[i], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"stack[{i}]" + ("  (now)" if i == 3 else f"  (t-{3 - i})"),
                     fontsize=8, color=INK_SECONDARY, loc="left")
        ax.axis("off")

    fig.suptitle("The Atari pipeline: what DQN is really handed, 4 frames at a time",
                 color=INK, fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    path = OUT / "atari_pipeline.png"
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
    pipe.close()
    return True


def plot_minipong_filmstrip():
    """Eight consecutive MiniPong frames: the ball's motion lives BETWEEN them."""
    env = make_minipong(seed=3, stack=1)
    obs, _ = env.reset(seed=3)
    frames = [obs[0].copy()]
    for t in range(7):
        obs, r, term, trunc, _ = env.step(2 if t % 2 == 0 else 0)
        frames.append(obs[0].copy())
        if term:
            break

    fig, axes = plt.subplots(1, len(frames), figsize=(1.25 * len(frames), 1.9), dpi=110)
    fig.patch.set_facecolor(SURFACE)
    for i, (ax, f) in enumerate(zip(axes, frames)):
        ax.imshow(f, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"t+{i}", fontsize=8, color=INK_SECONDARY)
        ax.axis("off")
    fig.suptitle(f"MiniPong, {GRID}x{GRID} pixels: ball (white), paddle (gray). "
                 "One frame hides the ball's direction — that is why we stack four.",
                 color=INK, fontsize=10, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    path = OUT / "minipong_filmstrip.png"
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# --------------------------------------------------------------------------
# Figures 2 and 3: the ablations
# --------------------------------------------------------------------------

def band(ax, runs, color, label):
    xs = runs[0]["eval_step"]
    ys = np.stack([r["eval_return"] for r in runs])
    ax.fill_between(xs, ys.min(0), ys.max(0), color=color, alpha=0.15, linewidth=0)
    ax.plot(xs, np.median(ys, axis=0), color=color, linewidth=1.9, marker="o",
            markersize=3.5, label=label)


def group(results, name):
    # "clip (sign r)" has no runs of its own: clipping the jackpot game reproduces
    # the plain game exactly, so the plain stack-4 runs ARE the clipped arm.
    if name == "clip (sign r)":
        name = "stack 4"
    return [r for r in results if r["name"] == name]


def plot_stack_ablation(results):
    fig, ax = new_axes(7.6, 4.4)
    band(ax, group(results, "stack 4"), SERIES[0], "4 stacked frames")
    band(ax, group(results, "stack 1"), SERIES[2], "1 frame (no motion)")
    ax.axhline(MAX_RALLIES, color=INK_MUTED, linestyle="--", linewidth=1.0)
    ax.text(TOTAL_STEPS * 0.99, MAX_RALLIES - 0.7, "perfect play (10 rallies)",
            ha="right", color=INK_MUTED, fontsize=9)
    ax.axhline(-1, color=INK_MUTED, linestyle=":", linewidth=1.0)
    ax.text(TOTAL_STEPS * 0.99, -0.85, "random play (≈ -0.7)", ha="right",
            color=INK_MUTED, fontsize=9)
    ax.set_ylim(-1.6, 11)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    finish(fig, ax, "Frame stacking is not a detail: without it the ball has no velocity",
           "environment step", "rallies returned (median of 3 seeds, min–max band)",
           OUT / "frame_stacking.png")


def plot_clip_ablation(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.2), dpi=110)
    fig.patch.set_facecolor(SURFACE)
    from plot_style import style_axes
    for ax in (ax1, ax2):
        style_axes(ax)

    for name, color in (("clip (sign r)", SERIES[1]), ("no clip", SERIES[2])):
        runs = group(results, name)
        xs = runs[0]["eval_step"]
        ys = np.stack([r["eval_return"] for r in runs])
        ax1.fill_between(xs, ys.min(0), ys.max(0), color=color, alpha=0.15, linewidth=0)
        ax1.plot(xs, np.median(ys, 0), color=color, linewidth=1.9, marker="o",
                 markersize=3.5, label=name)
        qs = np.stack([r["q_pred"] for r in runs])
        ax2.plot(runs[0]["step"], np.median(qs, 0), color=color, linewidth=1.4, label=name)

    ax1.axhline(MAX_RALLIES, color=INK_MUTED, linestyle="--", linewidth=1.0)
    ax1.set_ylim(-1.6, 11)
    ax1.legend(frameon=False, fontsize=9, loc="upper left")
    ax1.set_title("Score on the plain game", color=INK, fontsize=11, loc="left", pad=10)
    ax1.set_xlabel("environment step", color=INK_SECONDARY, fontsize=10)
    ax1.set_ylabel("rallies returned (median of 3 seeds)", color=INK_SECONDARY, fontsize=10)

    ax2.legend(frameon=False, fontsize=9, loc="upper left")
    ax2.set_title("Mean predicted Q on the training batch", color=INK, fontsize=11,
                  loc="left", pad=10)
    ax2.set_xlabel("environment step", color=INK_SECONDARY, fontsize=10)
    ax2.set_ylabel("Q", color=INK_SECONDARY, fontsize=10)

    fig.suptitle(f"Reward clipping: the same game, but every 5th rally secretly pays "
                 f"{int(JACKPOT + 1)}",
                 color=INK, fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = OUT / "reward_clipping.png"
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def summarize(results, names):
    """`last-3` (mean of the final three checkpoints) is the headline: a single final
    evaluation is one noisy 10-episode sample, and these curves oscillate enough that
    quoting it would be a coin flip dressed up as a result."""
    print(f"\n{'variant':<16}  {'last-3 score':>12}  {'best score':>10}  {'peak Q':>8}")
    for name in names:
        runs = group(results, name)
        last3 = np.mean([r["eval_return"][-3:].mean() for r in runs])
        best = np.mean([r["eval_return"].max() for r in runs])
        peak_q = max(r["q_pred"].max() for r in runs)
        print(f"{name:<16}  {last3:12.2f}  {best:10.2f}  {peak_q:8.2f}")


def main():
    plot_atari_pipeline()
    plot_minipong_filmstrip()

    jobs = []
    for seed in range(SEEDS):
        jobs.append(("stack 4", seed, 4, False, 0.0))          # doubles as the clipped arm
        jobs.append(("stack 1", seed, 1, False, 0.0))
        jobs.append(("no clip", seed, 4, False, JACKPOT))

    print(f"\n{len(jobs)} MiniPong runs x {TOTAL_STEPS:,} steps, in parallel...")
    with ProcessPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(run, jobs))

    print("\n--- frame stacking (plain game) ---")
    summarize(results, ["stack 4", "stack 1"])
    print(f"\n--- reward clipping (trained with a {int(JACKPOT + 1)}-point jackpot, "
          f"scored on the plain game) ---")
    print("    [the clipped arm needs no runs: sign(r) maps the jackpot game back")
    print("     onto the plain game, so the plain stack-4 runs already ARE it]")
    summarize(results, ["clip (sign r)", "no clip"])

    plot_stack_ablation(results)
    plot_clip_ablation(results)


if __name__ == "__main__":
    main()
