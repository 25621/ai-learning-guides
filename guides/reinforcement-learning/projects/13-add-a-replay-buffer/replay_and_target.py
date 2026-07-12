"""The 2x2: {replay on/off} x {target network on/off}, five seeds each, on CartPole.

Project 12 showed the naive agent diverging by eight orders of magnitude. This
script adds DQN's two stabilizers back one at a time and measures which one is
doing the work — the same agent, the same net, the same 40k steps, four corners:

    no replay,  no target   <- project 12's agent, the control
    no replay, + target
  + replay,     no target
  + replay,    + target     <- actual DQN

Then two follow-ups:
  * hard target copies (every N updates) vs Polyak averaging;
  * a diagnostic that measures the thing replay is *supposed* to fix — the
    correlation between the states inside one gradient batch.

Runtime: ~5 min on 12 CPU cores (29 runs, spread across cores, 1 thread each).
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-build-a-gridworld"))

import matplotlib.pyplot as plt  # noqa: E402
from dqn_lib import (Config, DQNAgent, MLPQNet, OnlineBuffer,  # noqa: E402
                     ReplayBuffer, train_dqn)
from plot_style import (INK, INK_MUTED, INK_SECONDARY, SERIES,  # noqa: E402
                        SURFACE, finish, new_axes, style_axes)

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

TOTAL_STEPS = 60_000
GAMMA = 0.99
SEEDS = 5
SOLVED = 475.0
Q_CEILING = (1 - GAMMA ** 500) / (1 - GAMMA)   # 99.3: CartPole cannot pay more than this

BASE = Config(
    total_steps=TOTAL_STEPS,
    gamma=GAMMA,
    lr=1e-3,
    batch_size=64,
    buffer_size=50_000,
    learning_starts=1_000,
    target_freq=500,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay_frac=0.25,
    eval_every=5_000,
    eval_episodes=5,
)

# The four corners of the ablation, plus the two target-update styles.
VARIANTS = {
    "no replay, no target": dict(use_replay=False, use_target=False),
    "no replay, + target": dict(use_replay=False, use_target=True),
    "+ replay, no target": dict(use_replay=True, use_target=False),
    "+ replay, + target": dict(use_replay=True, use_target=True),
}
TARGET_STYLES = {
    "hard copy every 500": dict(target_freq=500, tau=0.0),
    "hard copy every 2000": dict(target_freq=2_000, tau=0.0),
    "Polyak, tau = 0.005": dict(target_freq=0, tau=0.005),
}


def build(cfg, use_replay, obs_dim, n_actions):
    """Same agent either way; only the buffer changes."""
    agent = DQNAgent(lambda: MLPQNet(obs_dim, n_actions), n_actions, cfg)
    if use_replay:
        buffer = ReplayBuffer(cfg.buffer_size, (obs_dim,))
    else:
        # batch of one, always the newest transition: replay switched off
        buffer = OnlineBuffer((obs_dim,))
    return agent, buffer


def run(job):
    name, seed, overrides = job
    torch.set_num_threads(1)

    use_replay = overrides.pop("use_replay", True)
    cfg = replace(BASE, seed=seed, **overrides)
    if not use_replay:
        cfg = replace(cfg, learning_starts=1, batch_size=1)

    env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")
    agent, buffer = build(cfg, use_replay, 4, 2)
    hist = train_dqn(env, agent, buffer, cfg, eval_env=eval_env)
    env.close()
    eval_env.close()

    return {
        "name": name, "seed": seed,
        "eval_step": np.array(hist["eval_step"]),
        "eval_return": np.array(hist["eval_return"]),
        "step": np.array(hist["step"]),
        "q_pred": np.array(hist["q_pred"]),
    }


def group(results, name):
    return [r for r in results if r["name"] == name]


def band(ax, runs, key, color, label):
    """Median across seeds, with the min-max band behind it."""
    xs = runs[0]["eval_step"] if key == "eval_return" else runs[0]["step"]
    ys = np.stack([r[key] for r in runs])
    med = np.median(ys, axis=0)
    ax.fill_between(xs, ys.min(axis=0), ys.max(axis=0), color=color, alpha=0.15, linewidth=0)
    ax.plot(xs, med, color=color, linewidth=1.8, label=label)


def plot_ablation(results):
    fig, ax = new_axes(7.8, 4.6)
    for i, name in enumerate(VARIANTS):
        band(ax, group(results, name), "eval_return", SERIES[i], name)
    ax.axhline(SOLVED, color=INK_MUTED, linestyle="--", linewidth=1.0)
    ax.text(TOTAL_STEPS * 0.99, SOLVED + 12, "solved (475)", ha="right",
            color=INK_MUTED, fontsize=9)
    ax.set_ylim(0, 540)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    finish(fig, ax, "Only the corner with both stabilizers actually learns",
           "environment step", "greedy evaluation return (median of 5 seeds, min–max band)",
           OUT / "ablation_2x2.png")


def plot_q_drift(results):
    fig, ax = new_axes(7.8, 4.6)
    for i, name in enumerate(VARIANTS):
        runs = group(results, name)
        xs = runs[0]["step"]
        ys = np.median(np.stack([np.maximum(r["q_pred"], 1e-2) for r in runs]), axis=0)
        ax.plot(xs, ys, color=SERIES[i], linewidth=1.5, label=name)
    ax.set_yscale("log")
    ax.axhline(Q_CEILING, color=INK_MUTED, linestyle="--", linewidth=1.0)
    ax.text(TOTAL_STEPS * 0.99, Q_CEILING * 1.7, f"true ceiling ≈ {Q_CEILING:.0f}",
            ha="right", color=INK_MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    finish(fig, ax, "The target network is what keeps the Q-values honest",
           "environment step", "mean predicted Q on the training batch  [log scale]",
           OUT / "q_drift.png")


def plot_target_styles(results):
    fig, ax = new_axes(7.8, 4.4)
    for i, name in enumerate(TARGET_STYLES):
        band(ax, group(results, name), "eval_return", SERIES[i + 1], name)
    ax.axhline(SOLVED, color=INK_MUTED, linestyle="--", linewidth=1.0)
    ax.text(TOTAL_STEPS * 0.99, SOLVED + 12, "solved (475)", ha="right",
            color=INK_MUTED, fontsize=9)
    ax.set_ylim(0, 540)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    finish(fig, ax, "How you refresh the target matters less than that you have one",
           "environment step", "greedy evaluation return (median of 5 seeds)",
           OUT / "target_update_styles.png")


# CartPole's four dimensions live on wildly different scales (angle is radians and
# never exceeds 0.21; velocity is unbounded). Distances are measured in units of
# each dimension's natural range so that no single dimension dominates. This is a
# FIXED scale, not one derived from the data -- standardizing per-policy would
# divide the balancing policy's near-zero angle variance by itself and manufacture
# exactly the illusion this diagnostic exists to dispel.
STATE_SCALE = np.array([2.4, 2.0, 0.21, 2.0])


def collect_states(policy, n=20_000, seed=0):
    """Returns states, an end-of-episode mask, and the mean episode length."""
    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    states, ends, ep_len, lengths = [], [], 0, []
    for _ in range(n):
        obs, _, term, trunc, _ = env.step(policy(obs, rng))
        states.append(np.asarray(obs, dtype=np.float64))
        ep_len += 1
        done = term or trunc
        ends.append(done)
        if done:
            lengths.append(ep_len)
            ep_len = 0
            obs, _ = env.reset()
    env.close()
    return (np.stack(states) / STATE_SCALE, np.array(ends),
            float(np.mean(lengths)) if lengths else float(n))


def random_policy(obs, rng):
    return int(rng.integers(2))


def balancing_policy(obs, rng):
    """A hand-written controller that actually keeps the pole up: push in the
    direction the pole is falling, anticipating with its angular velocity. It is
    the stand-in for a COMPETENT agent, which is the regime that matters."""
    angle, angular_velocity = obs[2], obs[3]
    return int(angle + 0.5 * angular_velocity > 0)


def batch_correlation_diagnostic():
    """Measure what replay is *for*, in the units the no-replay agent actually uses.

    The no-replay agent takes ONE gradient step per transition, so the question is
    not "how varied is a 64-row batch" -- it never has one. The question is how far
    the training input MOVES between one gradient step and the next.

    Without replay it moves from `s_t` to `s_{t+1}`: two states 20 ms apart in a
    physics simulation, which is barely any move at all. With replay it jumps to a
    random row of the buffer -- and the buffer holds the agent's whole HISTORY, not
    just its current behavior, which is the part that matters. A real buffer mixes
    the flailing of early training with the competent balancing of late training,
    so a sampled row is unrelated to the current moment in a way that no
    consecutive pair of states can ever be.

    So: successive states are drawn from a competent trajectory (the regime where
    correlation bites hardest -- episodes run the full 500 steps, and the agent's
    own success is what keeps its states similar), and the buffer is the mixture an
    agent actually accumulates.
    """
    rng = np.random.default_rng(0)

    random_states, _, random_ep = collect_states(random_policy, n=10_000, seed=0)
    good_states, good_ends, good_ep = collect_states(balancing_policy, n=10_000, seed=0)

    # what the no-replay agent trains on: s_t then s_{t+1}, within one episode
    keep = ~good_ends[:-1]                        # skip the jump across a reset
    successive = np.linalg.norm(np.diff(good_states, axis=0), axis=-1)[keep]

    # what replay trains on: two random rows of a buffer holding the whole history
    buffer = np.concatenate([random_states, good_states])
    i, j = rng.integers(0, len(buffer), size=(2, 20_000))
    sampled = np.linalg.norm(buffer[i] - buffer[j], axis=-1)

    fig, ax = new_axes(7.6, 4.2)
    bins = np.linspace(0, np.percentile(sampled, 99.5), 60)
    ax.hist(successive, bins=bins, color=SERIES[2], alpha=0.85,
            label=f"no replay: s(t) → s(t+1)  —  mean {successive.mean():.3f}")
    ax.hist(sampled, bins=bins, color=SERIES[1], alpha=0.7,
            label=f"replay: two random rows of the buffer  —  mean {sampled.mean():.3f}")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    finish(fig, ax,
           f"Successive gradient steps see nearly the same state — "
           f"{sampled.mean() / successive.mean():.0f}× closer than replay's",
           "distance between the states used in successive gradient steps "
           "(units of each dimension's natural range)",
           "count  [log scale]", OUT / "batch_diversity.png")

    print(f"  random-policy episodes last ~{random_ep:.0f} steps; "
          f"competent ones last ~{good_ep:.0f}")
    print(f"  no replay: successive states are {successive.mean():.3f} apart")
    print(f"  replay:    sampled states are   {sampled.mean():.3f} apart "
          f"({sampled.mean() / successive.mean():.0f}x further)")
    return successive, sampled


def summarize(results, names, title):
    """`last-3` (mean of the final three evaluations) is the honest headline number:
    a single final evaluation is noisy, and DQN on CartPole oscillates enough that
    cherry-picking one point would flatter it."""
    print(f"\n--- {title} ---")
    print(f"{'variant':<24}  {'last-3 eval':>11}  {'best eval':>9}  "
          f"{'hit 475':>8}  {'peak Q':>10}")
    rows = {}
    for name in names:
        runs = group(results, name)
        last3 = np.mean([r["eval_return"][-3:].mean() for r in runs])
        best = np.mean([r["eval_return"].max() for r in runs])
        solved = sum(r["eval_return"].max() >= SOLVED for r in runs)
        peak_q = max(r["q_pred"].max() for r in runs)
        rows[name] = (last3, best, solved, peak_q)
        print(f"{name:<24}  {last3:11.1f}  {best:9.1f}  {solved:>4}/{len(runs)}  "
              f"{peak_q:10.3g}")
    return rows


def main():
    jobs = [(name, seed, dict(ov)) for name, ov in VARIANTS.items() for seed in range(SEEDS)]
    jobs += [(name, seed, dict(ov, use_replay=True))
             for name, ov in TARGET_STYLES.items() for seed in range(3)]

    print(f"{len(jobs)} runs x {TOTAL_STEPS:,} steps, in parallel across cores...")
    with ProcessPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(run, jobs))

    summarize(results, list(VARIANTS), "the 2x2 (5 seeds each)")
    summarize(results, list(TARGET_STYLES), "target-update style (replay on, 3 seeds each)")

    plot_ablation(results)
    plot_q_drift(results)
    plot_target_styles(results)
    print("\n--- what replay is for: batch diversity ---")
    batch_correlation_diagnostic()


if __name__ == "__main__":
    main()
