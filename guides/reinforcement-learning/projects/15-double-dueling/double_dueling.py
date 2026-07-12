"""Double DQN and Dueling DQN: two small changes, ablated -- and a warning.

Two experiments, because one of them alone would mislead.

1. THE MECHANISM, on an environment built to expose it. Sutton & Barto's
   maximization-bias MDP (Example 6.7): from state A you may go RIGHT and end with
   0, or LEFT into state B, from which every one of 8 actions ends the episode with
   a reward drawn from N(-0.1, 1). B is worth -0.1, so RIGHT is optimal and LEFT is
   a mistake. But `max_a Q(B, a)` takes the largest of 8 noisy estimates, and the
   largest of 8 noisy estimates of -0.1 is reliably POSITIVE. Q-learning therefore
   convinces itself that B is a good place to go, and walks LEFT. This is
   overestimation in its purest form, and Double DQN exists to fix exactly it.

2. THE ABLATION, on project 14's pixel Pong: the 2x2 of {vanilla, double} x
   {shared head, dueling head}, plus a direct measurement of the prediction gap
       what the network PREDICTS   -- max_a Q(s, a)
       what the policy then EARNS  -- the discounted return from that same state
   at states the greedy policy actually visits (the cheap Monte-Carlo estimate van
   Hasselt et al. used on Atari).

   Read those two experiments together. MiniPong is deterministic and has three
   actions, so there is very little noise for the max to launder into optimism --
   and the measured gap says so. A reader shown only the MiniPong numbers would
   conclude Double DQN does nothing; a reader shown only the biased MDP would
   conclude it always helps. Neither is true, and the difference between the two
   environments is precisely the condition under which the fix matters.

Runtime: ~9 min on 12 CPU cores.
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-build-a-gridworld"))
sys.path.insert(0, str(HERE.parent / "13-add-a-replay-buffer"))
sys.path.insert(0, str(HERE.parent / "14-atari-pong"))

import matplotlib.pyplot as plt  # noqa: E402
from dqn_lib import Config, DQNAgent, MLPQNet, ReplayBuffer, train_dqn  # noqa: E402
from plot_style import (BASELINE, INK, INK_MUTED, INK_SECONDARY,  # noqa: E402
                        SERIES, SURFACE, finish, new_axes, style_axes)
from pong_lib import MAX_RALLIES, ConvQNet, make_minipong  # noqa: E402

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

TOTAL_STEPS = 50_000
# 4 variants x 3 seeds = 12 runs = exactly one wave on 12 cores. A 4th seed would
# add a second wave and roughly double the wall time for a marginal gain in
# precision; the 30-seed statistics that the write-up's claims rest on come from
# the maximization-bias MDP below, which is cheap enough to run properly.
SEEDS = 3
GAMMA = 0.99
BIAS_EPISODES = 8        # rollouts per bias probe

# --- the maximization-bias MDP (Sutton & Barto, Example 6.7) ---
B_ACTIONS = 8            # how many noisy actions state B offers
B_MEAN, B_STD = -0.1, 1.0
MAXBIAS_EPISODES = 800
MAXBIAS_SEEDS = 30       # tiny net, tiny env: statistics are cheap here

BASE = Config(
    total_steps=TOTAL_STEPS, gamma=GAMMA, lr=1e-3, batch_size=32,
    buffer_size=20_000, learning_starts=1_000, train_freq=2, target_freq=500,
    eps_start=1.0, eps_end=0.05, eps_decay_frac=0.3,
    eval_every=5_000, eval_episodes=6,
)

VARIANTS = {
    "DQN":                 dict(double=False, dueling=False),
    "+ Double":            dict(double=True, dueling=False),
    "+ Dueling":           dict(double=False, dueling=True),
    "+ Double + Dueling":  dict(double=True, dueling=True),
}


# --------------------------------------------------------------------------
# Experiment 1: maximization bias, isolated
# --------------------------------------------------------------------------

class MaxBiasMDP:
    """A -> (LEFT) -> B -> one of 8 actions, each ending with reward ~ N(-0.1, 1).
    A -> (RIGHT) -> done, reward 0.

    True values: Q*(A, RIGHT) = 0, and Q*(A, LEFT) = gamma * (-0.1) < 0. So LEFT is
    always the wrong move, and an unbiased learner takes it only by exploration.
    """

    n_actions = B_ACTIONS       # one action space for both states; in A, action 0 is
    obs_shape = (2,)            # LEFT and everything else is RIGHT

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.eye = np.eye(2, dtype=np.float32)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = 0                       # A
        return self.eye[0].copy(), {}

    def step(self, action):
        if self.state == 0:
            if action == 0:                  # LEFT, into the trap
                self.state = 1
                return self.eye[1].copy(), 0.0, False, False, {"went_left": True}
            return self.eye[0].copy(), 0.0, True, False, {"went_left": False}  # RIGHT
        r = float(self.rng.normal(B_MEAN, B_STD))    # any action in B: noisy, terminal
        return self.eye[0].copy(), r, True, False, {}


def run_maxbias(job):
    """Train on the biased MDP and record how often the agent walks LEFT."""
    double, seed = job
    torch.set_num_threads(1)
    cfg = replace(BASE, seed=seed, double=double, total_steps=0, lr=5e-3,
                  batch_size=32, buffer_size=10_000, learning_starts=64,
                  train_freq=1, target_freq=100, eval_every=0)

    env = MaxBiasMDP(seed=seed)
    agent = DQNAgent(lambda: MLPQNet(2, B_ACTIONS, hidden=32), B_ACTIONS, cfg)
    buffer = ReplayBuffer(cfg.buffer_size, (2,))
    rng = np.random.default_rng(seed + 3)

    left_flags, q_left = [], []
    steps = 0
    for _ in range(MAXBIAS_EPISODES):
        obs, _ = env.reset()
        obs = np.asarray(obs, dtype=np.float32)
        went_left = False
        done = False
        while not done:
            a = agent.act(obs, 0.1, rng)     # fixed epsilon: exploration is not the story
            nxt, r, term, trunc, info = env.step(a)
            nxt = np.asarray(nxt, dtype=np.float32)
            went_left = went_left or info.get("went_left", False)
            buffer.add(obs, a, r, nxt, float(term))
            obs = nxt
            done = term or trunc
            steps += 1
            if steps >= cfg.learning_starts:
                agent.update(buffer.sample(cfg.batch_size, rng))

        left_flags.append(went_left)
        with torch.no_grad():
            q_left.append(float(agent.q_values(np.eye(2, dtype=np.float32)[0][None])[0, 0]))

    return {"double": double, "seed": seed,
            "left": np.array(left_flags, dtype=np.float64),
            "q_left": np.array(q_left)}


def plot_maxbias(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=110)
    fig.patch.set_facecolor(SURFACE)
    for ax in (ax1, ax2):
        style_axes(ax)

    episodes = np.arange(1, MAXBIAS_EPISODES + 1)
    for double, color, label in ((False, SERIES[2], "DQN"), (True, SERIES[1], "Double DQN")):
        runs = [r for r in results if r["double"] == double]
        left = np.stack([r["left"] for r in runs]).mean(axis=0) * 100
        qs = np.stack([r["q_left"] for r in runs])
        ax1.plot(episodes, left, color=color, linewidth=1.8, label=label)
        ax2.plot(episodes, np.median(qs, 0), color=color, linewidth=1.8, label=label)
        ax2.fill_between(episodes, np.percentile(qs, 25, axis=0),
                         np.percentile(qs, 75, axis=0), color=color, alpha=0.15,
                         linewidth=0)

    # epsilon-greedy over 8 actions still walks LEFT 10%/8 of the time by accident
    optimal = 100 * 0.1 / B_ACTIONS
    ax1.axhline(optimal, color=INK_MUTED, linestyle="--", linewidth=1.0)
    ax1.text(MAXBIAS_EPISODES * 0.99, optimal + 2.5, "optimal (exploration only)",
             ha="right", color=INK_MUTED, fontsize=9)
    ax1.set_ylim(0, 105)
    ax1.legend(frameon=False, fontsize=9, loc="upper right")
    ax1.set_title("How often the agent walks into the trap", color=INK, fontsize=11,
                  loc="left", pad=10)
    ax1.set_xlabel("episode", color=INK_SECONDARY, fontsize=10)
    ax1.set_ylabel("episodes going LEFT (%, mean of 30 seeds)", color=INK_SECONDARY,
                   fontsize=10)

    truth = GAMMA * B_MEAN
    ax2.axhline(truth, color=INK_MUTED, linestyle="--", linewidth=1.0)
    ax2.text(MAXBIAS_EPISODES * 0.99, truth - 0.06, f"truth  Q*(A, LEFT) = {truth:.2f}",
             ha="right", color=INK_MUTED, fontsize=9)
    ax2.axhline(0.0, color=BASELINE, linewidth=0.8)
    ax2.legend(frameon=False, fontsize=9, loc="upper right")
    ax2.set_title("What it believes LEFT is worth", color=INK, fontsize=11,
                  loc="left", pad=10)
    ax2.set_xlabel("episode", color=INK_SECONDARY, fontsize=10)
    ax2.set_ylabel("Q(A, LEFT)", color=INK_SECONDARY, fontsize=10)

    fig.suptitle("Maximization bias: the max of 8 noisy estimates of −0.1 is positive",
                 color=INK, fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = OUT / "maximization_bias.png"
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


@torch.no_grad()
def measure_bias(agent, env, episodes, gamma, rng):
    """Predicted value vs realized return, on states the policy actually visits.

    Returns (mean predicted max-Q, mean realized discounted return, mean gap).
    """
    preds, actuals = [], []
    for _ in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
        states, rewards = [], []
        done = False
        while not done:
            obs = np.asarray(obs, dtype=np.float32)
            states.append(obs)
            a = agent.act(obs, 0.02, rng)
            obs, r, term, trunc, _ = env.step(a)
            rewards.append(r)
            done = term or trunc

        # discounted return from every visited state, computed backwards
        g, returns = 0.0, np.zeros(len(rewards), dtype=np.float32)
        for t in range(len(rewards) - 1, -1, -1):
            g = rewards[t] + gamma * g
            returns[t] = g

        q = agent.net(torch.as_tensor(np.stack(states))).max(dim=-1).values.numpy()
        preds.append(q)
        actuals.append(returns)

    preds = np.concatenate(preds)
    actuals = np.concatenate(actuals)
    return float(preds.mean()), float(actuals.mean()), float((preds - actuals).mean())


def run(job):
    name, seed = job
    torch.set_num_threads(1)
    flags = VARIANTS[name]
    cfg = replace(BASE, seed=seed, double=flags["double"])

    env = make_minipong(seed=seed)
    eval_env = make_minipong(seed=seed + 1000)
    bias_env = make_minipong(seed=seed + 2000)
    agent = DQNAgent(
        lambda: ConvQNet(n_actions=env.n_actions, dueling=flags["dueling"]),
        env.n_actions, cfg)
    buffer = ReplayBuffer(cfg.buffer_size, env.obs_shape, obs_dtype=np.uint8)

    # The bias probe rides along on the evaluation checkpoints: a dozen extra
    # rollouts, no gradients, no second training loop to keep in sync.
    probe_rng = np.random.default_rng(seed + 7)

    def probe(agent_, step):
        pred, real, _ = measure_bias(agent_, bias_env, BIAS_EPISODES, GAMMA, probe_rng)
        return {"bias_pred": pred, "bias_real": real}

    hist = train_dqn(env, agent, buffer, cfg, eval_env=eval_env, probe=probe)

    return {
        "name": name, "seed": seed,
        "eval_step": np.array(hist["eval_step"]),
        "eval_return": np.array(hist["eval_return"]),
        "bias_step": np.array(hist["probe_step"]),
        "bias_pred": np.array(hist["bias_pred"]),
        "bias_real": np.array(hist["bias_real"]),
    }


def group(results, name):
    return [r for r in results if r["name"] == name]


def band(ax, runs, xkey, ykey, color, label, marker="o"):
    xs = runs[0][xkey]
    ys = np.stack([r[ykey] for r in runs])
    ax.fill_between(xs, ys.min(0), ys.max(0), color=color, alpha=0.13, linewidth=0)
    ax.plot(xs, np.median(ys, 0), color=color, linewidth=1.8, marker=marker,
            markersize=3.5, label=label)


def plot_scores(results):
    fig, ax = new_axes(7.8, 4.5)
    for i, name in enumerate(VARIANTS):
        band(ax, group(results, name), "eval_step", "eval_return", SERIES[i], name)
    ax.axhline(MAX_RALLIES, color=INK_MUTED, linestyle="--", linewidth=1.0)
    ax.text(TOTAL_STEPS * 0.99, MAX_RALLIES - 0.7, "perfect play", ha="right",
            color=INK_MUTED, fontsize=9)
    ax.set_ylim(-1.6, 11)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    finish(fig, ax, "On MiniPong, all four variants land inside one another's noise",
           "environment step", "rallies returned (median of 3 seeds, min–max band)",
           OUT / "scores.png")


def plot_bias(results):
    """Left: predicted vs realized value for plain DQN. Right: the gap, per variant."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.3), dpi=110)
    fig.patch.set_facecolor(SURFACE)
    for ax in (ax1, ax2):
        style_axes(ax)

    plain = group(results, "DQN")
    xs = plain[0]["bias_step"]
    pred = np.median(np.stack([r["bias_pred"] for r in plain]), 0)
    real = np.median(np.stack([r["bias_real"] for r in plain]), 0)
    ax1.plot(xs, pred, color=SERIES[2], linewidth=1.9, marker="o", markersize=4,
             label="what DQN predicts:  max$_a$ Q(s, a)")
    ax1.plot(xs, real, color=SERIES[0], linewidth=1.9, marker="s", markersize=4,
             label="what it actually earns:  realized return")
    ax1.fill_between(xs, real, pred, color=SERIES[2], alpha=0.15, linewidth=0)
    ax1.legend(frameon=False, fontsize=9, loc="upper left")
    ax1.set_title("On MiniPong, plain DQN mostly LAGS the truth", color=INK,
                  fontsize=11, loc="left", pad=10)
    ax1.set_xlabel("environment step", color=INK_SECONDARY, fontsize=10)
    ax1.set_ylabel("value at visited states", color=INK_SECONDARY, fontsize=10)

    for i, name in enumerate(VARIANTS):
        runs = group(results, name)
        gaps = np.stack([r["bias_pred"] - r["bias_real"] for r in runs])
        ax2.plot(runs[0]["bias_step"], np.median(gaps, 0), color=SERIES[i],
                 linewidth=1.8, marker="o", markersize=3.5, label=name)
    ax2.axhline(0.0, color=INK_MUTED, linestyle="--", linewidth=1.0)
    ax2.text(TOTAL_STEPS * 0.99, 0.06, "no bias", ha="right", color=INK_MUTED, fontsize=9)
    ax2.legend(frameon=False, fontsize=9, loc="lower left")
    ax2.set_title("Prediction gap (predicted − realized)", color=INK, fontsize=11,
                  loc="left", pad=10)
    ax2.set_xlabel("environment step", color=INK_SECONDARY, fontsize=10)
    ax2.set_ylabel("gap", color=INK_SECONDARY, fontsize=10)

    # Deliberately NOT titled "Double DQN shrinks the gap". It does push every curve
    # downward, which is exactly what a pessimistic correction should do -- but on
    # this environment there was barely any overestimation to remove, so the honest
    # headline is the absence of the effect, not its presence.
    fig.suptitle("A downward correction applied to a bias that is barely there",
                 color=INK, fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = OUT / "overestimation.png"
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def summarize(results):
    """Score is reported as the mean of the last three checkpoints: one final
    10-episode evaluation is too noisy to separate variants this close together."""
    print(f"\n{'variant':<20}  {'last-3 score':>12}  {'best score':>10}  "
          f"{'final gap':>9}  {'mean gap':>8}")
    for name in VARIANTS:
        runs = group(results, name)
        last3 = np.mean([r["eval_return"][-3:].mean() for r in runs])
        best = np.mean([r["eval_return"].max() for r in runs])
        gaps = np.stack([r["bias_pred"] - r["bias_real"] for r in runs])
        print(f"{name:<20}  {last3:12.2f}  {best:10.2f}  "
              f"{np.median(gaps, 0)[-1]:9.3f}  {gaps.mean():8.3f}")


def summarize_maxbias(results):
    print(f"\n{'agent':<12}  {'walks LEFT, last 50 eps':>23}  {'final Q(A, LEFT)':>17}"
          f"  {'truth':>7}")
    for double, label in ((False, "DQN"), (True, "Double DQN")):
        runs = [r for r in results if r["double"] == double]
        left = np.mean([r["left"][-50:].mean() for r in runs]) * 100
        q = np.mean([r["q_left"][-1] for r in runs])
        print(f"{label:<12}  {left:>22.1f}%  {q:>17.3f}  {GAMMA * B_MEAN:>7.2f}")


def main():
    # 1. the mechanism, isolated
    bias_jobs = [(d, s) for d in (False, True) for s in range(MAXBIAS_SEEDS)]
    print(f"maximization-bias MDP: {len(bias_jobs)} runs...")
    with ProcessPoolExecutor(max_workers=12) as pool:
        maxbias = list(pool.map(run_maxbias, bias_jobs))
    summarize_maxbias(maxbias)
    plot_maxbias(maxbias)

    # 2. the ablation, on pixels
    jobs = [(name, seed) for name in VARIANTS for seed in range(SEEDS)]
    print(f"\n{len(jobs)} MiniPong runs x {TOTAL_STEPS:,} steps, in parallel...")
    with ProcessPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(run, jobs))

    summarize(results)
    plot_scores(results)
    plot_bias(results)


if __name__ == "__main__":
    main()
