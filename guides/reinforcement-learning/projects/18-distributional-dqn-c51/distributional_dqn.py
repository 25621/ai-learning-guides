"""C51 on an environment built to embarrass the mean.

THE CORRIDOR. From the start state, three doors. Each leads down a two-step
corridor to a payout:

    SAFE    always +1
    RISKY   +3 or -1, a fair coin
    SPREAD  one of {-1, 0, +1, +2, +3}, uniformly

Every door has an expected payout of exactly +1. A correct Q-function must
therefore assign all three EXACTLY THE SAME VALUE -- and it does. DQN, asked what
happens behind each door, gives the same answer three times, and that answer
("about 0.98") is a number that the RISKY and SPREAD doors literally never pay.

C51 is asked the same question and describes three different futures. Nothing
about the reward signal changed; only what the network was asked to predict.

Three experiments:
  1. the projection step, drawn -- what `project_distribution` does to one atom
  2. the corridor -- learned distributions vs Monte-Carlo ground truth
  3. CartPole -- does modeling the distribution actually help CONTROL, or is it
     just a prettier answer to the same question?

Runtime: ~4 min on 12 CPU cores.
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
sys.path.insert(0, str(HERE.parent / "13-add-a-replay-buffer"))

import matplotlib.pyplot as plt  # noqa: E402
from c51 import N_ATOMS, V_MAX, V_MIN, C51Agent, C51Net, project_distribution  # noqa: E402
from dqn_lib import Config, DQNAgent, MLPQNet, ReplayBuffer, train_dqn  # noqa: E402
from plot_style import (INK, INK_MUTED, INK_SECONDARY, SERIES,  # noqa: E402
                        SURFACE, finish, new_axes, style_axes)

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

GAMMA = 0.99
CORRIDOR_STEPS = 30_000
CARTPOLE_STEPS = 60_000
SEEDS = 4
DOORS = ["SAFE", "RISKY", "SPREAD"]
PAYOUTS = {
    "SAFE": (np.array([1.0]), np.array([1.0])),
    "RISKY": (np.array([-1.0, 3.0]), np.array([0.5, 0.5])),
    "SPREAD": (np.array([-1.0, 0.0, 1.0, 2.0, 3.0]), np.full(5, 0.2)),
}
CORRIDOR_LEN = 2          # steps between the door and the payout


class Corridor:
    """States: 0 = the choice; then (door, position) for position in 1..CORRIDOR_LEN.

    Observations are one-hot over 1 + 3 * CORRIDOR_LEN states. All three doors pay
    +1 in expectation, so Q*(start, ·) is the same for every action -- by
    construction, the mean cannot tell them apart.
    """

    n_actions = 3
    n_states = 1 + 3 * CORRIDOR_LEN
    obs_shape = (n_states,)

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.eye = np.eye(self.n_states, dtype=np.float32)

    def _index(self, door, pos):
        return 1 + door * CORRIDOR_LEN + (pos - 1)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.door, self.pos = None, 0
        return self.eye[0].copy(), {}

    def step(self, action):
        if self.door is None:                       # at the choice point
            self.door, self.pos = int(action), 1
            return self.eye[self._index(self.door, 1)].copy(), 0.0, False, False, {}

        self.pos += 1
        if self.pos > CORRIDOR_LEN:                 # payout, then terminate
            values, probs = PAYOUTS[DOORS[self.door]]
            r = float(self.rng.choice(values, p=probs))
            return self.eye[0].copy(), r, True, False, {}
        return self.eye[self._index(self.door, self.pos)].copy(), 0.0, False, False, {}


def true_return_distribution(door):
    """The exact distribution of the discounted return from the START state.

    Choosing a door costs one step, then CORRIDOR_LEN - 1 more steps of zeros, so a
    payout of v arriving at the end discounts to gamma^CORRIDOR_LEN * v.
    """
    values, probs = PAYOUTS[door]
    return values * (GAMMA ** CORRIDOR_LEN), probs


def monte_carlo_returns(door, n=20_000, seed=0):
    """Ground truth the honest way: play the door 20,000 times and keep the returns."""
    rng = np.random.default_rng(seed)
    values, probs = PAYOUTS[door]
    draws = rng.choice(values, size=n, p=probs)
    return draws * (GAMMA ** CORRIDOR_LEN)


# --------------------------------------------------------------------------
# Experiment 1: what the projection actually does
# --------------------------------------------------------------------------

def plot_projection():
    """One atom, shifted by the Bellman update, split across its two neighbours."""
    support = torch.linspace(V_MIN, V_MAX, N_ATOMS)
    delta_z = (V_MAX - V_MIN) / (N_ATOMS - 1)

    # a two-spike distribution: the RISKY door, as the target net might see it
    next_probs = torch.zeros(1, N_ATOMS)
    for value, p in zip(*PAYOUTS["RISKY"]):
        i = int(round((value - V_MIN) / delta_z))
        next_probs[0, i] = p

    reward = torch.tensor([0.0])
    done = torch.tensor([0.0])
    projected = project_distribution(next_probs, reward, done, support, GAMMA)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 3.9), dpi=110, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax in (ax1, ax2):
        style_axes(ax)

    z = support.numpy()
    ax1.bar(z, next_probs[0].numpy(), width=delta_z * 0.85, color=SERIES[0])
    ax1.set_title("before: the target net's distribution at s′", color=INK,
                  fontsize=11, loc="left", pad=10)
    ax1.set_xlabel("return", color=INK_SECONDARY, fontsize=10)
    ax1.set_ylabel("probability", color=INK_SECONDARY, fontsize=10)

    ax2.bar(z, projected[0].numpy(), width=delta_z * 0.85, color=SERIES[1])
    ax2.set_title(f"after: shifted by r + γ·z (γ = {GAMMA}) and re-landed on the grid",
                  color=INK, fontsize=11, loc="left", pad=10)
    ax2.set_xlabel("return", color=INK_SECONDARY, fontsize=10)

    for ax in (ax1, ax2):
        for value, _ in zip(*PAYOUTS["RISKY"]):
            ax.axvline(value, color=INK_MUTED, linestyle=":", linewidth=0.9)

    moved = float((projected[0] * support).sum())
    before = float((next_probs[0] * support).sum())
    fig.suptitle(f"The projection: γ·z lands between atoms, so its mass is split "
                 f"(mean {before:.3f} → {moved:.3f})",
                 color=INK, fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    path = OUT / "projection.png"
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
    print(f"  projection conserves mass: sum p = {float(projected.sum()):.6f}")


# --------------------------------------------------------------------------
# Experiment 2: the corridor
# --------------------------------------------------------------------------

def corridor_config(seed, total=CORRIDOR_STEPS):
    return Config(total_steps=total, gamma=GAMMA, lr=1e-3, batch_size=64,
                  buffer_size=20_000, learning_starts=500, train_freq=1,
                  target_freq=250, eps_start=1.0, eps_end=0.1, eps_decay_frac=0.3,
                  eval_every=0, seed=seed)


def run_corridor(job):
    method, seed = job
    torch.set_num_threads(1)
    cfg = corridor_config(seed)
    env = Corridor(seed=seed)
    n_states, n_actions = Corridor.n_states, Corridor.n_actions

    if method == "C51":
        agent = C51Agent(lambda: C51Net(n_states, n_actions), n_actions, cfg)
    else:
        agent = DQNAgent(lambda: MLPQNet(n_states, n_actions), n_actions, cfg)
    buffer = ReplayBuffer(cfg.buffer_size, (n_states,))
    train_dqn(env, agent, buffer, cfg)

    start = np.eye(n_states, dtype=np.float32)[0][None]
    q = agent.q_values(start).numpy()[0]
    out = {"method": method, "seed": seed, "q": q}
    if method == "C51":
        out["dist"] = agent.distribution(start).numpy()[0]      # (3 doors, 51 atoms)
    return out


def plot_corridor(results):
    """Left column: what C51 learned. Right: 20,000 Monte-Carlo rollouts."""
    c51 = [r for r in results if r["method"] == "C51"]
    dqn = [r for r in results if r["method"] == "DQN"]
    dist = np.mean([r["dist"] for r in c51], axis=0)            # average over seeds
    support = np.linspace(V_MIN, V_MAX, N_ATOMS)
    delta_z = (V_MAX - V_MIN) / (N_ATOMS - 1)

    fig, axes = plt.subplots(3, 2, figsize=(11.0, 8.2), dpi=110, sharex=True)
    fig.patch.set_facecolor(SURFACE)

    for row, door in enumerate(DOORS):
        ax_learned, ax_true = axes[row]
        for ax in (ax_learned, ax_true):
            style_axes(ax)

        ax_learned.bar(support, dist[row], width=delta_z * 0.9, color=SERIES[row])
        mean_learned = float((dist[row] * support).sum())
        ax_learned.axvline(mean_learned, color=INK, linestyle="--", linewidth=1.2)
        ax_learned.set_title(f"{door}: C51's learned distribution   (mean {mean_learned:+.2f})",
                             color=INK, fontsize=10.5, loc="left", pad=8)
        ax_learned.set_ylabel("probability", color=INK_SECONDARY, fontsize=9)

        draws = monte_carlo_returns(door, seed=row)
        ax_true.hist(draws, bins=np.linspace(V_MIN, V_MAX, N_ATOMS + 1),
                     density=False, weights=np.full(len(draws), 1 / len(draws)),
                     color=SERIES[row], alpha=0.55)
        ax_true.axvline(draws.mean(), color=INK, linestyle="--", linewidth=1.2)
        ax_true.set_title(f"{door}: 20,000 actual rollouts   (mean {draws.mean():+.2f})",
                          color=INK_SECONDARY, fontsize=10.5, loc="left", pad=8)

    for ax in axes[-1]:
        ax.set_xlabel("discounted return", color=INK_SECONDARY, fontsize=10)

    q_mean = np.mean([r["q"] for r in dqn], axis=0)
    fig.suptitle(
        "Three doors, one number: DQN says "
        f"{q_mean[0]:.2f} / {q_mean[1]:.2f} / {q_mean[2]:.2f} — C51 says what actually happens",
        color=INK, fontsize=12.5, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = OUT / "corridor_distributions.png"
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# --------------------------------------------------------------------------
# Experiment 3: does it help control?
# --------------------------------------------------------------------------

def run_cartpole(job):
    method, seed = job
    torch.set_num_threads(1)
    cfg = Config(total_steps=CARTPOLE_STEPS, gamma=0.99, lr=1e-3, batch_size=64,
                 buffer_size=50_000, learning_starts=1_000, train_freq=1,
                 target_freq=500, eps_start=1.0, eps_end=0.05, eps_decay_frac=0.25,
                 eval_every=5_000, eval_episodes=5, seed=seed)
    env, eval_env = gym.make("CartPole-v1"), gym.make("CartPole-v1")

    if method == "C51":
        # CartPole returns live in [0, 100] (gamma = 0.99, 500-step cap), so the
        # atom grid has to be moved: an atom range of [-2, 4] could not represent a
        # return of 80 at all. Choosing v_min/v_max is C51's one real chore.
        agent = C51Agent(lambda: C51Net(4, 2), 2, cfg, v_min=0.0, v_max=110.0)
    else:
        agent = DQNAgent(lambda: MLPQNet(4, 2), 2, cfg)
    buffer = ReplayBuffer(cfg.buffer_size, (4,))
    hist = train_dqn(env, agent, buffer, cfg, eval_env=eval_env)
    env.close()
    eval_env.close()
    return {"method": method, "seed": seed,
            "eval_step": np.array(hist["eval_step"]),
            "eval_return": np.array(hist["eval_return"])}


def plot_cartpole(results):
    fig, ax = new_axes(7.6, 4.4)
    for i, method in enumerate(("DQN", "C51")):
        runs = [r for r in results if r["method"] == method]
        xs = runs[0]["eval_step"]
        ys = np.stack([r["eval_return"] for r in runs])
        color = SERIES[0] if method == "DQN" else SERIES[3]
        ax.fill_between(xs, ys.min(0), ys.max(0), color=color, alpha=0.15, linewidth=0)
        ax.plot(xs, np.median(ys, 0), color=color, linewidth=1.9, marker="o",
                markersize=3.5, label=method)
    ax.axhline(475, color=INK_MUTED, linestyle="--", linewidth=1.0)
    ax.text(CARTPOLE_STEPS * 0.99, 487, "solved (475)", ha="right",
            color=INK_MUTED, fontsize=9)
    ax.set_ylim(0, 540)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    finish(fig, ax, "Distributional heads earn their keep on control too",
           "environment step", "greedy evaluation return (median of 4 seeds, min–max)",
           OUT / "cartpole_c51_vs_dqn.png")


def main():
    plot_projection()

    jobs = [(m, s) for m in ("C51", "DQN") for s in range(SEEDS)]
    print(f"\ncorridor: {len(jobs)} runs x {CORRIDOR_STEPS:,} steps...")
    with ProcessPoolExecutor(max_workers=8) as pool:
        corridor = list(pool.map(run_corridor, jobs))

    print(f"\n{'door':<8}  {'true mean':>9}  {'DQN Q(start, door)':>19}  "
          f"{'C51 mean':>9}  {'C51 P(return < 0)':>18}")
    support = np.linspace(V_MIN, V_MAX, N_ATOMS)
    c51_dist = np.mean([r["dist"] for r in corridor if r["method"] == "C51"], axis=0)
    dqn_q = np.mean([r["q"] for r in corridor if r["method"] == "DQN"], axis=0)
    for i, door in enumerate(DOORS):
        values, probs = true_return_distribution(door)
        true_mean = float((values * probs).sum())
        c51_mean = float((c51_dist[i] * support).sum())
        p_loss = float(c51_dist[i][support < 0].sum())
        print(f"{door:<8}  {true_mean:>9.3f}  {dqn_q[i]:>19.3f}  {c51_mean:>9.3f}  "
              f"{p_loss:>17.1%}")
    print("\nEvery door is worth the same. Only one of the two agents can say how.")

    plot_corridor(corridor)

    print(f"\ncartpole: {len(jobs)} runs x {CARTPOLE_STEPS:,} steps...")
    with ProcessPoolExecutor(max_workers=8) as pool:
        cartpole = list(pool.map(run_cartpole, jobs))

    print(f"\n{'method':<8}  {'last-3 eval':>11}  {'best eval':>9}  {'hit 475':>8}")
    for method in ("DQN", "C51"):
        runs = [r for r in cartpole if r["method"] == method]
        last3 = np.mean([r["eval_return"][-3:].mean() for r in runs])
        best = np.mean([r["eval_return"].max() for r in runs])
        hit = sum(r["eval_return"].max() >= 475 for r in runs)
        print(f"{method:<8}  {last3:11.1f}  {best:9.1f}  {hit:>4}/{len(runs)}")

    plot_cartpole(cartpole)


if __name__ == "__main__":
    main()
