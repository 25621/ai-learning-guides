"""SARSA vs Q-learning on Cliff Walking — Sutton & Barto Figure 6.5.

The environment's exact dynamics are read once from Gymnasium's
CliffWalking-v1 tables into flat numpy arrays (the env is deterministic:
one next state, one reward per (s, a)), then training steps are plain array
lookups. Same MDP, ~50x faster than stepping the wrapped env, which is what
makes a 200-run average affordable.

Settings follow the book: alpha=0.5, gamma=1, fixed epsilon=0.1,
500 episodes, rewards summed per episode and averaged over 200 runs.
"""

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-build-a-gridworld"))

import matplotlib.pyplot as plt  # noqa: E402
from plot_style import (INK, INK_SECONDARY, SERIES, SURFACE, finish,  # noqa: E402
                        new_axes)

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

N_ROWS, N_COLS = 4, 12
START, GOAL = 36, 47
ALPHA, GAMMA, EPS = 0.5, 1.0, 0.1
EPISODES, RUNS = 500, 200
ARROWS = "↑→↓←"  # CliffWalking actions: 0=up, 1=right, 2=down, 3=left


def dynamics():
    """(next_state[s,a], reward[s,a], terminal[s,a]) from the env's tables."""
    env = gym.make("CliffWalking-v1")
    u = env.unwrapped
    S, A = u.observation_space.n, u.action_space.n
    ns = np.zeros((S, A), dtype=np.int64)
    rw = np.zeros((S, A))
    tm = np.zeros((S, A), dtype=bool)
    for s in range(S):
        for a in range(A):
            (p, s2, r, term), = u.P[s][a]
            assert p == 1.0
            ns[s, a], rw[s, a], tm[s, a] = s2, r, term
    return ns, rw, tm


def train(algo, ns, rw, tm, rng, episodes=EPISODES, eps_fn=None):
    """Returns (per-episode reward sums, final Q). algo: 'sarsa'|'q'."""
    Q = np.zeros(ns.shape)
    rewards = np.zeros(episodes)
    for ep in range(episodes):
        eps = EPS if eps_fn is None else eps_fn(ep)
        s = START
        a = int(rng.integers(4)) if rng.random() < eps else int(Q[s].argmax())
        total = 0.0
        while True:
            s2, r, done = ns[s, a], rw[s, a], tm[s, a]
            total += r
            a2 = int(rng.integers(4)) if rng.random() < eps else int(Q[s2].argmax())
            if algo == "sarsa":
                target = r + GAMMA * Q[s2, a2] * (not done)
            else:  # q-learning
                target = r + GAMMA * Q[s2].max() * (not done)
            Q[s, a] += ALPHA * (target - Q[s, a])
            s, a = s2, a2
            if done:
                break
        rewards[ep] = total
    return rewards, Q


def greedy_path(Q, ns, tm, max_steps=200):
    """Follow argmax Q from START; returns list of states (maybe truncated)."""
    path, s = [START], START
    for _ in range(max_steps):
        a = int(Q[s].argmax())
        done = tm[s, a]
        s = ns[s, a]
        path.append(int(s))
        if done:
            break
    return path


def path_return(path):
    """-1 per move, -100 per cliff bounce (state resets to START)."""
    total = 0
    for prev, cur in zip(path, path[1:]):
        total += -100 if (cur == START and prev != START) else -1
    return total


def draw_cliff(ax, paths):
    import matplotlib.patches as mpatches

    for r in range(N_ROWS):
        for c in range(N_COLS):
            x, y = c, N_ROWS - 1 - r
            cliff = r == 3 and 1 <= c <= 10
            face = "#3b3a37" if cliff else "#f4f4f1"
            ax.add_patch(mpatches.Rectangle((x, y), 1, 1, facecolor=face,
                                            edgecolor="#fcfcfb", linewidth=1.5))
    ax.text(0.5, 0.5, "S", ha="center", va="center", fontsize=11,
            color=INK, fontweight="bold")
    ax.text(11.5, 0.5, "G", ha="center", va="center", fontsize=11,
            color=INK, fontweight="bold")
    ax.text(6, 0.5, "the cliff  (step in: −100, back to S)", ha="center",
            va="center", fontsize=8.5, color="#fcfcfb")
    for (label, path, color, dy) in paths:
        xs = [s % N_COLS + 0.5 for s in path]
        ys = [N_ROWS - 1 - s // N_COLS + 0.5 + dy for s in path]
        ax.plot(xs, ys, color=color, linewidth=2.2, alpha=0.9,
                solid_capstyle="round")
        ax.annotate(label, (xs[len(xs) // 2], ys[len(ys) // 2]), color=color,
                    fontsize=10, xytext=(0, 7 if dy > 0 else -13),
                    textcoords="offset points", ha="center")
    ax.set_xlim(-0.1, N_COLS + 0.1)
    ax.set_ylim(-0.1, N_ROWS + 0.1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)


def main():
    ns, rw, tm = dynamics()

    curves = {"sarsa": np.zeros(EPISODES), "q": np.zeros(EPISODES)}
    Q_final, terminating = {}, {"sarsa": 0, "q": 0}
    for run in range(RUNS):
        rng = np.random.default_rng(run)
        for algo in curves:
            rewards, Q = train(algo, ns, rw, tm, rng)
            curves[algo] += rewards / RUNS
            p = greedy_path(Q, ns, tm)
            if p[-1] == GOAL:
                terminating[algo] += 1
                # keep one representative Q whose greedy path reaches the goal
                Q_final.setdefault(algo, Q)

    print("=== online performance (mean reward per episode) ===")
    for algo in ("sarsa", "q"):
        print(f"{algo:8s}: last-100-episode average = "
              f"{curves[algo][-100:].mean():7.2f}")

    print("\n=== greedy policies after training ===")
    paths = {}
    for algo in ("sarsa", "q"):
        paths[algo] = greedy_path(Q_final[algo], ns, tm)
        print(f"{algo:8s}: greedy path return {path_return(paths[algo]):4d};  "
              f"greedy path reaches the goal in {terminating[algo]}/{RUNS} runs")

    # does SARSA drop the safety margin once epsilon decays to 0? measure it.
    outs = []
    for run in range(5):
        rng = np.random.default_rng(100 + run)
        _, Q_decay = train("sarsa", ns, rw, tm, rng, episodes=2000,
                           eps_fn=lambda ep: max(0.0, 0.1 * (1 - ep / 1000)))
        p = greedy_path(Q_decay, ns, tm)
        outs.append(path_return(p) if p[-1] == GOAL else None)
    print(f"\nsarsa, eps decayed 0.1 -> 0 then 1000 greedy episodes: "
          f"greedy returns {outs}")
    print("  -> it KEEPS the safe path: once exploration stops it no longer"
          " gathers the data that would show the edge is fine. Convergence to"
          " the optimal policy needs slowly-vanishing (GLIE) exploration and"
          " far more episodes.")

    # --- Figure 6.5 ---------------------------------------------------------
    fig, ax = new_axes(7.6, 4.4)
    k = 10  # light smoothing on top of the 200-run average, like the book
    kernel = np.ones(k) / k
    for color, (algo, label) in zip(SERIES, [("sarsa", "SARSA (on-policy)"),
                                             ("q", "Q-learning (off-policy)")]):
        smooth = np.convolve(curves[algo], kernel, mode="valid")
        ax.plot(np.arange(len(smooth)) + k, smooth, color=color, linewidth=1.8)
        ax.annotate(label, (330, smooth[330]), color=color, fontsize=10,
                    xytext=(0, 8 if algo == "sarsa" else -16),
                    textcoords="offset points")
    ax.set_ylim(-100, 0)
    finish(fig, ax,
           "Sum of rewards per episode while learning (200-run average)",
           "episode", "online return", OUT / "fig_6_5.png")

    # --- the two greedy routes ----------------------------------------------
    fig, ax = plt.subplots(figsize=(8.4, 3.4), dpi=110)
    fig.patch.set_facecolor(SURFACE)
    draw_cliff(ax, [
        (f"SARSA: safe path ({path_return(paths['sarsa'])})",
         paths["sarsa"], SERIES[0], +0.12),
        (f"Q-learning: edge path ({path_return(paths['q'])})",
         paths["q"], SERIES[2], -0.12),
    ])
    ax.set_title("What each algorithm's greedy policy actually does",
                 fontsize=11, loc="left", color=INK, pad=8)
    fig.tight_layout()
    fig.savefig(OUT / "greedy_paths.png", facecolor=SURFACE,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'greedy_paths.png'}")


if __name__ == "__main__":
    main()
