"""Solve FrozenLake-v1 (slippery, 4x4 and 8x8) with value iteration.

Because the ice pays +1 only for reaching the goal and the episode ends at
holes and the goal, with gamma = 1 the optimal value V*(s) IS the probability
of eventually reaching the goal from s under the optimal policy — so the
whole heatmap can be read as "chance of making it home from here", and the
number at the start cell can be checked against rollouts of the greedy
policy. The main figure uses that gamma; a gamma sweep shows how discounting
reshapes the policy.
"""

import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-build-a-gridworld"))
sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt  # noqa: E402
from frozenlake_lib import (ARROWS, draw_lake, greedy_policy, mdp_arrays,  # noqa: E402
                            rollout_stats, value_iteration)
from plot_style import SURFACE, finish, new_axes, style_axes  # noqa: E402

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)


def solve(map_name, gamma=1.0):
    # A generous step limit so truncation doesn't distort the success rate
    # (the default limit of 100/200 steps clips a few slow successes).
    env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=True,
                   max_episode_steps=2000)
    P, R = mdp_arrays(env)
    t0 = time.perf_counter()
    V, history = value_iteration(P, R, gamma)
    dt = time.perf_counter() - t0
    pi = greedy_policy(P, R, gamma, V)
    return env, P, R, V, pi, history, dt


def main():
    results = {}
    for map_name in ["4x4", "8x8"]:
        env, P, R, V, pi, history, dt = solve(map_name)
        rate, steps = rollout_stats(env, pi, episodes=20_000)
        print(f"\n=== FrozenLake {map_name} (slippery, gamma=1) ===")
        print(f"value iteration: {len(history)} sweeps, {dt*1e3:.1f} ms")
        print(f"V*(start) = {V[0]:.4f}   (predicted success probability)")
        print(f"measured greedy success over 20k episodes = {rate:.4f}"
              f"   (mean steps when successful: {steps:.0f})")
        results[map_name] = (env, V, pi, history, rate)

    # --- Figure 1: V* heatmap + greedy policy for both maps -----------------
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.4), dpi=110,
                             gridspec_kw={"width_ratios": [1, 2]})
    fig.patch.set_facecolor(SURFACE)
    for ax, map_name in zip(axes, ["4x4", "8x8"]):
        env, V, pi, history, rate = results[map_name]
        draw_lake(ax, env.unwrapped.desc, V, pi,
                  title=f"{map_name}: V* and greedy policy "
                        f"(success {rate:.0%})")
    fig.tight_layout()
    fig.savefig(OUT / "frozenlake_solution.png", facecolor=SURFACE,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'frozenlake_solution.png'}")

    # --- Figure 2: convergence of the backup ------------------------------
    fig, ax = new_axes(7.2, 4.2)
    from plot_style import SERIES
    for color, map_name in zip(SERIES, ["4x4", "8x8"]):
        history = results[map_name][3]
        ax.plot(history, color=color, linewidth=1.8)
        ax.annotate(map_name, (len(history) * 0.55, history[int(len(history) * 0.55)]),
                    color=color, fontsize=10, xytext=(4, 6),
                    textcoords="offset points")
    ax.set_yscale("log")
    finish(fig, ax, "Value iteration contracts geometrically (gamma = 1, episodic)",
           "sweep", "max value change ‖V(k+1) − V(k)‖∞", OUT / "convergence.png")

    # --- The slip in action: policy vs gamma on 4x4 ------------------------
    print("\n=== how the slip and gamma shape the 4x4 policy ===")
    env, P, R, V1, pi1, *_ = *solve("4x4", gamma=1.0), None
    header = "state (row,col) | " + " | ".join(f"g={g}" for g in (1.0, 0.9, 0.5))
    rows = {}
    for gamma in (1.0, 0.9, 0.5):
        _, _, _, V, pi, _, _ = solve("4x4", gamma)
        rows[gamma] = pi
    desc = env.unwrapped.desc
    print(header)
    for s in range(16):
        r, c = divmod(s, 4)
        if desc[r, c] in (b"H", b"G"):
            continue
        print(f"  ({r},{c})          |  " +
              "  |  ".join(ARROWS[int(rows[g][s])] for g in (1.0, 0.9, 0.5)))

    # sanity: aiming at the goal is not what the arrows show next to holes
    _, _, _, V, pi, _, _ = solve("4x4", 1.0)
    print("\ncell (1,0) sits beside a hole at (1,1); greedy action:",
          ARROWS[int(pi[4])], "(points AWAY from the hole)")


if __name__ == "__main__":
    main()
