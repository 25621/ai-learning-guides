"""Tabular Q-learning on slippery FrozenLake 4x4.

The twist that makes the learning curve honest: project 06 already extracted
the exact dynamics (P, R), so after every chunk of training the current
greedy policy is graded EXACTLY — its true success probability comes from a
linear solve at gamma=1, not from noisy evaluation rollouts. The learner
never sees P and R; only the report card uses them.

Also runs an exploration-schedule ablation: a decayed epsilon, two fixed
epsilons, and the greedy trap epsilon=0.
"""

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-build-a-gridworld"))
sys.path.insert(0, str(HERE.parent / "06-value-iteration-on-frozenlake"))

import matplotlib.pyplot as plt  # noqa: E402
from frozenlake_lib import (draw_lake, greedy_policy, mdp_arrays,  # noqa: E402
                            policy_evaluation_solve, rollout_stats,
                            value_iteration)
from plot_style import SERIES, SURFACE, finish, new_axes  # noqa: E402

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

GAMMA = 0.99
ALPHA = 0.1
EPISODES = 20_000
EVAL_EVERY = 250
SEEDS = 5


def epsilon_schedules():
    return {
        "decayed 1.0 -> 0.01": lambda ep: max(0.01, 1.0 - ep / 10_000),
        "fixed 0.1": lambda ep: 0.1,
        "fixed 0.01": lambda ep: 0.01,
        "greedy (eps = 0)": lambda ep: 0.0,
    }


def success_prob(P, R, policy):
    """Exact success probability of a deterministic policy (gamma=1 solve)."""
    return policy_evaluation_solve(P, R, 1.0, policy)[0]


def train(env, P, R, eps_fn, seed):
    """One Q-learning run; returns (checkpoint success probs, final Q)."""
    rng = np.random.default_rng(seed)
    S, A = P.shape[0], P.shape[1]
    Q = np.zeros((S, A))
    curve = []
    for ep in range(EPISODES):
        s, _ = env.reset(seed=int(rng.integers(1 << 30)))
        eps = eps_fn(ep)
        done = False
        while not done:
            a = int(rng.integers(A)) if rng.random() < eps else int(Q[s].argmax())
            s2, r, term, trunc, _ = env.step(a)
            done = term or trunc
            target = r + GAMMA * Q[s2].max() * (not term)
            Q[s, a] += ALPHA * (target - Q[s, a])
            s = s2
        if (ep + 1) % EVAL_EVERY == 0:
            curve.append(success_prob(P, R, Q.argmax(axis=1)))
    return np.array(curve), Q


def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    P, R = mdp_arrays(env)

    # the ceiling: what planning with full knowledge of P, R achieves
    V_star, _ = value_iteration(P, R, 1.0)
    ceiling = V_star[0]
    print(f"optimal success probability (value iteration, gamma=1): {ceiling:.4f}")

    curves = {}
    final_Q = None
    for name, eps_fn in epsilon_schedules().items():
        runs = []
        for seed in range(SEEDS):
            curve, Q = train(env, P, R, eps_fn, seed=1000 + seed)
            runs.append(curve)
            if name.startswith("decayed") and seed == 0:
                final_Q = Q
        curves[name] = np.mean(runs, axis=0)
        finals = ", ".join(f"{r[-1]:.2f}" for r in runs)
        print(f"{name:22s}: final success prob per seed = [{finals}]  "
              f"mean {curves[name][-1]:.4f}")

    # confirm the exact grade with actual rollouts for the decayed run
    pi_learned = final_Q.argmax(axis=1)
    exact = success_prob(P, R, pi_learned)
    eval_env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True,
                        max_episode_steps=2000)
    measured, _ = rollout_stats(eval_env, pi_learned, episodes=20_000)
    print(f"\ndecayed-eps seed-0 policy: exact success {exact:.4f}, "
          f"measured over 20k rollouts {measured:.4f}")

    pi_star = greedy_policy(P, R, GAMMA, value_iteration(P, R, GAMMA)[0])
    agree = (pi_learned == pi_star).sum()
    print(f"learned greedy policy matches the VI policy on {agree}/16 states")
    print("(disagreements sit on states where several actions are "
          "near-optimal or that the optimal route never visits)")

    # --- learning curves -----------------------------------------------------
    fig, ax = new_axes(7.6, 4.4)
    x = np.arange(EVAL_EVERY, EPISODES + 1, EVAL_EVERY)
    ax.axhline(ceiling, color="#c3c2b7", linewidth=1.4, linestyle="--")
    ax.annotate(f"planning ceiling {ceiling:.2f}", (EPISODES * 0.72, ceiling),
                color="#898781", fontsize=9, xytext=(0, 5),
                textcoords="offset points")
    label_at = {"decayed 1.0 -> 0.01": (0.55, 8), "fixed 0.1": (0.35, -14),
                "fixed 0.01": (0.75, -13), "greedy (eps = 0)": (0.3, 6)}
    for color, (name, curve) in zip(SERIES, curves.items()):
        ax.plot(x, curve, color=color, linewidth=1.8)
        fx, dy = label_at[name]
        k = int(len(curve) * fx)
        ax.annotate(name, (x[k], curve[k]), color=color, fontsize=9,
                    xytext=(4, dy), textcoords="offset points")
    ax.set_ylim(-0.03, 0.95)
    finish(fig, ax,
           "True success probability of the greedy policy during learning",
           "training episodes",
           "P(reach goal) - exact, via linear solve",
           OUT / "learning_curves.png")

    # --- learned vs planned solution ----------------------------------------
    V_learned = policy_evaluation_solve(P, R, 1.0, pi_learned)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.8), dpi=110)
    fig.patch.set_facecolor(SURFACE)
    draw_lake(axes[0], env.unwrapped.desc, V_learned, pi_learned,
              title=f"Q-learning after 20k episodes (success {exact:.0%})")
    draw_lake(axes[1], env.unwrapped.desc, V_star,
              greedy_policy(P, R, 1.0, V_star),
              title=f"value iteration with known P, R ({ceiling:.0%})")
    fig.tight_layout()
    fig.savefig(OUT / "learned_vs_planned.png", facecolor=SURFACE,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'learned_vs_planned.png'}")


if __name__ == "__main__":
    main()
