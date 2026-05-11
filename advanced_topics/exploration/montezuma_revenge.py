"""
Work item 2 -- "Train on Montezuma's Revenge"

Montezuma's Revenge is the poster child of *hard-exploration* games.  In
the very first room you have to climb down a ladder, jump over a moving
skull, climb another ladder, and grab a key -- roughly a hundred precise
joystick actions -- before the game gives you a SINGLE point.  Until then
the reward signal is a flat zero, so a vanilla DQN (which only learns from
rewards it actually receives) wanders aimlessly forever and scores 0.  It
was *the* environment that classic deep-RL agents (DQN, A3C, ...) could
not crack, and the breakthrough -- Random Network Distillation, 2018 --
worked precisely by adding an intrinsic curiosity bonus.

Training a real pixel-level Montezuma agent needs a convolutional network,
frame stacking, an RND module, and tens of millions of environment frames
(hours on a GPU) -- well outside what a teaching script should do.  So we
do two things here:

  1. **Touch the real game** (if it is installed): load
     `ALE/MontezumaRevenge-v5` via Gymnasium, run a short random rollout,
     and watch it score exactly 0 -- the concrete face of "sparse reward".

  2. **Train on a scale model**: `MiniMontezumaEnv`, a tiny gridworld with
     the *same skeleton* -- walk to a key, pick it up, the door opens,
     walk to the treasure, reward only at the very end.  We train a
     tabular Q-learner twice: once plain (epsilon-greedy) and once with a
     prediction-error curiosity bonus (the tabular cousin of RND/ICM).
     The plain agent scores 0 forever; the curious one solves it.  Same
     lesson as the real game, 100,000x faster.

Run:
    python montezuma_revenge.py
Outputs:
    outputs/montezuma_revenge.png   -- learning curves + the learned route
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hard_exploration_envs import MiniMontezumaEnv
from curiosity_bonus import q_learning_with_curiosity


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===========================================================================
# 1. Poke the real Atari game (optional -- skipped gracefully if not installed)
# ===========================================================================
def try_real_montezuma(n_steps=2000, seed=0):
    print("-" * 72)
    print("1) The real thing: ALE/MontezumaRevenge-v5")
    try:
        import gymnasium as gym
        import ale_py  # noqa: F401  (registers the ALE environments)
        env = gym.make("ALE/MontezumaRevenge-v5")
    except Exception as exc:                       # ale-py / ROMs not installed
        print("   (skipped -- the Atari env isn't installed in this repo's venv)")
        print(f"   reason: {type(exc).__name__}: {exc}")
        print("   to try it yourself:")
        print("       pip install 'gymnasium[atari]' 'ale-py' 'autorom[accept-rom-license]'")
        print("   then re-run this script.  Expect: a random agent scores 0.0,")
        print("   because the first point is ~100 precise actions away.")
        print("-" * 72)
        return

    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    total = 0.0
    steps = 0
    for _ in range(n_steps):
        a = env.action_space.sample()             # uniform-random joystick
        obs, r, term, trunc, _ = env.step(a)
        total += r
        steps += 1
        if term or trunc:
            obs, _ = env.reset()
    env.close()
    print(f"   observation shape : {obs.shape}   (210x160 RGB pixels)")
    print(f"   action space      : {env.action_space.n} discrete joystick actions")
    print(f"   random agent over {steps} steps  ->  total game reward = {total:.1f}")
    print("   ^ almost certainly 0.0.  That flat-zero signal is exactly why a")
    print("     reward-only agent (vanilla DQN) is helpless here, and why the")
    print("     RND paper added an intrinsic *curiosity* bonus to crack it.")
    print("-" * 72)


# ===========================================================================
# 2. Train on MiniMontezuma: no curiosity  vs  prediction-error curiosity
# ===========================================================================
def greedy_rollout(Q, env, max_steps=300):
    """Run the greedy policy once; return the list of (row, col) cells visited
    and the extrinsic return."""
    s = env.reset()
    cells = [(env.r, env.c)]
    total, done, steps = 0.0, False, 0
    while not done and steps < max_steps:
        a = int(np.argmax(Q[s]))
        s, r, done, _ = env.step(a)
        cells.append((env.r, env.c))
        total += r
        steps += 1
    return cells, total


def _sliding(x, win):
    return np.array([x[max(0, i - win + 1):i + 1].mean() for i in range(len(x))])


def main():
    np.random.seed(0)
    try_real_montezuma()

    print("\n2) The scale model: MiniMontezuma (tabular)\n")
    env_fn = lambda: MiniMontezumaEnv(max_steps=300)
    n_episodes = 1500
    n_seeds = 8

    agents = {
        "no curiosity (epsilon-greedy)": dict(curiosity_kind="none", color="#e74c3c"),
        "prediction-error curiosity":    dict(curiosity_kind="prediction", color="#27ae60"),
    }

    learning_curves = {}
    sample_Q = {}
    for label, cfg in agents.items():
        returns = np.zeros((n_seeds, n_episodes))
        for seed in range(n_seeds):
            out = q_learning_with_curiosity(env_fn, curiosity_kind=cfg["curiosity_kind"],
                                            n_episodes=n_episodes, seed=seed)
            returns[seed] = out["returns"]
            if seed == 0:
                sample_Q[label] = out["Q"]
        learning_curves[label] = returns
        final = returns[:, -100:].mean()
        solved = (returns[:, -100:] > 0).mean()
        print(f"  {label:<32}  avg return (last 100 eps) = {final:5.2f}   "
              f"solve rate = {solved:5.2f}")

    # ---- figure ------------------------------------------------------------
    fig, (ax_lc, ax_route) = plt.subplots(1, 2, figsize=(14, 5.6),
                                          gridspec_kw={"width_ratios": [1.5, 1]})

    # left: learning curves
    win = 50
    for label, cfg in agents.items():
        m = _sliding(learning_curves[label].mean(0), win)
        ax_lc.plot(m, color=cfg["color"], lw=2.4, label=label)
    ax_lc.axhline(1.0, color="#7f8c8d", ls="--", lw=1, label="treasure reward = 1.0")
    ax_lc.set_xlabel("Episode")
    ax_lc.set_ylabel(f"Extrinsic return  [sliding {win} eps, {n_seeds} seeds]")
    ax_lc.set_title("MiniMontezuma: reward stays 0 forever without curiosity")
    ax_lc.set_ylim(-0.05, 1.1)
    ax_lc.grid(alpha=0.3)
    ax_lc.legend(loc="center right", fontsize=9)

    # right: the route the curious agent learned, drawn on the grid
    probe = MiniMontezumaEnv()
    rows, cols = probe.rows, probe.cols
    wall = np.array([[1 if probe.grid[r][c] == "#" else 0
                      for c in range(cols)] for r in range(rows)], dtype=float)
    ax_route.imshow(wall, cmap="Greys", alpha=0.35)          # walls as light blocks
    curious_label = "prediction-error curiosity"
    cells, ret = greedy_rollout(sample_Q[curious_label], MiniMontezumaEnv(max_steps=300))
    ys = [c[0] for c in cells]
    xs = [c[1] for c in cells]
    ax_route.plot(xs, ys, "-o", color="#27ae60", ms=4, lw=2,
                  label=f"greedy route (return {ret:.0f})")
    ax_route.scatter([probe.start[1]], [probe.start[0]], marker="s", s=110,
                     color="white", edgecolor="black", zorder=5, label="start")
    ax_route.scatter([probe.key_pos[1]], [probe.key_pos[0]], marker="P", s=170,
                     color="gold", edgecolor="black", zorder=5, label="key")
    ax_route.scatter([probe.door_pos[1]], [probe.door_pos[0]], marker="|", s=300,
                     color="orange", linewidths=4, zorder=4, label="door (needs key)")
    ax_route.scatter([probe.treasure_pos[1]], [probe.treasure_pos[0]], marker="*",
                     s=300, color="cyan", edgecolor="black", zorder=5, label="treasure")
    ax_route.set_title("The route the curious agent learned")
    ax_route.set_xticks([]); ax_route.set_yticks([])
    ax_route.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=8)

    # also print where the greedy curiosity agent ends up
    demo_env = MiniMontezumaEnv(max_steps=300)
    s = demo_env.reset()
    done, steps = False, 0
    while not done and steps < 120:
        a = int(np.argmax(sample_Q[curious_label][s]))
        s, r, done, _ = demo_env.step(a)
        steps += 1
    reached = (demo_env.r, demo_env.c) == demo_env.treasure_pos
    print("\nGreedy curiosity agent, final frame:")
    print(demo_env.render_ascii())
    print(f"  -> has_key={demo_env.has_key}  reached_treasure={reached}  "
          f"steps={demo_env.steps}")

    fig.suptitle("\"Training on Montezuma's Revenge\": the real game is too heavy "
                 "for a teaching script, so we train on a tabular scale model with "
                 "the same key->door->treasure structure.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(OUTPUT_DIR, "montezuma_revenge.png")
    fig.savefig(out, dpi=120)
    print(f"\nPlot saved to {out}")
    print("\nTakeaway: 'sparse reward' isn't a quirk -- it's the default in any "
          "world where success requires a long correct sequence.  Reward-only "
          "agents (vanilla DQN) cannot bootstrap from a signal that is zero "
          "everywhere; a curiosity bonus manufactures a dense, self-generated "
          "signal that carries the agent to the first real reward.  That's the "
          "RND idea, shrunk to fit in a Q-table.")


if __name__ == "__main__":
    main()
