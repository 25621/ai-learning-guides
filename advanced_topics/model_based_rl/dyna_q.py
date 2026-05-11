"""
Dyna-Q on a Deterministic GridWorld (Sutton & Barto, Chapter 8)

Dyna-Q = direct RL (Q-learning) + a learned model + planning from imagined experience.

After each real step (s, a, r, s'):
    1. Direct update:  Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') - Q(s,a) ]
    2. Model learning: Model[(s, a)] = (r, s')                  (deterministic env)
    3. Planning:       Repeat n times:
                         sample (s, a) seen before
                         (r, s') = Model[(s, a)]
                         apply the same Q-update from the imagined sample

Increasing the planning steps n dramatically reduces the real environment
interactions needed to find a good policy.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# A tiny deterministic maze (the classic "Dyna maze" from Sutton & Barto).
# `#` = wall, `S` = start, `G` = goal. Each step gives reward 0; reaching G
# gives reward +1 and ends the episode.
# ----------------------------------------------------------------------
MAZE = [
    "         G",
    "   #     #",
    "   #     #",
    "   #     #",
    "         #",
    "      #   ",
    "      #   ",
    "S     #   ",
]
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # up, down, left, right
ACTION_NAMES = ["↑", "↓", "←", "→"]


class GridWorld:
    def __init__(self, maze):
        self.grid = [list(row) for row in maze]
        self.H = len(self.grid)
        self.W = len(self.grid[0])
        self.walls = {(r, c) for r in range(self.H) for c in range(self.W)
                      if self.grid[r][c] == "#"}
        self.start = next((r, c) for r in range(self.H) for c in range(self.W)
                          if self.grid[r][c] == "S")
        self.goal = next((r, c) for r in range(self.H) for c in range(self.W)
                         if self.grid[r][c] == "G")
        self.n_states = self.H * self.W
        self.n_actions = 4

    def s_to_idx(self, s):
        return s[0] * self.W + s[1]

    def reset(self):
        self.pos = self.start
        return self.s_to_idx(self.pos)

    def step(self, a):
        dr, dc = ACTIONS[a]
        nr, nc = self.pos[0] + dr, self.pos[1] + dc
        # Stay in place if blocked
        if 0 <= nr < self.H and 0 <= nc < self.W and (nr, nc) not in self.walls:
            self.pos = (nr, nc)
        done = self.pos == self.goal
        reward = 1.0 if done else 0.0
        return self.s_to_idx(self.pos), reward, done


# ----------------------------------------------------------------------
# Dyna-Q agent
# ----------------------------------------------------------------------
def dyna_q(env, n_episodes=50, n_planning=0, alpha=0.1, gamma=0.95,
           epsilon=0.1, seed=SEED):
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions))
    model = {}                      # (s, a) -> (r, s_next)
    steps_per_episode = []

    for ep in range(n_episodes):
        s = env.reset()
        steps = 0
        done = False
        while not done:
            # ε-greedy action
            if rng.random() < epsilon:
                a = rng.integers(env.n_actions)
            else:
                a = int(np.argmax(Q[s]))

            s_next, r, done = env.step(a)
            steps += 1

            # 1) Direct RL update
            td_target = r + gamma * np.max(Q[s_next]) * (0.0 if done else 1.0)
            Q[s, a] += alpha * (td_target - Q[s, a])

            # 2) Model learning (deterministic, so just remember last outcome)
            model[(s, a)] = (r, s_next, done)

            # 3) Planning: replay n imagined transitions from the model
            if model and n_planning > 0:
                keys = list(model.keys())
                for _ in range(n_planning):
                    sp, ap = keys[rng.integers(len(keys))]
                    rp, snp, dp = model[(sp, ap)]
                    tp = rp + gamma * np.max(Q[snp]) * (0.0 if dp else 1.0)
                    Q[sp, ap] += alpha * (tp - Q[sp, ap])

            s = s_next

        steps_per_episode.append(steps)

    return Q, steps_per_episode


def greedy_path(env, Q, max_steps=200):
    """Walk the greedy policy and return the path of (row, col) cells."""
    path = []
    s = env.reset()
    path.append(env.pos)
    for _ in range(max_steps):
        a = int(np.argmax(Q[s]))
        s, _, done = env.step(a)
        path.append(env.pos)
        if done:
            break
    return path


# ----------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------
def plot_results(results, env):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Steps-per-episode curves
    colors = ["#e74c3c", "#f1c40f", "#2ecc71"]
    for (n, steps), c in zip(results.items(), colors):
        axes[0].plot(range(1, len(steps) + 1), steps, color=c,
                     linewidth=2, label=f"n = {n} planning steps")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Steps to reach goal")
    axes[0].set_title("Dyna-Q: planning makes real experience go further")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Greedy path of the best (most planning) agent
    Q_best = results_Q[max(results_Q.keys())]
    path = greedy_path(env, Q_best)

    grid = np.zeros((env.H, env.W))
    for (r, c) in env.walls:
        grid[r, c] = -1
    grid[env.goal] = 2
    grid[env.start] = 1

    axes[1].imshow(grid, cmap="Pastel2")
    # Draw path
    xs = [p[1] for p in path]
    ys = [p[0] for p in path]
    axes[1].plot(xs, ys, color="#2c3e50", linewidth=2, marker="o", markersize=4)
    axes[1].plot(env.start[1], env.start[0], "g*", markersize=18, label="Start")
    axes[1].plot(env.goal[1], env.goal[0], "r*", markersize=18, label="Goal")
    axes[1].set_xticks(range(env.W))
    axes[1].set_yticks(range(env.H))
    axes[1].set_title(f"Greedy path found by Dyna-Q (n={max(results_Q.keys())})")
    axes[1].legend(loc="lower right")
    axes[1].grid(False)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "dyna_q.png")
    plt.savefig(out, dpi=120)
    print(f"Plot saved to {out}")


def main():
    print("=== Dyna-Q on a deterministic maze ===\n")
    env = GridWorld(MAZE)
    print(f"Maze: {env.H} x {env.W}, start={env.start}, goal={env.goal}, "
          f"walls={len(env.walls)}")

    global results_Q
    results_Q = {}
    results_steps = {}
    for n in [0, 5, 50]:
        Q, steps = dyna_q(env, n_episodes=50, n_planning=n)
        results_Q[n] = Q
        results_steps[n] = steps
        avg_last10 = np.mean(steps[-10:])
        print(f"n = {n:2d} planning steps | "
              f"avg steps (last 10 episodes): {avg_last10:6.1f} | "
              f"episode 1 steps: {steps[0]:4d}")

    plot_results(results_steps, env)

    print("\nWhat to notice:")
    print("  n=0  → pure Q-learning, needs many episodes to converge")
    print("  n=5  → 5 imagined replays per real step → far fewer real steps")
    print("  n=50 → 50 replays per real step → almost optimal after 2-3 episodes")
    print("\nMore planning = more value gets propagated per real step taken.")


if __name__ == "__main__":
    main()
