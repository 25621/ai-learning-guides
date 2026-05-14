"""
Goal-conditioned Q-learning on a gridworld.

One policy is trained to reach many possible goals. The goal is treated as part
of the input, so the same learner can change behavior when the desired
destination changes.

Run:
    python goal_conditioned_policy.py
Output:
    outputs/goal_conditioned_policy.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

GRID = [
    ".......",
    ".##.##.",
    ".......",
    ".##.##.",
    ".......",
]
ROWS, COLS = len(GRID), len(GRID[0])
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
N_ACTIONS = len(ACTIONS)
N_STATES = ROWS * COLS
VALID_STATES = [r * COLS + c for r in range(ROWS) for c in range(COLS) if GRID[r][c] != "#"]


def pos(state):
    return divmod(state, COLS)


def blocked(r, c):
    return r < 0 or r >= ROWS or c < 0 or c >= COLS or GRID[r][c] == "#"


def step(state, action, goal):
    r, c = pos(state)
    dr, dc = ACTIONS[action]
    nr, nc = r + dr, c + dc
    if blocked(nr, nc):
        nr, nc = r, c
    next_state = nr * COLS + nc
    done = next_state == goal
    reward = 1.0 if done else -0.02
    return next_state, reward, done


def choose_action(q, state, goal, rng, epsilon):
    if rng.random() < epsilon:
        return int(rng.integers(N_ACTIONS))
    return int(np.argmax(q[state, goal]))


def train(episodes=5000, alpha=0.35, gamma=0.96, epsilon=0.25, seed=7):
    rng = np.random.default_rng(seed)
    q = np.zeros((N_STATES, N_STATES, N_ACTIONS))
    successes = np.zeros(episodes)
    lengths = np.zeros(episodes)

    for ep in range(episodes):
        goal = int(rng.choice(VALID_STATES))
        start_choices = [s for s in VALID_STATES if s != goal]
        state = int(rng.choice(start_choices))
        eps = max(0.04, epsilon * (1.0 - ep / episodes))

        for t in range(70):
            action = choose_action(q, state, goal, rng, eps)
            next_state, reward, done = step(state, action, goal)
            target = reward if done else reward + gamma * np.max(q[next_state, goal])
            q[state, goal, action] += alpha * (target - q[state, goal, action])
            state = next_state
            if done:
                successes[ep] = 1.0
                lengths[ep] = t + 1
                break
        if not successes[ep]:
            lengths[ep] = 70

    return q, successes, lengths


def evaluate(q, n_trials=400, seed=11):
    rng = np.random.default_rng(seed)
    solved = 0
    lengths = []
    for _ in range(n_trials):
        goal = int(rng.choice(VALID_STATES))
        start = int(rng.choice([s for s in VALID_STATES if s != goal]))
        state = start
        for t in range(70):
            action = int(np.argmax(q[state, goal]))
            state, _, done = step(state, action, goal)
            if done:
                solved += 1
                lengths.append(t + 1)
                break
        else:
            lengths.append(70)
    return solved / n_trials, float(np.mean(lengths))


def rollout(q, start, goal):
    state = start
    path = [state]
    for _ in range(70):
        action = int(np.argmax(q[state, goal]))
        state, _, done = step(state, action, goal)
        path.append(state)
        if done:
            break
    return path


def moving_average(values, window=100):
    return np.array([values[max(0, i - window + 1):i + 1].mean() for i in range(len(values))])


def draw_grid(ax, path, start, goal, title):
    image = np.zeros((ROWS, COLS))
    for state in path:
        r, c = pos(state)
        image[r, c] += 1
    ax.imshow(image, cmap="YlGnBu")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    for r in range(ROWS):
        for c in range(COLS):
            if GRID[r][c] == "#":
                ax.text(c, r, "#", ha="center", va="center", fontsize=15, weight="bold")
    coords = [pos(s) for s in path]
    ax.plot([c for _, c in coords], [r for r, _ in coords], color="#e6550d", lw=3)
    sr, sc = pos(start)
    gr, gc = pos(goal)
    ax.scatter([sc], [sr], marker="o", s=90, color="#31a354")
    ax.scatter([gc], [gr], marker="*", s=180, color="#de2d26")


def plot(q, successes, lengths):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    axes[0].plot(moving_average(successes), color="#2ca25f", lw=2.4, label="success rate")
    axes[0].plot(moving_average(lengths / 70.0), color="#756bb1", lw=2.0, label="normalized length")
    axes[0].set_title("One learner adapts to many goals")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Moving average")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    start = 4 * COLS
    goal_a = 0 * COLS + 6
    goal_b = 4 * COLS + 6
    draw_grid(axes[1], rollout(q, start, goal_a), start, goal_a, "Same start, goal at top right")
    draw_grid(axes[2], rollout(q, start, goal_b), start, goal_b, "Same start, goal at bottom right")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "goal_conditioned_policy.png")
    plt.savefig(out, dpi=130)
    return out


def main():
    print("=== Goal-conditioned policy: many goals, one learner ===")
    q, successes, lengths = train()
    success_rate, mean_length = evaluate(q)
    out = plot(q, successes, lengths)
    print(f"Final training success rate over last 500 episodes: {successes[-500:].mean():.2f}")
    print(f"Greedy evaluation success rate: {success_rate:.2f}")
    print(f"Greedy evaluation average path length: {mean_length:.1f} steps")
    print(f"Plot saved to {out}")


if __name__ == "__main__":
    main()
