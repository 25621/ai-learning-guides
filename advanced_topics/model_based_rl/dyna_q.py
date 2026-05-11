"""
Dyna-Q on the Sutton & Barto Dyna Maze (Figure 8.2, 6x9 gridworld).

Dyna-Q combines three ingredients in one loop:
  1. Direct RL          — Q-learning update from a real environment step
  2. Model learning     — record (s, a) -> (r, s') from every real transition
  3. Planning           — do n extra Q-learning updates using simulated experience
                          sampled from the learned model

Pseudocode (Sutton & Barto Chapter 8):
    Loop forever:
        s = current state
        a = epsilon-greedy(Q, s)
        Take action a, observe r, s'
        Q(s,a) += alpha * (r + gamma * max_a' Q(s', a') - Q(s,a))    # direct RL
        Model[s,a] = (r, s')                                          # model learning
        Repeat n times:                                               # planning
            (s_p, a_p) <- random previously-seen (s, a)
            (r_p, s'_p) <- Model[s_p, a_p]
            Q(s_p, a_p) += alpha * (r_p + gamma * max_a' Q(s'_p, a') - Q(s_p, a_p))

For a deterministic environment, n planning steps per real step makes Dyna-Q
learn roughly n+1 times faster than plain Q-learning.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- The Maze ---------------------------------------------------------------
# Sutton & Barto Figure 8.2-style maze.  S = start, G = goal, # = wall.
MAZE = [
    list(".......G."),
    list(".....#..."),
    list("S....#..."),
    list("...#.#..."),
    list("...#....."),
    list("........."),
]
ROWS, COLS = len(MAZE), len(MAZE[0])
START = (2, 0)
GOAL = (0, 7)
# (Up, Down, Left, Right)
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
N_ACTIONS = 4


def step_env(state, action):
    """Deterministic transition: bumping into walls/edges keeps you in place."""
    dr, dc = ACTIONS[action]
    nr, nc = state[0] + dr, state[1] + dc
    if nr < 0 or nr >= ROWS or nc < 0 or nc >= COLS or MAZE[nr][nc] == "#":
        nr, nc = state  # blocked
    next_state = (nr, nc)
    if next_state == GOAL:
        return next_state, 1.0, True
    return next_state, 0.0, False


def s_idx(state):
    return state[0] * COLS + state[1]


# --- Dyna-Q -----------------------------------------------------------------
def dyna_q(n_planning, n_episodes=50, alpha=0.1, gamma=0.95,
           epsilon=0.1, seed=0, step_cap=10_000):
    rng = np.random.default_rng(seed)
    Q = np.zeros((ROWS * COLS, N_ACTIONS))
    model = {}  # (state_idx, action) -> (reward, next_state_idx)
    steps_per_episode = []

    for _ in range(n_episodes):
        s = START
        steps = 0
        done = False
        while not done and steps < step_cap:
            si = s_idx(s)
            # epsilon-greedy action selection
            if rng.random() < epsilon:
                a = int(rng.integers(N_ACTIONS))
            else:
                a = int(np.argmax(Q[si]))

            ns, r, done = step_env(s, a)
            nsi = s_idx(ns)

            # (1) direct RL  -- Q-learning update from real experience
            Q[si, a] += alpha * (r + gamma * np.max(Q[nsi]) - Q[si, a])

            # (2) model learning (env is deterministic, so just overwrite)
            model[(si, a)] = (r, nsi)

            # (3) planning: n simulated Q-learning updates from the model
            if n_planning > 0 and len(model) > 0:
                keys = list(model.keys())
                for _ in range(n_planning):
                    psi, pa = keys[rng.integers(len(keys))]
                    pr, pnsi = model[(psi, pa)]
                    Q[psi, pa] += alpha * (pr + gamma * np.max(Q[pnsi]) - Q[psi, pa])

            s = ns
            steps += 1
        steps_per_episode.append(steps)

    return Q, steps_per_episode


# --- Visualisation ----------------------------------------------------------
def print_policy(Q, title=""):
    chars = ["↑", "↓", "←", "→"]
    print(f"\nLearned greedy policy {title}:")
    for r in range(ROWS):
        row = ""
        for c in range(COLS):
            if MAZE[r][c] == "#":
                row += " █ "
            elif (r, c) == GOAL:
                row += " G "
            elif (r, c) == START:
                row += " S "
            else:
                a = int(np.argmax(Q[r * COLS + c]))
                row += f" {chars[a]} "
        print(row)


def main():
    print("=== Dyna-Q on the 6x9 Dyna Maze ===\n")
    n_episodes = 50
    n_runs = 30  # average over seeds to smooth out exploration noise
    planning_settings = [0, 5, 50]

    results = {n: np.zeros(n_episodes) for n in planning_settings}

    for n in planning_settings:
        label = "plain Q-learning" if n == 0 else f"Dyna-Q with n={n} planning steps"
        print(f"Training {label}  (averaged over {n_runs} seeds)...")
        for run in range(n_runs):
            _, steps = dyna_q(n_planning=n, n_episodes=n_episodes, seed=run)
            results[n] += np.asarray(steps, dtype=float)
        results[n] /= n_runs

    print("\nAverage steps per episode in the last 10 episodes:")
    for n in planning_settings:
        print(f"  n={n:<3}  ->  {np.mean(results[n][-10:]):6.1f} steps")

    # Show greedy policy from a single n=50 run
    Q_demo, _ = dyna_q(n_planning=50, n_episodes=50, seed=0)
    print_policy(Q_demo, "(n=50 planning steps)")

    plt.figure(figsize=(11, 6))
    colors = {0: "#e74c3c", 5: "#f39c12", 50: "#27ae60"}
    labels = {
        0: "n = 0   (plain Q-learning)",
        5: "n = 5   planning updates / real step",
        50: "n = 50  planning updates / real step",
    }
    for n in planning_settings:
        plt.plot(range(1, n_episodes + 1), results[n],
                 color=colors[n], linewidth=2.2, label=labels[n])
    plt.xlabel("Episode")
    plt.ylabel(f"Steps per episode (average over {n_runs} runs)")
    plt.title("Dyna-Q: more planning -> fewer real-world steps to reach the goal")
    plt.yscale("log")
    plt.grid(alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "dyna_q.png")
    plt.savefig(out, dpi=120)
    print(f"\nPlot saved to {out}")


if __name__ == "__main__":
    main()
