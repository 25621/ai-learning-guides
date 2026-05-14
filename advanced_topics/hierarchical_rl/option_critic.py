"""
Option-Critic on a small two-room gridworld.

The example keeps the architecture tabular so the moving parts are visible:
  - Q(state, option): the high-level value of each option
  - pi(action | state, option): the low-level policy inside an option
  - beta(state, option): the probability that an option terminates

Run:
    python option_critic.py
Output:
    outputs/option_critic.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

GRID = [
    ".....#....",
    ".....#....",
    "..........",
    ".....#....",
    ".....#....",
]
ROWS, COLS = len(GRID), len(GRID[0])
START = (4, 0)
GOAL = (0, 9)
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ["up", "down", "left", "right"]
N_STATES = ROWS * COLS
N_ACTIONS = len(ACTIONS)
N_OPTIONS = 4


def state_id(pos):
    return pos[0] * COLS + pos[1]


def pos_from_state(s):
    return divmod(s, COLS)


def is_wall(pos):
    r, c = pos
    return r < 0 or r >= ROWS or c < 0 or c >= COLS or GRID[r][c] == "#"


def step(state, action):
    r, c = pos_from_state(state)
    dr, dc = ACTIONS[action]
    nxt = (r + dr, c + dc)
    if is_wall(nxt):
        nxt = (r, c)
    done = nxt == GOAL
    reward = 1.0 if done else -0.01
    return state_id(nxt), reward, done


def softmax(x):
    z = x - np.max(x)
    exp = np.exp(z)
    return exp / exp.sum()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def choose_option(q_options, state, rng, epsilon):
    if rng.random() < epsilon:
        return int(rng.integers(N_OPTIONS))
    return int(np.argmax(q_options[state]))


def sample_action(policy_logits, state, option, rng):
    probs = softmax(policy_logits[state, option])
    return int(rng.choice(N_ACTIONS, p=probs)), probs


def train_option_critic(
    episodes=650,
    alpha_q=0.18,
    alpha_pi=0.035,
    alpha_beta=0.03,
    gamma=0.98,
    epsilon=0.12,
    seed=4,
):
    rng = np.random.default_rng(seed)
    q_options = np.zeros((N_STATES, N_OPTIONS))
    policy_logits = rng.normal(0.0, 0.03, size=(N_STATES, N_OPTIONS, N_ACTIONS))
    beta_logits = np.full((N_STATES, N_OPTIONS), -1.0)
    episode_steps = []
    option_switches = []

    for _ in range(episodes):
        state = state_id(START)
        option = choose_option(q_options, state, rng, epsilon)
        steps = 0
        switches = 0
        done = False

        while not done and steps < 250:
            action, probs = sample_action(policy_logits, state, option, rng)
            next_state, reward, done = step(state, action)

            beta = sigmoid(beta_logits[next_state, option])
            continuation = q_options[next_state, option]
            new_option_value = np.max(q_options[next_state])
            mixed_next_value = (1.0 - beta) * continuation + beta * new_option_value
            target = reward if done else reward + gamma * mixed_next_value
            td_error = target - q_options[state, option]
            q_options[state, option] += alpha_q * td_error

            grad = -probs
            grad[action] += 1.0
            policy_logits[state, option] += alpha_pi * td_error * grad

            option_advantage = q_options[next_state, option] - np.max(q_options[next_state])
            beta_logits[next_state, option] -= alpha_beta * option_advantage * beta * (1.0 - beta)

            if done:
                break

            should_stop = rng.random() < sigmoid(beta_logits[next_state, option])
            if should_stop:
                option = choose_option(q_options, next_state, rng, epsilon)
                switches += 1

            state = next_state
            steps += 1

        episode_steps.append(steps + 1)
        option_switches.append(switches)

    return q_options, policy_logits, beta_logits, np.array(episode_steps), np.array(option_switches)


def greedy_rollout(q_options, policy_logits, beta_logits):
    state = state_id(START)
    option = int(np.argmax(q_options[state]))
    path = [state]
    options = [option]

    for _ in range(120):
        action = int(np.argmax(policy_logits[state, option]))
        next_state, _, done = step(state, action)
        path.append(next_state)
        if done:
            break
        if sigmoid(beta_logits[next_state, option]) > 0.5:
            option = int(np.argmax(q_options[next_state]))
        state = next_state
        options.append(option)
    return path, options


def moving_average(values, window=25):
    return np.array([values[max(0, i - window + 1):i + 1].mean() for i in range(len(values))])


def plot_results(q_options, policy_logits, beta_logits, episode_steps, option_switches):
    path, options = greedy_rollout(q_options, policy_logits, beta_logits)
    option_grid = np.full((ROWS, COLS), np.nan)
    for s in range(N_STATES):
        pos = pos_from_state(s)
        if not is_wall(pos):
            option_grid[pos] = np.argmax(q_options[s])

    visit_grid = np.zeros((ROWS, COLS))
    for s in path:
        visit_grid[pos_from_state(s)] += 1

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    axes[0].plot(moving_average(episode_steps), color="#1f77b4", lw=2.5, label="steps")
    axes[0].plot(moving_average(option_switches), color="#d62728", lw=2.0, label="option switches")
    axes[0].set_title("Option-Critic learns shorter routines")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Moving average")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    cmap = plt.get_cmap("Set2", N_OPTIONS)
    masked_options = np.ma.masked_invalid(option_grid)
    axes[1].imshow(masked_options, cmap=cmap, vmin=0, vmax=N_OPTIONS - 1)
    axes[1].set_title("Preferred option by location")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    for r in range(ROWS):
        for c in range(COLS):
            if GRID[r][c] == "#":
                axes[1].text(c, r, "#", ha="center", va="center", fontsize=14, weight="bold")
            elif (r, c) == START:
                axes[1].text(c, r, "S", ha="center", va="center", weight="bold")
            elif (r, c) == GOAL:
                axes[1].text(c, r, "G", ha="center", va="center", weight="bold")

    axes[2].imshow(visit_grid, cmap="Blues")
    axes[2].set_title("Greedy route after training")
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    coords = [pos_from_state(s) for s in path]
    axes[2].plot([c for _, c in coords], [r for r, _ in coords], color="#ff7f0e", lw=3)
    axes[2].scatter([START[1]], [START[0]], marker="o", s=90, color="#2ca02c", label="start")
    axes[2].scatter([GOAL[1]], [GOAL[0]], marker="*", s=180, color="#d62728", label="goal")
    axes[2].legend(loc="lower right")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "option_critic.png")
    plt.savefig(out, dpi=130)
    return out, path, options


def main():
    print("=== Option-Critic: two-room gridworld ===")
    q_options, policy_logits, beta_logits, steps, switches = train_option_critic()
    out, path, options = plot_results(q_options, policy_logits, beta_logits, steps, switches)

    print(f"First 50 episodes: {steps[:50].mean():.1f} steps on average")
    print(f"Last 50 episodes:  {steps[-50:].mean():.1f} steps on average")
    print(f"Greedy route length after training: {len(path) - 1} steps")
    print(f"Options used on the greedy route: {sorted(set(options))}")
    print(f"Plot saved to {out}")


if __name__ == "__main__":
    main()
