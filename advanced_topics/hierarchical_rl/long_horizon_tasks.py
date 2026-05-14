"""
Long-horizon sparse-reward task.

The agent must collect a key, pass a door, then reach treasure. A flat learner
only receives the final reward. A hierarchical learner gets small subgoal
rewards for the natural milestones.

Run:
    python long_horizon_tasks.py
Output:
    outputs/long_horizon_tasks.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_POS = 15
START = 0
KEY = 4
DOOR = 9
TREASURE = 14
N_KEY_STATES = 2
N_DOOR_STATES = 2
N_STATES = N_POS * N_KEY_STATES * N_DOOR_STATES
N_ACTIONS = 2


def encode(position, has_key, door_open):
    return (position * N_KEY_STATES + int(has_key)) * N_DOOR_STATES + int(door_open)


def decode(state):
    door_open = state % N_DOOR_STATES
    state //= N_DOOR_STATES
    has_key = state % N_KEY_STATES
    position = state // N_KEY_STATES
    return position, bool(has_key), bool(door_open)


def env_step(state, action):
    position, has_key, door_open = decode(state)
    move = -1 if action == 0 else 1
    next_position = int(np.clip(position + move, 0, N_POS - 1))

    if next_position == DOOR and not has_key:
        next_position = position

    next_has_key = has_key or next_position == KEY
    next_door_open = door_open or (next_position == DOOR and next_has_key)
    done = next_position == TREASURE and next_door_open
    reward = 1.0 if done else 0.0
    return encode(next_position, next_has_key, next_door_open), reward, done


def shaped_reward(prev_state, next_state, done):
    if done:
        return 1.0
    _, had_key, door_was_open = decode(prev_state)
    _, has_key, door_open = decode(next_state)
    if has_key and not had_key:
        return 0.25
    if door_open and not door_was_open:
        return 0.25
    return -0.002


def choose_action(q, state, rng, epsilon):
    if rng.random() < epsilon:
        return int(rng.integers(N_ACTIONS))
    return int(np.argmax(q[state]))


def train_flat(episodes=1200, alpha=0.25, gamma=0.97, seed=3):
    rng = np.random.default_rng(seed)
    q = np.zeros((N_STATES, N_ACTIONS))
    successes = np.zeros(episodes)
    lengths = np.zeros(episodes)

    for ep in range(episodes):
        state = encode(START, False, False)
        epsilon = max(0.05, 0.35 * (1.0 - ep / episodes))
        for t in range(90):
            action = choose_action(q, state, rng, epsilon)
            next_state, reward, done = env_step(state, action)
            target = reward if done else reward + gamma * np.max(q[next_state])
            q[state, action] += alpha * (target - q[state, action])
            state = next_state
            if done:
                successes[ep] = 1.0
                lengths[ep] = t + 1
                break
        if not successes[ep]:
            lengths[ep] = 90
    return q, successes, lengths


def train_hierarchical(episodes=1200, alpha=0.25, gamma=0.97, seed=5):
    rng = np.random.default_rng(seed)
    q = np.zeros((N_STATES, N_ACTIONS))
    successes = np.zeros(episodes)
    lengths = np.zeros(episodes)

    for ep in range(episodes):
        state = encode(START, False, False)
        epsilon = max(0.03, 0.25 * (1.0 - ep / episodes))
        for t in range(90):
            action = choose_action(q, state, rng, epsilon)
            next_state, env_reward, done = env_step(state, action)
            reward = shaped_reward(state, next_state, done)
            target = reward if done else reward + gamma * np.max(q[next_state])
            q[state, action] += alpha * (target - q[state, action])
            state = next_state
            if done:
                successes[ep] = 1.0 if env_reward > 0 else 0.0
                lengths[ep] = t + 1
                break
        if not successes[ep]:
            lengths[ep] = 90
    return q, successes, lengths


def greedy_path(q):
    state = encode(START, False, False)
    path = [decode(state)[0]]
    events = []
    for _ in range(90):
        action = int(np.argmax(q[state]))
        next_state, _, done = env_step(state, action)
        old = decode(state)
        new = decode(next_state)
        if new[1] and not old[1]:
            events.append(("key", len(path)))
        if new[2] and not old[2]:
            events.append(("door", len(path)))
        path.append(new[0])
        state = next_state
        if done:
            events.append(("treasure", len(path) - 1))
            break
    return path, events


def moving_average(values, window=60):
    return np.array([values[max(0, i - window + 1):i + 1].mean() for i in range(len(values))])


def plot(flat, hier):
    flat_q, flat_success, flat_lengths = flat
    hier_q, hier_success, hier_lengths = hier
    h_path, h_events = greedy_path(hier_q)
    f_path, _ = greedy_path(flat_q)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1.2, 1])
    axes[0].plot(moving_average(flat_success), color="#de2d26", lw=2.2, label="flat sparse reward")
    axes[0].plot(moving_average(hier_success), color="#238b45", lw=2.6, label="hierarchical milestones")
    axes[0].set_title("Long-horizon task: milestones make the distant reward reachable")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Success rate, moving average")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(h_path, np.zeros(len(h_path)), "-o", color="#238b45", lw=3, label="hierarchical greedy path")
    axes[1].plot(f_path, np.ones(len(f_path)) * 0.18, "-o", color="#de2d26", alpha=0.6, label="flat greedy path")
    axes[1].scatter([START, KEY, DOOR, TREASURE], [0, 0, 0, 0], s=[90, 120, 120, 180],
                    color=["#636363", "#fdae6b", "#6baed6", "#756bb1"], zorder=5)
    axes[1].text(START, -0.12, "start", ha="center")
    axes[1].text(KEY, -0.12, "key", ha="center")
    axes[1].text(DOOR, -0.12, "door", ha="center")
    axes[1].text(TREASURE, -0.12, "treasure", ha="center")
    for label, index in h_events:
        axes[1].annotate(label, xy=(h_path[index], 0), xytext=(h_path[index], 0.35),
                         arrowprops={"arrowstyle": "->", "color": "#525252"}, ha="center")
    axes[1].set_xlim(-0.7, N_POS - 0.3)
    axes[1].set_ylim(-0.25, 0.55)
    axes[1].set_yticks([])
    axes[1].set_xlabel("Position on the corridor")
    axes[1].legend(loc="upper left")
    axes[1].grid(axis="x", alpha=0.2)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "long_horizon_tasks.png")
    plt.savefig(out, dpi=130)
    return out


def main():
    print("=== Long-horizon sparse task: key -> door -> treasure ===")
    flat = train_flat()
    hier = train_hierarchical()
    out = plot(flat, hier)

    _, flat_success, flat_lengths = flat
    _, hier_success, hier_lengths = hier
    print(f"Flat learner final success rate: {flat_success[-200:].mean():.2f}")
    print(f"Hierarchical learner final success rate: {hier_success[-200:].mean():.2f}")
    print(f"Flat learner final average length: {flat_lengths[-200:].mean():.1f}")
    print(f"Hierarchical learner final average length: {hier_lengths[-200:].mean():.1f}")
    print(f"Plot saved to {out}")


if __name__ == "__main__":
    main()
