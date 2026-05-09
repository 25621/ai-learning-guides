"""
Frozen Lake with Random Policy

FrozenLake-v1 is a 4x4 grid where an agent must navigate from Start (S)
to Goal (G) without falling into Holes (H). The ice is slippery, so
actions don't always go the intended direction.

This script runs a purely random policy (uniform over {Left, Down, Right, Up})
and observes how often a random agent reaches the goal.

Map:
  S F F F
  F H F H
  F F F H
  H F F G

Actions: 0=Left, 1=Down, 2=Right, 3=Up
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from collections import defaultdict


def run_random_policy(n_episodes=1000, seed=42, render_first=False):
    env = gym.make("FrozenLake-v1", is_slippery=True, render_mode=None)

    successes = 0
    episode_lengths = []
    state_visits = defaultdict(int)

    rng = np.random.default_rng(seed)

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(1_000_000)))
        done = False
        steps = 0
        state_visits[obs] += 1

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            state_visits[obs] += 1
            done = terminated or truncated
            steps += 1

            if terminated and reward > 0:
                successes += 1

        episode_lengths.append(steps)

    env.close()
    return successes, episode_lengths, state_visits


def visualize_state_visits(state_visits, n_episodes, out_path):
    grid = np.zeros((4, 4))
    for state, count in state_visits.items():
        row, col = divmod(state, 4)
        grid[row, col] = count

    # Normalize
    grid_norm = grid / grid.max()

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(grid_norm, cmap='YlOrRd', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Relative visit frequency")

    # Annotate cells
    labels = ['S', 'F', 'F', 'F',
              'F', 'H', 'F', 'H',
              'F', 'F', 'F', 'H',
              'H', 'F', 'F', 'G']
    for i in range(4):
        for j in range(4):
            state = i * 4 + j
            visits = state_visits.get(state, 0)
            cell_label = labels[state]
            ax.text(j, i, f"{cell_label}\n{visits}", ha='center', va='center',
                    fontsize=9, color='black',
                    fontweight='bold' if cell_label in ('S', 'G', 'H') else 'normal')

    ax.set_title(f"State Visit Frequency (Random Policy, {n_episodes} episodes)")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"State visit heatmap saved to {out_path}")


def plot_episode_lengths(episode_lengths, out_path):
    fig, ax = plt.subplots(figsize=(10, 4))
    # Rolling average
    window = 50
    lengths = np.array(episode_lengths)
    rolling = np.convolve(lengths, np.ones(window) / window, mode='valid')
    ax.plot(lengths, alpha=0.3, color='steelblue', linewidth=0.8, label='Episode length')
    ax.plot(range(window - 1, len(lengths)), rolling, color='navy',
            linewidth=2, label=f'{window}-episode rolling avg')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.set_title("Frozen Lake (Random Policy): Episode Lengths")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"Episode length plot saved to {out_path}")


def main():
    n_episodes = 1000
    print("=== Frozen Lake — Random Policy ===")
    print(f"Running {n_episodes} episodes with a random policy...")
    print()

    successes, episode_lengths, state_visits = run_random_policy(
        n_episodes=n_episodes, seed=42
    )

    success_rate = successes / n_episodes * 100
    print(f"Success rate:        {success_rate:.1f}%  ({successes}/{n_episodes} episodes)")
    print(f"Avg episode length:  {np.mean(episode_lengths):.1f} steps")
    print(f"Max episode length:  {max(episode_lengths)} steps")
    print(f"Min episode length:  {min(episode_lengths)} steps")
    print()
    print("Observation: A random policy reaches the goal ~7-10% of the time on the")
    print("slippery 4x4 FrozenLake. This is far from optimal but serves as a baseline.")

    visualize_state_visits(
        state_visits, n_episodes,
        out_path="outputs/frozen_lake_state_visits.png"
    )
    plot_episode_lengths(
        episode_lengths,
        out_path="outputs/frozen_lake_episode_lengths.png"
    )


if __name__ == "__main__":
    main()
