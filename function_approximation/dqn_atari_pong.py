"""
DQN on Atari Pong (ALE/Pong-v5)

Implements the full DQN pipeline from Mnih et al. (2015):
  - CNN feature extractor (3 conv layers)
  - Frame preprocessing: grayscale + resize to 84x84 + normalize
  - Frame stacking (4 consecutive frames for motion)
  - Replay buffer
  - Target network with periodic sync
  - Reward clipping to [-1, +1]

Note: Full convergence to positive reward requires ~10M frames.
This script runs 300K steps to demonstrate the architecture and show early
training progress. Learning curves are saved for inspection.
"""

import os
import random
import collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

gym.register_envs(ale_py)

DEVICE = torch.device("cpu")

# Hyperparameters (from original DQN paper, adapted for CPU demo)
BUFFER_CAPACITY = 5_000        # reduced for CPU RAM; original paper uses 1M
BATCH_SIZE = 32
GAMMA = 0.99
LR = 1e-4
EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY_STEPS = 50_000   # linear decay over this many steps
TARGET_UPDATE_FREQ = 500       # steps between target syncs
WARMUP_STEPS = 2_000
TRAIN_FREQ = 4                 # update every N environment steps
MAX_STEPS = 100_000            # total environment steps for this demo
LOG_FREQ = 5_000               # print progress every N steps


# ─── Frame preprocessing ─────────────────────────────────────────────────────

def preprocess_frame(obs):
    """Convert raw Atari frame to 84×84 grayscale float."""
    img = Image.fromarray(obs).convert('L')         # → grayscale (210×160)
    img = img.resize((84, 84), Image.BILINEAR)      # → 84×84
    return np.array(img, dtype=np.float32) / 255.0  # → [0, 1]


class FrameStack:
    """Maintains a sliding window of the last `k` preprocessed frames."""

    def __init__(self, k=4):
        self.k = k
        self.frames = collections.deque(maxlen=k)

    def reset(self, obs):
        frame = preprocess_frame(obs)
        for _ in range(self.k):
            self.frames.append(frame)
        return self._get()

    def step(self, obs):
        self.frames.append(preprocess_frame(obs))
        return self._get()

    def _get(self):
        return np.stack(self.frames, axis=0)  # shape: (k, 84, 84)


# ─── CNN Q-Network ────────────────────────────────────────────────────────────

class AtariQNetwork(nn.Module):
    """CNN from the original DQN paper (Mnih et al., 2015)."""

    def __init__(self, n_frames=4, n_actions=6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_frames, 32, kernel_size=8, stride=4),  # → (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),        # → (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),        # → (64, 7, 7)
            nn.ReLU(),
        )
        conv_out = 64 * 7 * 7
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


# ─── Replay Buffer ────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Stores frames as uint8 (0-255) to reduce RAM; converts to float32 on sample."""

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Store as uint8 to save 4× memory vs float32
        s = (state * 255).astype(np.uint8)
        s_ = (next_state * 255).astype(np.uint8)
        self.buffer.append((s, action, reward, s_, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_, d = zip(*batch)
        return (
            np.array(s, dtype=np.float32) / 255.0,
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.array(s_, dtype=np.float32) / 255.0,
            np.array(d, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ─── DQN Agent ────────────────────────────────────────────────────────────────

class DQNAtariAgent:
    def __init__(self, n_actions=6):
        self.n_actions = n_actions
        self.qnet = AtariQNetwork(n_actions=n_actions).to(DEVICE)
        self.target_net = AtariQNetwork(n_actions=n_actions).to(DEVICE)
        self.target_net.load_state_dict(self.qnet.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.qnet.parameters(), lr=LR)
        self.loss_fn = nn.SmoothL1Loss()
        self.buffer = ReplayBuffer(BUFFER_CAPACITY)

    def act(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            return int(self.qnet(s).argmax(dim=1).item())

    def update(self):
        if len(self.buffer) < WARMUP_STEPS:
            return None

        s, a, r, s_, d = self.buffer.sample(BATCH_SIZE)
        s = torch.tensor(s, device=DEVICE)
        a = torch.tensor(a, device=DEVICE).unsqueeze(1)
        r = torch.tensor(r, device=DEVICE).unsqueeze(1)
        s_ = torch.tensor(s_, device=DEVICE)
        d = torch.tensor(d, device=DEVICE).unsqueeze(1)

        with torch.no_grad():
            q_next = self.target_net(s_).max(dim=1, keepdim=True).values
            target = r + GAMMA * q_next * (1 - d)

        q_pred = self.qnet(s).gather(1, a)
        loss = self.loss_fn(q_pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.qnet.parameters(), 10.0)
        self.optimizer.step()
        return loss.item()

    def sync_target(self):
        self.target_net.load_state_dict(self.qnet.state_dict())


# ─── Training loop ────────────────────────────────────────────────────────────

def linear_epsilon(step):
    """Linearly decay epsilon from START to MIN over DECAY_STEPS."""
    frac = min(1.0, step / EPSILON_DECAY_STEPS)
    return EPSILON_START + frac * (EPSILON_MIN - EPSILON_START)


def train():
    env = gym.make("ALE/Pong-v5")
    n_actions = env.action_space.n
    agent = DQNAtariAgent(n_actions=n_actions)
    stacker = FrameStack(k=4)

    total_steps = 0
    episode = 0
    episode_rewards = []
    step_log = []   # (step, avg_reward_over_last_10_eps)

    obs, _ = env.reset(seed=42)
    state = stacker.reset(obs)
    ep_reward = 0

    print(f"Device: {DEVICE}")
    print(f"Network params: {sum(p.numel() for p in agent.qnet.parameters()):,}")
    print(f"Training for {MAX_STEPS:,} steps (demo run)...")
    print(f"Note: Full convergence to +reward requires ~10M steps.\n")

    while total_steps < MAX_STEPS:
        epsilon = linear_epsilon(total_steps)
        action = agent.act(state, epsilon)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = stacker.step(next_obs)

        # Clip reward to [-1, 1] (standard DQN trick)
        clipped_reward = float(np.clip(reward, -1, 1))
        agent.buffer.push(state, action, clipped_reward, next_state, float(done))
        ep_reward += reward  # track unclipped for logging

        state = next_state
        total_steps += 1

        # Train every TRAIN_FREQ steps
        if total_steps % TRAIN_FREQ == 0:
            agent.update()

        # Sync target network
        if total_steps % TARGET_UPDATE_FREQ == 0:
            agent.sync_target()

        if done:
            episode += 1
            episode_rewards.append(ep_reward)

            if total_steps % LOG_FREQ < 500 or episode % 20 == 0:
                avg10 = float(np.mean(episode_rewards[-10:]))
                step_log.append((total_steps, avg10))
                print(f"  Step {total_steps:7,} | Ep {episode:4d} | "
                      f"ep_reward={ep_reward:6.1f} | avg10={avg10:6.1f} | ε={epsilon:.3f}")

            obs, _ = env.reset()
            state = stacker.reset(obs)
            ep_reward = 0

    env.close()
    return episode_rewards, step_log


def plot_results(episode_rewards, step_log):
    os.makedirs("outputs", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Episode rewards
    r = np.array(episode_rewards)
    if len(r) >= 10:
        window = min(10, len(r))
        rolling = np.convolve(r, np.ones(window) / window, mode='valid')
        axes[0].plot(r, alpha=0.3, color='steelblue', linewidth=0.8)
        axes[0].plot(range(window - 1, len(r)), rolling, color='navy',
                     linewidth=2, label=f'{window}-ep avg')
    axes[0].axhline(0, color='green', linestyle='--', linewidth=1.5, label='Positive reward')
    axes[0].axhline(-21, color='red', linestyle='--', linewidth=1, label='Worst score (−21)')
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Episode Reward (unclipped)")
    axes[0].set_title("DQN on Atari Pong — Episode Rewards (300K step demo)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Step-based log
    if step_log:
        steps_x, avg_y = zip(*step_log)
        axes[1].plot(steps_x, avg_y, color='crimson', linewidth=2, marker='o', markersize=4)
        axes[1].axhline(0, color='green', linestyle='--', linewidth=1.5)
        axes[1].set_xlabel("Environment Steps")
        axes[1].set_ylabel("Avg Reward (last 10 eps)")
        axes[1].set_title("Training Progress vs Steps")
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = "outputs/dqn_atari_pong.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Plot saved to {path}")


def main():
    print("=== DQN on Atari Pong (300K step demo) ===\n")
    episode_rewards, step_log = train()

    print(f"\nTraining complete.")
    print(f"Total episodes: {len(episode_rewards)}")
    if episode_rewards:
        print(f"Final avg reward (last 10 eps): {np.mean(episode_rewards[-10:]):.1f}")
        print(f"Best episode reward: {max(episode_rewards):.1f}")
        print(f"Note: Pong starts at −21 and requires ~10M frames to reach +reward.")

    plot_results(episode_rewards, step_log)


if __name__ == "__main__":
    main()
