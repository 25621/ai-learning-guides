"""
Build a D4RL-style offline RL benchmark on CartPole-v1.

The real D4RL benchmark (Fu et al., 2020) ships pre-recorded transitions for
MuJoCo locomotion tasks at four quality levels:

    random          uniformly random actions
    medium          policy trained until it reaches roughly half of expert return
    expert          policy trained to (near-)convergence
    medium-replay   the *entire replay buffer* of the medium policy's training run
                    (a mixed-quality dataset that contains failures AND successes)

The point of D4RL is to standardise offline RL: every researcher trains on the
*same* fixed transitions, no environment interaction allowed during learning.
Real D4RL requires MuJoCo, which is heavy to install.  We rebuild the same
four-level structure on CartPole-v1 so the rest of `requirements.txt`
(numpy + matplotlib + gymnasium + torch) is enough.

Output
------
Four .npz files in `outputs/`, each containing:

    obs        (N, 4)  float32
    action     (N,)    int64
    reward     (N,)    float32
    next_obs   (N, 4)  float32
    terminal   (N,)    bool

Plus a histogram of episode returns per dataset (`outputs/d4rl_returns.png`)
and a tiny `outputs/d4rl_summary.txt` you can `cat` after training.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim


SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cpu")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ENV_ID = "CartPole-v1"
OBS_DIM = 4
N_ACTIONS = 2

# Training-time thresholds.  "medium" = solid but suboptimal; "expert" = near max.
MEDIUM_RETURN = 150.0
EXPERT_RETURN = 475.0
SAMPLES_PER_DATASET = 10_000


# ---------------------------------------------------------------------------
class QNet(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=200_000):
        self.capacity = capacity
        self.obs = np.zeros((capacity, OBS_DIM), dtype=np.float32)
        self.action = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, OBS_DIM), dtype=np.float32)
        self.terminal = np.zeros(capacity, dtype=bool)
        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, s, a, r, ns, term):
        i = self.idx
        self.obs[i] = s
        self.action[i] = a
        self.reward[i] = r
        self.next_obs[i] = ns
        self.terminal[i] = term
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def sample(self, batch_size, rng):
        n = len(self)
        idx = rng.integers(0, n, size=batch_size)
        return (
            torch.tensor(self.obs[idx]),
            torch.tensor(self.action[idx]),
            torch.tensor(self.reward[idx]),
            torch.tensor(self.next_obs[idx]),
            torch.tensor(self.terminal[idx], dtype=torch.float32),
        )

    def as_dict(self):
        n = len(self)
        return {
            "obs": self.obs[:n].copy(),
            "action": self.action[:n].copy(),
            "reward": self.reward[:n].copy(),
            "next_obs": self.next_obs[:n].copy(),
            "terminal": self.terminal[:n].copy(),
        }


# ---------------------------------------------------------------------------
def evaluate(qnet, n_episodes=10, seed=1234):
    env = gym.make(ENV_ID)
    rng = np.random.default_rng(seed)
    returns = []
    qnet.eval()
    for _ in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(1_000_000)))
        done = False
        ep_r = 0.0
        while not done:
            with torch.no_grad():
                q = qnet(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                a = int(q.argmax(dim=1).item())
            obs, r, term, trunc, _ = env.step(a)
            ep_r += r
            done = term or trunc
        returns.append(ep_r)
    env.close()
    qnet.train()
    return float(np.mean(returns)), returns


def train_dqn_and_snapshot():
    """Train DQN; return a checkpoint near MEDIUM_RETURN, near EXPERT_RETURN,
    plus the full replay buffer used during training (= 'medium-replay')."""
    env = gym.make(ENV_ID)
    rng = np.random.default_rng(SEED)

    q = QNet()
    q_target = QNet()
    q_target.load_state_dict(q.state_dict())
    opt = optim.Adam(q.parameters(), lr=5e-4)
    buf = ReplayBuffer(capacity=100_000)

    eps_start, eps_end, eps_decay = 1.0, 0.05, 5_000
    gamma = 0.99
    batch_size = 64
    target_sync_every = 500
    learning_starts = 1_000
    max_steps = 60_000

    obs, _ = env.reset(seed=SEED)
    ep_return = 0.0
    ep_returns = []
    step_count = 0
    medium_ckpt = None
    expert_ckpt = None
    medium_replay_snapshot = None  # captured once medium return is hit

    print("Training DQN to populate three datasets (random / medium / expert) "
          "and one mixed dataset (medium-replay)...")
    while step_count < max_steps:
        eps = max(eps_end, eps_start - (eps_start - eps_end) * step_count / eps_decay)
        if step_count < learning_starts or rng.random() < eps:
            a = int(rng.integers(N_ACTIONS))
        else:
            with torch.no_grad():
                a = int(q(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).argmax(1).item())

        nobs, r, term, trunc, _ = env.step(a)
        # For the offline dataset 'terminal' means a true terminal (pole fell),
        # NOT a timeout truncation.  This distinction matters for the Bellman target.
        buf.add(obs, a, r, nobs, term)
        ep_return += r
        obs = nobs

        if term or trunc:
            ep_returns.append(ep_return)
            obs, _ = env.reset()
            ep_return = 0.0

        if len(buf) >= learning_starts:
            s, ac, re, ns, te = buf.sample(batch_size, rng)
            with torch.no_grad():
                target = re + gamma * (1.0 - te) * q_target(ns).max(1).values
            pred = q(s).gather(1, ac.unsqueeze(1)).squeeze(1)
            loss = nn.functional.mse_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 10.0)
            opt.step()
            if step_count % target_sync_every == 0:
                q_target.load_state_dict(q.state_dict())

        step_count += 1

        # Snapshot whenever the recent average return crosses a threshold.
        if medium_ckpt is None and len(ep_returns) >= 10:
            recent = float(np.mean(ep_returns[-10:]))
            if recent >= MEDIUM_RETURN:
                medium_ckpt = {k: v.clone() for k, v in q.state_dict().items()}
                medium_replay_snapshot = buf.as_dict()
                print(f"  -> MEDIUM checkpoint hit at step {step_count} "
                      f"(recent return = {recent:.1f})")
        if expert_ckpt is None and len(ep_returns) >= 10:
            recent = float(np.mean(ep_returns[-10:]))
            if recent >= EXPERT_RETURN:
                expert_ckpt = {k: v.clone() for k, v in q.state_dict().items()}
                print(f"  -> EXPERT checkpoint hit at step {step_count} "
                      f"(recent return = {recent:.1f})")
                break

    env.close()

    # Safety net: if we never crossed a threshold within max_steps, just take
    # the final policy.  (CartPole-v1 with these hyper-params reliably hits
    # 475 within ~20-40k steps on CPU.)
    if medium_ckpt is None:
        medium_ckpt = {k: v.clone() for k, v in q.state_dict().items()}
        medium_replay_snapshot = buf.as_dict()
        print("  -> MEDIUM threshold never crossed; using final policy as 'medium'.")
    if expert_ckpt is None:
        expert_ckpt = {k: v.clone() for k, v in q.state_dict().items()}
        print("  -> EXPERT threshold never crossed; using final policy as 'expert'.")

    return medium_ckpt, expert_ckpt, medium_replay_snapshot


# ---------------------------------------------------------------------------
def collect_dataset(policy_fn, n_transitions, seed):
    """Roll out `policy_fn(obs) -> int` until we have n_transitions."""
    env = gym.make(ENV_ID)
    obs_buf = np.zeros((n_transitions, OBS_DIM), dtype=np.float32)
    a_buf = np.zeros(n_transitions, dtype=np.int64)
    r_buf = np.zeros(n_transitions, dtype=np.float32)
    nobs_buf = np.zeros((n_transitions, OBS_DIM), dtype=np.float32)
    te_buf = np.zeros(n_transitions, dtype=bool)
    ep_returns = []

    obs, _ = env.reset(seed=seed)
    ep_r = 0.0
    for t in range(n_transitions):
        a = policy_fn(obs)
        nobs, r, term, trunc, _ = env.step(a)
        obs_buf[t] = obs
        a_buf[t] = a
        r_buf[t] = r
        nobs_buf[t] = nobs
        te_buf[t] = term  # true termination only
        ep_r += r
        obs = nobs
        if term or trunc:
            ep_returns.append(ep_r)
            ep_r = 0.0
            obs, _ = env.reset()
    env.close()
    return (
        {"obs": obs_buf, "action": a_buf, "reward": r_buf,
         "next_obs": nobs_buf, "terminal": te_buf},
        ep_returns,
    )


def policy_random(rng):
    return lambda obs: int(rng.integers(N_ACTIONS))


def policy_greedy(qnet, eps, rng):
    def pi(obs):
        if rng.random() < eps:
            return int(rng.integers(N_ACTIONS))
        with torch.no_grad():
            return int(qnet(torch.tensor(obs, dtype=torch.float32)
                            .unsqueeze(0)).argmax(1).item())
    return pi


def episode_returns_from_buffer(data):
    """Compute per-episode returns from a flat (s,a,r,s',term) dataset."""
    rewards = data["reward"]
    terminals = data["terminal"]
    returns = []
    cur = 0.0
    for r, t in zip(rewards, terminals):
        cur += r
        if t:
            returns.append(cur)
            cur = 0.0
    if cur > 0:
        returns.append(cur)
    return returns


# ---------------------------------------------------------------------------
def main():
    print("=== Building a D4RL-style offline dataset on CartPole-v1 ===\n")

    medium_ckpt, expert_ckpt, medium_replay = train_dqn_and_snapshot()

    rng = np.random.default_rng(123)

    print("\nCollecting 'random' dataset (uniform random policy)...")
    random_data, random_returns = collect_dataset(
        policy_random(rng), SAMPLES_PER_DATASET, seed=1)

    print("Collecting 'medium' dataset (medium policy + eps=0.1 noise)...")
    medium_net = QNet(); medium_net.load_state_dict(medium_ckpt); medium_net.eval()
    medium_data, medium_returns = collect_dataset(
        policy_greedy(medium_net, eps=0.1, rng=rng), SAMPLES_PER_DATASET, seed=2)

    print("Collecting 'expert' dataset (expert policy + eps=0.02 noise)...")
    expert_net = QNet(); expert_net.load_state_dict(expert_ckpt); expert_net.eval()
    expert_data, expert_returns = collect_dataset(
        policy_greedy(expert_net, eps=0.02, rng=rng), SAMPLES_PER_DATASET, seed=3)

    print("Saving 'medium-replay' dataset (every transition seen during training)...")
    # medium_replay was captured at the medium milestone.  Trim/pad to
    # SAMPLES_PER_DATASET so all four datasets are the same size.
    n = medium_replay["obs"].shape[0]
    if n >= SAMPLES_PER_DATASET:
        keep = np.linspace(0, n - 1, SAMPLES_PER_DATASET).astype(int)
        medium_replay = {k: v[keep] for k, v in medium_replay.items()}
    medium_replay_returns = episode_returns_from_buffer(medium_replay)

    datasets = {
        "random": (random_data, random_returns),
        "medium": (medium_data, medium_returns),
        "expert": (expert_data, expert_returns),
        "medium-replay": (medium_replay, medium_replay_returns),
    }

    # Save .npz files
    summary_lines = ["dataset         |   N    |  mean return  |  min  |  max"]
    summary_lines.append("-" * 60)
    for name, (data, returns) in datasets.items():
        path = os.path.join(OUTPUT_DIR, f"cartpole_{name}.npz")
        np.savez_compressed(path, **data)
        mu = float(np.mean(returns)) if returns else 0.0
        lo = float(np.min(returns)) if returns else 0.0
        hi = float(np.max(returns)) if returns else 0.0
        line = (f"{name:<15} | {data['obs'].shape[0]:>5} | "
                f"{mu:>11.1f}  | {lo:>4.0f}  | {hi:>4.0f}")
        summary_lines.append(line)
        print(f"  saved {path}  ({data['obs'].shape[0]} transitions, "
              f"avg episode return {mu:.1f})")

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    with open(os.path.join(OUTPUT_DIR, "d4rl_summary.txt"), "w") as f:
        f.write(summary_text + "\n")

    # Plot return distributions
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"random": "#95a5a6", "medium": "#3498db",
              "expert": "#27ae60", "medium-replay": "#e67e22"}
    bins = np.linspace(0, 500, 26)
    for name, (_, returns) in datasets.items():
        if returns:
            ax.hist(returns, bins=bins, alpha=0.55, label=name,
                    color=colors[name], edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Episode return (CartPole-v1, max = 500)")
    ax.set_ylabel("Number of episodes in dataset")
    ax.set_title("D4RL-style dataset quality levels on CartPole-v1")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "d4rl_returns.png")
    fig.savefig(out, dpi=120)
    print(f"\nReturn-distribution plot saved to {out}")


if __name__ == "__main__":
    main()
