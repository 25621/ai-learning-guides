"""Fine-tune a tiny policy with PPO using the trained reward model.

We treat the "policy" as a small network that takes a prompt embedding and
emits a response embedding. The reward model scores the response. A KL
penalty against a frozen reference keeps the policy from drifting too far
from its starting behavior - this is the core RLHF safety knob.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from reward_modeling import RewardModel

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


class Policy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mean_net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.Tanh(),
            nn.Linear(64, dim),
        )
        self.log_std = nn.Parameter(torch.zeros(dim) - 0.5)

    def forward(self, x):
        return self.mean_net(x), self.log_std.expand_as(self.mean_net(x))


def sample_and_logprob(policy, prompts):
    means, log_stds = policy(prompts)
    stds = log_stds.exp()
    dist = torch.distributions.Normal(means, stds)
    actions = dist.rsample()
    log_probs = dist.log_prob(actions).sum(dim=-1)
    return actions, log_probs


def logprob_of(policy, prompts, actions):
    means, log_stds = policy(prompts)
    dist = torch.distributions.Normal(means, log_stds.exp())
    return dist.log_prob(actions).sum(dim=-1)


def train_ppo_rlhf():
    torch.manual_seed(0)
    dim = 16
    reward_model = RewardModel(dim)
    rm_path = os.path.join(OUT_DIR, "reward_model.pt")
    if os.path.exists(rm_path):
        reward_model.load_state_dict(torch.load(rm_path))
        print(f"Loaded reward model from {rm_path}")
    else:
        print("No reward model found; using random reward model.")
    reward_model.eval()
    for p in reward_model.parameters():
        p.requires_grad_(False)

    policy = Policy(dim)
    ref_policy = Policy(dim)
    ref_policy.load_state_dict(policy.state_dict())
    for p in ref_policy.parameters():
        p.requires_grad_(False)
    ref_policy.eval()

    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    num_iters = 150
    batch_size = 128
    ppo_epochs = 4
    clip_eps = 0.2
    kl_coeff = 0.05

    rewards_hist, kl_hist, shaped_hist = [], [], []

    for it in range(num_iters):
        # Rollout: sample a batch from the current policy.
        prompts = torch.randn(batch_size, dim)
        with torch.no_grad():
            actions, old_log_probs = sample_and_logprob(policy, prompts)
            rm_scores = reward_model(actions)

            ref_log_probs = logprob_of(ref_policy, prompts, actions)
            kl = old_log_probs - ref_log_probs

            # Shaped reward = RM score - KL penalty (InstructGPT-style).
            shaped = rm_scores - kl_coeff * kl
            advantages = (shaped - shaped.mean()) / (shaped.std() + 1e-8)

        # PPO optimization with clipped surrogate objective.
        for _ in range(ppo_epochs):
            new_log_probs = logprob_of(policy, prompts, actions)
            ratio = (new_log_probs - old_log_probs).exp()
            unclipped = ratio * advantages
            clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
            loss = -torch.min(unclipped, clipped).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        rewards_hist.append(rm_scores.mean().item())
        kl_hist.append(kl.mean().item())
        shaped_hist.append(shaped.mean().item())

        if (it + 1) % 15 == 0:
            print(
                f"Iter {it+1:3d} | reward={rewards_hist[-1]:+.3f} "
                f"| KL={kl_hist[-1]:+.3f} | shaped={shaped_hist[-1]:+.3f}"
            )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(rewards_hist, color="#1f77b4")
    axes[0].set_title("Reward Model Score Over Training")
    axes[0].set_xlabel("PPO iteration")
    axes[0].set_ylabel("Mean RM score")
    axes[0].grid(alpha=0.3)

    axes[1].plot(kl_hist, color="#ff7f0e")
    axes[1].set_title("KL(policy || reference)")
    axes[1].set_xlabel("PPO iteration")
    axes[1].set_ylabel("KL divergence")
    axes[1].grid(alpha=0.3)

    axes[2].plot(shaped_hist, color="#2ca02c", label="reward - β·KL")
    axes[2].plot(rewards_hist, color="#1f77b4", alpha=0.5, label="raw reward")
    axes[2].set_title("Shaped vs Raw Reward")
    axes[2].set_xlabel("PPO iteration")
    axes[2].set_ylabel("Score")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "ppo_fine_tuning.png")
    plt.savefig(out_path, dpi=110)
    plt.close()

    print(f"Reward gained: {rewards_hist[-1] - rewards_hist[0]:+.3f}")
    print(f"Final KL:      {kl_hist[-1]:+.3f}")
    print(f"Saved plot:    {out_path}")


if __name__ == "__main__":
    train_ppo_rlhf()
