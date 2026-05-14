"""Direct Preference Optimization (DPO).

DPO skips the explicit reward model entirely. It trains the policy directly
on preference pairs, using a closed-form loss that increases the probability
of chosen responses and decreases the probability of rejected ones - while
staying close to a frozen reference policy.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from reward_modeling import generate_preference_data

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

    def log_prob(self, prompts, responses):
        means = self.mean_net(prompts)
        stds = self.log_std.exp().expand_as(means)
        dist = torch.distributions.Normal(means, stds)
        return dist.log_prob(responses).sum(dim=-1)


def train_dpo():
    torch.manual_seed(0)
    dim = 16
    prompts, chosen, rejected = generate_preference_data(num_samples=2000, dim=dim, seed=1)

    policy = Policy(dim)
    ref_policy = Policy(dim)
    ref_policy.load_state_dict(policy.state_dict())
    for p in ref_policy.parameters():
        p.requires_grad_(False)
    ref_policy.eval()

    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    beta = 0.1

    losses, accuracies, margin_hist = [], [], []
    num_epochs = 300
    for epoch in range(num_epochs):
        log_pi_c = policy.log_prob(prompts, chosen)
        log_pi_r = policy.log_prob(prompts, rejected)
        with torch.no_grad():
            log_ref_c = ref_policy.log_prob(prompts, chosen)
            log_ref_r = ref_policy.log_prob(prompts, rejected)

        # DPO loss: -log sigmoid( beta * ((log pi_c - log ref_c) - (log pi_r - log ref_r)) )
        chosen_ratio = log_pi_c - log_ref_c
        rejected_ratio = log_pi_r - log_ref_r
        logits = beta * (chosen_ratio - rejected_ratio)
        loss = -torch.nn.functional.logsigmoid(logits).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # "Implicit reward" of DPO is beta * log(pi / ref).
            implicit_r_chosen = beta * chosen_ratio
            implicit_r_rejected = beta * rejected_ratio
            margin = (implicit_r_chosen - implicit_r_rejected).mean().item()
            acc = (implicit_r_chosen > implicit_r_rejected).float().mean().item()

        losses.append(loss.item())
        accuracies.append(acc)
        margin_hist.append(margin)

        if (epoch + 1) % 30 == 0:
            print(
                f"Epoch {epoch+1:3d} | loss={loss.item():.4f} | "
                f"acc={acc:.4f} | margin={margin:+.3f}"
            )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(losses, color="#1f77b4")
    axes[0].set_title("DPO Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)

    axes[1].plot(accuracies, color="#2ca02c")
    axes[1].set_title("Preference Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("P(chosen preferred)")
    axes[1].set_ylim(0.4, 1.02)
    axes[1].grid(alpha=0.3)

    axes[2].plot(margin_hist, color="#9467bd")
    axes[2].set_title("Implicit Reward Margin (chosen - rejected)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("β · Δ log-ratio")
    axes[2].axhline(0, color="black", linewidth=0.5)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "dpo_implementation.png")
    plt.savefig(out_path, dpi=110)
    plt.close()

    print(f"Final accuracy: {accuracies[-1]:.4f}")
    print(f"Final margin:   {margin_hist[-1]:+.3f}")
    print(f"Saved plot:     {out_path}")


if __name__ == "__main__":
    train_dpo()
