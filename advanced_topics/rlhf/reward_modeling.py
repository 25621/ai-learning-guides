"""Train a reward model from preference data.

Concept: humans rank two responses (chosen vs rejected). The reward model
learns to assign a higher scalar score to the chosen response. We use the
classic Bradley-Terry loss used in InstructGPT-style RLHF.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def generate_preference_data(num_samples=2000, dim=16, seed=0):
    """Build a synthetic preference dataset.

    Each prompt has two candidate responses (embeddings). A hidden "true"
    reward vector decides which response is preferred. This mimics how a
    human rater would prefer the response that better satisfies the prompt.
    """
    g = torch.Generator().manual_seed(seed)
    prompts = torch.randn(num_samples, dim, generator=g)
    a = torch.randn(num_samples, dim, generator=g)
    b = torch.randn(num_samples, dim, generator=g)

    true_reward = torch.randn(dim, generator=g)
    score_a = (a * true_reward).sum(dim=1)
    score_b = (b * true_reward).sum(dim=1)

    chosen = torch.where((score_a > score_b).unsqueeze(1), a, b)
    rejected = torch.where((score_a > score_b).unsqueeze(1), b, a)
    return prompts, chosen, rejected


class RewardModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_reward_model():
    torch.manual_seed(0)
    dim = 16
    _, chosen, rejected = generate_preference_data(dim=dim)

    model = RewardModel(dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    losses, accuracies = [], []
    num_epochs = 200
    for epoch in range(num_epochs):
        r_chosen = model(chosen)
        r_rejected = model(rejected)

        # Bradley-Terry loss: maximize P(chosen > rejected)
        loss = -torch.nn.functional.logsigmoid(r_chosen - r_rejected).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (r_chosen > r_rejected).float().mean().item()
        losses.append(loss.item())
        accuracies.append(acc)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d} | loss={loss.item():.4f} | acc={acc:.4f}")

    torch.save(model.state_dict(), os.path.join(OUT_DIR, "reward_model.pt"))

    with torch.no_grad():
        r_chosen = model(chosen).numpy()
        r_rejected = model(rejected).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(losses, color="#1f77b4")
    axes[0].set_title("Bradley-Terry Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)

    axes[1].plot(accuracies, color="#2ca02c")
    axes[1].set_title("Preference Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Fraction chosen > rejected")
    axes[1].set_ylim(0.4, 1.02)
    axes[1].grid(alpha=0.3)

    axes[2].hist(r_chosen, bins=40, alpha=0.6, label="chosen", color="#2ca02c")
    axes[2].hist(r_rejected, bins=40, alpha=0.6, label="rejected", color="#d62728")
    axes[2].set_title("Learned Reward Distribution")
    axes[2].set_xlabel("Reward score")
    axes[2].set_ylabel("Count")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "reward_modeling.png")
    plt.savefig(out_path, dpi=110)
    plt.close()

    print(f"Final accuracy: {accuracies[-1]:.4f}")
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    train_reward_model()
