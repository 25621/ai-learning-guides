"""A small MNIST classifier used only as a feature extractor.

True FID uses InceptionV3 features, which are meaningless for 28x28 digits.
The standard trick for MNIST-scale studies is to train a small classifier and
compute the same Frechet distance in ITS feature space — the metric is then
sensitive to digit identity and stroke style rather than ImageNet textures.

Run standalone to train and cache the network:
    python feature_net.py --data-dir data
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class FeatureNet(nn.Module):
    """Three conv blocks -> global average pool -> 64-d feature -> 10 logits."""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.SiLU(),   # 28 -> 14
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),  # 14 -> 7
            nn.Conv2d(64, 64, 3, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),                 # -> (B, 64)
        )
        self.head = nn.Linear(64, 10)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


def train_feature_net(data_dir: str, ckpt_path: Path, device: str = "cpu", epochs: int = 4):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=tf)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=512)

    torch.manual_seed(0)
    net = FeatureNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    for epoch in range(epochs):
        net.train()
        for x, y in train_loader:
            loss = F.cross_entropy(net(x.to(device)), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        correct = sum(
            (net(x.to(device)).argmax(1) == y.to(device)).sum().item()
            for x, y in test_loader
        )
        print(f"epoch {epoch + 1}: test accuracy {correct / len(test_ds):.4f}")

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), ckpt_path)
    print(f"saved {ckpt_path}")
    return net


def load_feature_net(ckpt_path: Path, data_dir: str, device: str = "cpu") -> FeatureNet:
    net = FeatureNet().to(device)
    if not ckpt_path.exists():
        return train_feature_net(data_dir, ckpt_path, device)
    net.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    net.eval()
    return net


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="checkpoints/feature_net.pt")
    args = ap.parse_args()
    train_feature_net(args.data_dir, Path(args.out))
