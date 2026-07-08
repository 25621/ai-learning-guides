"""A small CIFAR-10 classifier, used only to *read* what class a generated
image looks like — the automatic judge for "did conditioning actually work?"

Reaches ~65-70% test accuracy in under a minute on CPU, which is more than
enough to tell whether a generator is hitting its requested class far above
the 10% random-guess floor.
"""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from PIL import Image


class CIFAR10Npz(Dataset):
    def __init__(self, npz_path, transform):
        arr = np.load(npz_path)
        self.images, self.labels = arr["images"], arr["labels"]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.transform(Image.fromarray(self.images[i])), int(self.labels[i])


class CifarClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 8
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 4
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128), nn.ReLU(), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_or_train(data_dir, out, steps=600, seed=0):
    out = Path(out)
    model = CifarClassifier()
    if out.exists():
        model.load_state_dict(torch.load(out))
        return model.eval()

    torch.manual_seed(seed)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    ds = CIFAR10Npz(Path(data_dir) / "cifar10_train.npz", tf)
    loader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=0, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    it = iter(loader)
    for _ in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader); x, y = next(it)
        loss = nn.functional.cross_entropy(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    return model.eval()


@torch.no_grad()
def read_classes(model, imgs):
    """imgs in [-1, 1] -> (pred_labels, confidences)."""
    probs = torch.softmax(model(imgs), dim=1)
    conf, pred = probs.max(dim=1)
    return pred, conf
