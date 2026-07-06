"""A small MNIST CNN used across the Phase-10 evaluation projects as an
automatic 'reader' of generated digits.

Whenever a project needs to ask "what digit did the generator actually draw?"
it runs this classifier. Trained on real MNIST it reaches ~99% test accuracy in
about 30 seconds on CPU, so its verdict on a clean generated digit is reliable
enough to use as ground truth. Projects 58 (caption ablation), 63 (GenEval) and
65 (text-rendering) all import it via sys.path.
"""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MnistClassifier(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 7
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_or_train(data_dir, out, steps=700, seed=0):
    """Load a cached classifier, or train one (~30s) and cache it."""
    out = Path(out)
    model = MnistClassifier()
    if out.exists():
        model.load_state_dict(torch.load(out))
        return model.eval()

    torch.manual_seed(seed)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    ds = datasets.MNIST(str(data_dir), train=True, download=True, transform=tf)
    loader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=2, drop_last=True)
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
def read_digits(model, imgs):
    """Return (pred_labels, confidences) for a batch of images in [-1, 1]."""
    probs = torch.softmax(model(imgs), dim=1)
    conf, pred = probs.max(dim=1)
    return pred, conf
