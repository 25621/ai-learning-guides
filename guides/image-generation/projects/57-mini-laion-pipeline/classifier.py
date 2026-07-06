"""A tiny MNIST CNN that stands in for the big pretrained models a real data
pipeline leans on.

In a production LAION-style pipeline two learned models do the heavy lifting:
a CLIP model scores how well a caption matches an image, and a VLM writes new
captions. We have neither on a CPU in ten minutes, so we train one small digit
classifier and use it for both jobs:

    - as the "CLIP" filter: alignment(image, caption) := P(classifier says the
      caption's claimed digit | image). A caption that names the wrong digit
      gets a low score, exactly like a low CLIP similarity.
    - as the "VLM" recaptioner: its top prediction plus a couple of cheap image
      statistics (stroke boldness, centering) compose a richer caption.

It is a faithful *shape* of the real pipeline; only the models are toy-sized.
"""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent


class MnistCNN(nn.Module):
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


def train_classifier(data_dir, out=None, steps=600, seed=0):
    """Train (or load) the stand-in scorer. ~30s on CPU."""
    out = Path(out or HERE / "checkpoints/classifier.pt")
    if out.exists():
        model = MnistCNN()
        model.load_state_dict(torch.load(out))
        return model.eval()

    torch.manual_seed(seed)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    ds = datasets.MNIST(str(data_dir), train=True, download=True, transform=tf)
    loader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=2, drop_last=True)
    model = MnistCNN()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    it = iter(loader)
    for step in range(steps):
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
def class_probs(model, x):
    """Softmax probabilities over the 10 digits for a batch in [-1, 1]."""
    return torch.softmax(model(x), dim=1)
