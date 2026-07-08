"""A small MNIST VQ tokenizer shared by projects 16 and 17.

Turns each 28x28 digit into a 7x7 grid of 49 discrete tokens from a 128-entry
codebook (EMA + dead-code re-init, so the codebook doesn't collapse). Once an
image is 49 tokens, generating an image is just generating a short sequence of
symbols — which is what the transformer projects do.

We use MNIST rather than CIFAR here purely for legibility: a tiny transformer on
a CPU produces *recognizable* generated digits, so the autoregressive-vs-parallel
comparison is easy to see. The mechanism is identical for CIFAR VQ-GAN tokens.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "12-vq-vae-on-cifar-10"))
from vq_lib import ResBlock, VectorQuantizerEMA, perplexity  # noqa: E402

K, D, GRID = 128, 16, 7
SEQ = GRID * GRID    # 49


class Tokenizer(nn.Module):
    def __init__(self, ch=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, ch, 4, 2, 1), nn.ReLU(),     # 28 -> 14
            nn.Conv2d(ch, ch * 2, 4, 2, 1),           # 14 -> 7
            ResBlock(ch * 2), ResBlock(ch * 2),
            nn.Conv2d(ch * 2, D, 1))
        self.vq = VectorQuantizerEMA(K=K, D=D, reinit=True, dead_after=128)
        self.dec = nn.Sequential(
            nn.Conv2d(D, ch * 2, 1),
            ResBlock(ch * 2), ResBlock(ch * 2), nn.ReLU(),
            nn.ConvTranspose2d(ch * 2, ch, 4, 2, 1), nn.ReLU(),   # 7 -> 14
            nn.ConvTranspose2d(ch, 1, 4, 2, 1), nn.Sigmoid())     # 14 -> 28

    def forward(self, x):
        z_q, vq_loss, idx = self.vq(self.enc(x))
        return self.dec(z_q), vq_loss, idx

    @torch.no_grad()
    def encode_indices(self, x):
        _, _, idx = self.vq(self.enc(x))
        return idx.reshape(x.size(0), -1)                          # (B, 49)

    @torch.no_grad()
    def decode_indices(self, idx):
        z_q = F.embedding(idx.view(-1, GRID, GRID), self.vq.embed).permute(0, 3, 1, 2)
        return self.dec(z_q)


def mnist_loader(data_dir, bs=128, train=True):
    tf = transforms.Compose([transforms.ToTensor()])
    ds = datasets.MNIST(str(data_dir), train=train, download=True, transform=tf)
    return DataLoader(ds, batch_size=bs, shuffle=train, num_workers=0, drop_last=train)


def load_or_train_tokenizer(data_dir, out, steps=1500, lr=2e-4):
    out = Path(out)
    model = Tokenizer()
    if out.exists():
        model.load_state_dict(torch.load(out)["model"]); return model.eval()
    import time
    loader = iter(mnist_loader(data_dir, 128, True))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    for step in range(1, steps + 1):
        try:
            x, _ = next(loader)
        except StopIteration:
            loader = iter(mnist_loader(data_dir, 128, True)); x, _ = next(loader)
        xhat, vq_loss, idx = model(x)
        loss = F.binary_cross_entropy(xhat, x) + vq_loss
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0:
            p, _ = perplexity(idx, K)
            print(f"  [tokenizer] step {step}/{steps} | recon {F.binary_cross_entropy(xhat,x).item():.4f} "
                  f"| perplexity {p:.0f}/{K} | {step/(time.time()-t0):.1f} it/s", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, out)
    return model.eval()


@torch.no_grad()
def encode_all(model, data_dir, n=20000):
    """Encode the first n MNIST train images to token sequences (n, 49)."""
    loader = mnist_loader(data_dir, 256, train=True)
    toks = []
    for x, _ in loader:
        toks.append(model.encode_indices(x))
        if sum(t.size(0) for t in toks) >= n:
            break
    return torch.cat(toks)[:n]
