"""A minimal but complete decoder-only GPT — the shared skeleton for Phase 2.

This is nanoGPT with the modern refinements made optional knobs so the later
ablation projects can flip exactly one at a time:

  - norm placement:  pre-norm (default, stable) or post-norm        -> project 09
  - positions:       RoPE (default) or learned absolute             -> project 10
  - attention:       n_kv_heads for MHA / GQA / MQA                 -> project 11
  - MLP:             dense SwiGLU (default) or top-k Mixture-of-Experts -> project 12

Also holds the shared char-level data pipeline and training loop so the ablations
stay tiny. Imported by projects 09, 11, 12, 13 via sys.path.
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 65
    n_layer: int = 4
    n_head: int = 4
    n_kv_heads: int = None          # None => = n_head (MHA); < n_head => GQA/MQA
    n_embd: int = 128
    block_size: int = 128
    norm: str = "pre"               # "pre" | "post"
    pos: str = "rope"               # "rope" | "learned"
    moe: dict = field(default=None)  # e.g. {"n_experts": 8, "top_k": 2}
    dropout: float = 0.0

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_head
        assert self.n_head % self.n_kv_heads == 0, "n_head must be divisible by n_kv_heads"


# ------------------------------------------------------------------ primitives
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d)); self.eps = eps

    def forward(self, x):
        return self.g * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def rope_tables(d_head, T, device, base=10000.0, scale=1.0):
    # scale < 1 compresses positions into the trained range (position interpolation)
    inv = 1.0 / base ** (torch.arange(0, d_head, 2, device=device).float() / d_head)
    ang = torch.outer(torch.arange(T, device=device).float() * scale, inv)
    return torch.cos(ang).repeat(1, 2), torch.sin(ang).repeat(1, 2)   # (T, d_head)


def apply_rope(x, cos, sin):        # x: (B, h, T, d_head)
    d = x.size(-1)
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    rot = torch.cat([-x2, x1], dim=-1)
    return x * cos + rot * sin


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.nh, self.nkv = cfg.n_head, cfg.n_kv_heads
        self.dh = cfg.n_embd // cfg.n_head
        self.pos = cfg.pos
        self.q = nn.Linear(cfg.n_embd, self.nh * self.dh, bias=False)
        self.k = nn.Linear(cfg.n_embd, self.nkv * self.dh, bias=False)
        self.v = nn.Linear(cfg.n_embd, self.nkv * self.dh, bias=False)
        self.o = nn.Linear(self.nh * self.dh, cfg.n_embd, bias=False)

    def forward(self, x, rope):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.nh, self.dh).transpose(1, 2)
        k = self.k(x).view(B, T, self.nkv, self.dh).transpose(1, 2)
        v = self.v(x).view(B, T, self.nkv, self.dh).transpose(1, 2)
        if self.pos == "rope":
            cos, sin = rope
            q = apply_rope(q, cos, sin); k = apply_rope(k, cos, sin)
        if self.nkv != self.nh:
            rep = self.nh // self.nkv
            k = k.repeat_interleave(rep, 1); v = v.repeat_interleave(rep, 1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).contiguous().view(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, d, hidden=None):
        super().__init__()
        hidden = hidden or int(8 / 3 * d)
        self.w = nn.Linear(d, hidden, bias=False)
        self.g = nn.Linear(d, hidden, bias=False)
        self.o = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.o(F.silu(self.g(x)) * self.w(x))


class MoE(nn.Module):
    """Top-k Mixture-of-Experts MLP with a load-balancing auxiliary loss."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.n = cfg.moe["n_experts"]; self.k = cfg.moe["top_k"]
        self.router = nn.Linear(cfg.n_embd, self.n, bias=False)
        self.experts = nn.ModuleList(SwiGLU(cfg.n_embd) for _ in range(self.n))
        self.aux = torch.tensor(0.0)

    def forward(self, x):
        B, T, d = x.shape
        flat = x.reshape(-1, d)
        logits = self.router(flat)                       # (N, n_experts)
        probs = F.softmax(logits, dim=-1)
        topv, topi = probs.topk(self.k, dim=-1)          # (N, k)
        topv = topv / topv.sum(-1, keepdim=True)
        out = torch.zeros_like(flat)
        for slot in range(self.k):
            idx = topi[:, slot]
            for e in range(self.n):
                m = idx == e
                if m.any():
                    out[m] += topv[m, slot:slot + 1] * self.experts[e](flat[m])
        # load-balancing loss (Switch Transformer): fraction routed * mean prob
        frac = torch.zeros(self.n, device=x.device)
        top1 = topi[:, 0]
        for e in range(self.n):
            frac[e] = (top1 == e).float().mean()
        self.aux = self.n * (frac * probs.mean(0)).sum()
        self.frac = frac.detach()
        return out.view(B, T, d)


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.norm = cfg.norm
        self.n1, self.n2 = RMSNorm(cfg.n_embd), RMSNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.mlp = MoE(cfg) if cfg.moe else SwiGLU(cfg.n_embd)

    def forward(self, x, rope):
        if self.norm == "pre":                           # x + f(norm(x))
            x = x + self.attn(self.n1(x), rope)
            x = x + self.mlp(self.n2(x))
        else:                                            # norm(x + f(x))
            x = self.n1(x + self.attn(x, rope))
            x = self.n2(x + self.mlp(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos = nn.Embedding(cfg.block_size, cfg.n_embd) if cfg.pos == "learned" else None
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm_f = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.tok.weight = self.head.weight              # weight tying
        self._rope = None
        self.rope_scale = 1.0                            # <1 => position interpolation
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def _rope_for(self, T, device):
        # keyed on (length, scale) so changing rope_scale rebuilds the tables
        key = (T, self.rope_scale)
        if self._rope is None or self._rope[0] != key:
            tables = rope_tables(self.cfg.n_embd // self.cfg.n_head, T, device,
                                 scale=self.rope_scale)
            self._rope = (key, tables)
        return self._rope[1][0][:T], self._rope[1][1][:T]

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx)
        if self.pos is not None:
            x = x + self.pos(torch.arange(T, device=idx.device))
        rope = self._rope_for(T, idx.device) if self.cfg.pos == "rope" else None
        for blk in self.blocks:
            x = blk(x, rope)
        logits = self.head(self.norm_f(x))
        loss = aux = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            aux = sum(b.mlp.aux for b in self.blocks if isinstance(b.mlp, MoE))
            if isinstance(aux, torch.Tensor):
                loss = loss + 0.01 * aux
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters()) - self.tok.weight.numel()

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            crop = idx[:, -self.cfg.block_size:]
            logits, _ = self(crop)
            logits = logits[:, -1] / temperature
            if top_k:
                v, _ = logits.topk(min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


# ------------------------------------------------------------------ data + train
class CharData:
    def __init__(self, text, block_size, split=0.9):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(split * len(data))
        self.train, self.val = data[:n], data[n:]
        self.block_size = block_size

    def batch(self, split, batch_size, device="cpu"):
        d = self.train if split == "train" else self.val
        ix = torch.randint(len(d) - self.block_size - 1, (batch_size,))
        x = torch.stack([d[i:i + self.block_size] for i in ix])
        y = torch.stack([d[i + 1:i + 1 + self.block_size] for i in ix])
        return x.to(device), y.to(device)

    def encode(self, s):
        return torch.tensor([[self.stoi[c] for c in s]], dtype=torch.long)

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


def cosine_lr(step, total, base_lr, warmup=0, min_lr_frac=0.1):
    if warmup and step < warmup:
        return base_lr * step / warmup
    if step >= total:
        return base_lr * min_lr_frac
    prog = (step - warmup) / max(1, total - warmup)
    return base_lr * (min_lr_frac + (1 - min_lr_frac) * 0.5 * (1 + math.cos(math.pi * prog)))


@torch.no_grad()
def estimate_loss(model, data, batch_size, iters=20, device="cpu"):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(iters):
            x, y = data.batch(split, batch_size, device)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def train_model(model, data, *, steps, batch_size=32, lr=3e-3, warmup=0,
                eval_every=250, device="cpu", clip=1.0, log=True, tag=""):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    curve = []
    import time
    t0 = time.time()
    for step in range(steps + 1):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, steps, lr, warmup)
        if step % eval_every == 0 or step == steps:
            losses = estimate_loss(model, data, batch_size, device=device)
            curve.append((step, losses["train"], losses["val"]))
            if log:
                print(f"  {tag}step {step}/{steps} | train {losses['train']:.3f} | "
                      f"val {losses['val']:.3f} | {step/(time.time()-t0+1e-9):.1f} it/s", flush=True)
        if step == steps:
            break
        x, y = data.batch("train", batch_size, device)
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
    return curve
