"""KV-cached inference for the Phase-2 GPT — the shared inference stack for Phase 9.

Project 08's `model.py` trains fine but decodes naively: to produce token t it
re-runs the whole prefix through every layer, recomputing keys and values it
already computed at step t-1. This module adds the fix — a KV cache — *without
touching the trained weights*. Caching is a pure inference-time change: the
functions below run the very same `nn.Linear` modules, they just remember K and V.

Exports (imported by projects 61, 62 and 64 via sys.path):

  KVCache          preallocated per-layer key/value store
  forward_cached   GPT forward that reads/writes a cache and accepts a position offset
  generate_naive   recompute-the-whole-prefix decoding (the baseline)
  generate_cached  prefill once, then one-token steps against the cache
  load_or_train    the shared ~1.8M-param tiny-Shakespeare model, cached to checkpoints/
"""

import os
import sys
import urllib.request

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "08-nanogpt-reproduction"))

from model import Config, GPT, CharData, apply_rope, rope_tables, train_model  # noqa: E402

DATA = os.path.join(HERE, "data")
CKPT = os.path.join(HERE, "checkpoints")
CORPUS_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
              "master/data/tinyshakespeare/input.txt")


# --------------------------------------------------------------------- the cache
class KVCache:
    """Per-layer key/value store, preallocated to `max_len`.

    Shape per layer: (B, n_kv_heads, max_len, d_head) for K and for V. Real
    engines allocate exactly like this — one contiguous slab per sequence sized
    to the longest generation it might do — which is precisely the waste that
    PagedAttention removes (see project 61).
    """

    def __init__(self, cfg: Config, batch_size, max_len, device="cpu", dtype=torch.float32):
        d_head = cfg.n_embd // cfg.n_head
        shape = (batch_size, cfg.n_kv_heads, max_len, d_head)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layer)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layer)]
        self.max_len = max_len
        self.t = 0                      # tokens currently cached

    def append(self, layer, k, v):
        """Write this step's K/V at the current position; return the full history."""
        T = k.size(2)
        self.k[layer][:, :, self.t:self.t + T] = k
        self.v[layer][:, :, self.t:self.t + T] = v
        end = self.t + T
        return self.k[layer][:, :, :end], self.v[layer][:, :, :end]

    def advance(self, T):
        self.t += T

    def bytes(self):
        """Memory actually reserved (not just used) — 2 x layers x B x n_kv x max_len x d_head."""
        n = sum(t.numel() for t in self.k) + sum(t.numel() for t in self.v)
        return n * self.k[0].element_size()


# ------------------------------------------------------------------ cached forward
def attn_cached(attn, x, cos, sin, cache, layer):
    """One attention layer, reading and writing the cache.

    Identical math to `model.Attention.forward` except that K and V for earlier
    tokens come from the cache instead of being recomputed, and the causal mask
    has to account for the cached prefix sitting to the left of the new tokens.
    """
    B, T, _ = x.shape
    q = attn.q(x).view(B, T, attn.nh, attn.dh).transpose(1, 2)
    k = attn.k(x).view(B, T, attn.nkv, attn.dh).transpose(1, 2)
    v = attn.v(x).view(B, T, attn.nkv, attn.dh).transpose(1, 2)
    if attn.pos == "rope":
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

    k, v = cache.append(layer, k, v)                     # (B, n_kv, t_past+T, d_head)

    if attn.nkv != attn.nh:                              # GQA: broadcast kv heads
        rep = attn.nh // attn.nkv
        k = k.repeat_interleave(rep, 1)
        v = v.repeat_interleave(rep, 1)

    if T == 1:
        # A single new query attends to everything cached — no mask needed.
        y = F.scaled_dot_product_attention(q, k, v)
    else:
        # Query i sits at absolute position p0+i and may see keys j <= p0+i.
        p0 = k.size(2) - T
        mask = torch.ones(T, k.size(2), dtype=torch.bool, device=x.device).tril(diagonal=p0)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return attn.o(y.transpose(1, 2).contiguous().view(B, T, -1))


@torch.no_grad()
def forward_cached(model: GPT, idx, cache: KVCache, pos_offset=None):
    """Run `idx` (B, T) through the model, appending to `cache`.

    `pos_offset` is where these tokens sit in the sequence — it selects the RoPE
    angles. Getting this wrong is the classic KV-cache bug: the cache works, the
    speedup is real, and the model quietly generates gibberish because every
    decoded token thinks it is at position 0.
    """
    B, T = idx.shape
    p0 = cache.t if pos_offset is None else pos_offset
    x = model.tok(idx)
    if model.pos is not None:                            # learned absolute positions
        x = x + model.pos(torch.arange(p0, p0 + T, device=idx.device))

    cos = sin = None
    if model.cfg.pos == "rope":
        d_head = model.cfg.n_embd // model.cfg.n_head
        full_cos, full_sin = rope_tables(d_head, cache.max_len, idx.device,
                                         scale=model.rope_scale)
        cos, sin = full_cos[p0:p0 + T], full_sin[p0:p0 + T]

    for li, blk in enumerate(model.blocks):
        if blk.norm == "pre":
            x = x + attn_cached(blk.attn, blk.n1(x), cos, sin, cache, li)
            x = x + blk.mlp(blk.n2(x))
        else:
            x = blk.n1(x + attn_cached(blk.attn, x, cos, sin, cache, li))
            x = blk.n2(x + blk.mlp(x))
    cache.advance(T)
    return model.head(model.norm_f(x))


# ---------------------------------------------------------------------- decoding
def _pick(logits, temperature, top_k, generator=None):
    if temperature == 0:                                 # greedy
        return logits.argmax(-1, keepdim=True)
    logits = logits / temperature
    if top_k:
        v, _ = logits.topk(min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < v[:, [-1]], -float("inf"))
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1, generator=generator)


@torch.no_grad()
def generate_naive(model, idx, max_new_tokens, temperature=0.0, top_k=40, generator=None):
    """The baseline: every step re-runs the *entire* prefix through the model."""
    for _ in range(max_new_tokens):
        crop = idx[:, -model.cfg.block_size:]
        logits, _ = model(crop)
        idx = torch.cat([idx, _pick(logits[:, -1], temperature, top_k, generator)], dim=1)
    return idx


@torch.no_grad()
def generate_cached(model, idx, max_new_tokens, temperature=0.0, top_k=40,
                    generator=None, cache=None):
    """Prefill the prompt in one pass, then feed back one token at a time."""
    B, T = idx.shape
    if cache is None:
        cache = KVCache(model.cfg, B, T + max_new_tokens, idx.device)
    logits = forward_cached(model, idx, cache)           # prefill
    nxt = _pick(logits[:, -1], temperature, top_k, generator)
    idx = torch.cat([idx, nxt], dim=1)
    for _ in range(max_new_tokens - 1):
        logits = forward_cached(model, nxt, cache)       # decode: T=1 forever
        nxt = _pick(logits[:, -1], temperature, top_k, generator)
        idx = torch.cat([idx, nxt], dim=1)
    return idx


# ------------------------------------------------------------------ shared model
def corpus():
    os.makedirs(DATA, exist_ok=True)
    p = os.path.join(DATA, "shakespeare.txt")
    if not os.path.exists(p):
        urllib.request.urlretrieve(CORPUS_URL, p)
    return open(p).read()


def make_config(vocab_size):
    # Small enough to train in ~3 min on CPU, big enough that a 512-token context
    # makes the quadratic cost of naive decoding painfully visible.
    return Config(vocab_size=vocab_size, n_embd=192, n_layer=4, n_head=6,
                  block_size=512, pos="rope")


def load_or_train(steps=600, device="cpu", quiet=False):
    """The model every Phase-9 serving project is built on (trained once, then cached)."""
    os.makedirs(CKPT, exist_ok=True)
    data = CharData(corpus(), block_size=256)
    cfg = make_config(data.vocab_size)
    model = GPT(cfg).to(device)
    path = os.path.join(CKPT, "gpt.pt")
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
    else:
        if not quiet:
            print(f"training the shared {model.num_params()/1e6:.2f}M-param model "
                  f"({steps} steps, ~3 min)...", flush=True)
        train_model(model, data, steps=steps, batch_size=16, lr=3e-3,
                    warmup=50, eval_every=200, device=device, log=not quiet)
        torch.save(model.state_dict(), path)
    model.eval()
    return model, data
