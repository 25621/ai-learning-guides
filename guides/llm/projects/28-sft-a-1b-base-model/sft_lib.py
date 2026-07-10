"""Shared toy post-training stack for Phase 5 — a verifiable arithmetic task.

Every Phase-5 project (SFT, loss masking, LoRA, reward models, PPO, DPO, GRPO,
reward-hacking) needs the same three things: a small model, an instruction task
whose answers can be *checked*, and helpers to generate and score completions.
This module provides all of it around a char-level addition task:

    prompt      "37+8="            (the "user turn")
    completion  "45;"              (the "assistant turn", ';' is the end token)

Because a verifier (`is_correct`) can check any answer exactly, this one task
supports the whole phase: SFT demonstrations, preference pairs (correct vs wrong),
a reward model, and RLVR/GRPO. Numbers are 0-50 so a 4-layer char GPT can learn it
to ~90% with enough steps — leaving a partial SFT model for RL to improve.

Imported by projects 29-35 via sys.path. Reuses the GPT skeleton from project 08.
"""

import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "08-nanogpt-reproduction"))
sys.path.insert(0, str(HERE.parent / "01-train-a-bpe-from-scratch"))
from model import Config, GPT   # noqa: E402

MAXN = 50                       # operands in [0, MAXN]
CHARS = "0123456789+=;"
STOI = {c: i for i, c in enumerate(CHARS)}
ITOS = {i: c for c, i in STOI.items()}
VOCAB = len(CHARS)
BLOCK = 16
END = STOI[";"]


def encode(s):
    return [STOI[c] for c in s]


def decode(ids):
    return "".join(ITOS[int(i)] for i in ids)


def new_model(n_layer=4, n_head=4, n_embd=128, seed=0):
    torch.manual_seed(seed); torch.set_num_threads(12)
    return GPT(Config(vocab_size=VOCAB, n_layer=n_layer, n_head=n_head,
                      n_embd=n_embd, block_size=BLOCK))


# ------------------------------------------------------------------ task / data
def sample_problem(rng):
    a, b = rng.randint(0, MAXN), rng.randint(0, MAXN)
    return a, b, a + b


def prompt_str(a, b):
    return f"{a}+{b}="


def is_correct(a, b, completion):
    """Verifier: does the generated completion state the right sum?"""
    return completion.split(";")[0] == str(a + b)


def sft_batch(bs, rng, corrupt=False, full_loss=False):
    """A padded SFT batch. Loss masks the prompt (assistant-only) unless full_loss.

    corrupt=True writes a wrong answer (a noisy 'web' pretraining corpus).
    """
    xs, ys, masks = [], [], []
    for _ in range(bs):
        a, b, c = sample_problem(rng)
        ans = rng.randint(0, 2 * MAXN) if corrupt else c
        s = f"{a}+{b}={ans};"
        ids = encode(s) + [END] * (BLOCK + 1 - len(s))
        eq = s.index("=")
        # target at position i is s[i+1]; it is an answer token when i >= eq
        m = [1.0] * BLOCK if full_loss else [1.0 if i >= eq else 0.0 for i in range(BLOCK)]
        xs.append(ids[:BLOCK]); ys.append(ids[1:BLOCK + 1]); masks.append(m)
    return (torch.tensor(xs), torch.tensor(ys), torch.tensor(masks))


def masked_ce(logits, y, mask):
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                           reduction="none")
    return (loss * mask.reshape(-1)).sum() / mask.sum().clamp(min=1)


def sft_model(steps=500, seed=0, lr=3e-3):
    """A fresh model supervised-fine-tuned to a *partial* accuracy — the shared
    starting policy for the RL/preference projects (they improve on it)."""
    m = new_model(seed=seed)
    train_sft(m, steps=steps, corrupt=False, full_loss=False, seed=seed, lr=lr)
    return m


def train_sft(model, steps, lr=3e-3, corrupt=False, full_loss=False, seed=0,
              opt=None, log_every=0):
    rng = random.Random(seed)
    opt = opt or torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                                   weight_decay=0.1)
    for step in range(steps):
        x, y, mask = sft_batch(64, rng, corrupt=corrupt, full_loss=full_loss)
        logits, _ = model(x)
        loss = masked_ce(logits, y, mask)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if log_every and step % log_every == 0:
            print(f"  sft step {step}/{steps} loss {loss.item():.3f}", flush=True)
    return model


# ------------------------------------------------------------------ generation
@torch.no_grad()
def generate(model, prompts, max_new=4, temperature=0.0, rng=None):
    """Greedy (temperature 0) or sampled completions for a list of (a,b) prompts.

    Returns list of completion strings (without the prompt)."""
    outs = []
    for a, b in prompts:
        ids = encode(prompt_str(a, b))
        start = len(ids)
        for _ in range(max_new):
            x = torch.tensor([ids[-BLOCK:]])
            logits, _ = model(x)
            logits = logits[0, -1]
            if temperature and temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                nxt = int(torch.multinomial(probs, 1))
            else:
                nxt = int(logits.argmax())
            ids.append(nxt)
            if nxt == END:
                break
        outs.append(decode(ids[start:]))
    return outs


@torch.no_grad()
def sample_batch(model, ab_list, max_new=4, temperature=1.0):
    """Batched completions for many prompts, bucketed by prompt length so each
    batch is a clean rectangle (no padding artifacts). Returns strings aligned to
    ab_list. This is the fast path used by accuracy() and the RL rollouts."""
    from collections import defaultdict
    by_len = defaultdict(list)
    for i, (a, b) in enumerate(ab_list):
        by_len[len(prompt_str(a, b))].append(i)
    comps = [None] * len(ab_list)
    for idxs in by_len.values():
        cur = torch.tensor([encode(prompt_str(*ab_list[i])) for i in idxs])
        B = len(idxs); gen = [[] for _ in range(B)]; done = torch.zeros(B, dtype=torch.bool)
        for _ in range(max_new):
            logits, _ = model(cur[:, -BLOCK:])
            logits = logits[:, -1]
            if temperature and temperature > 0:
                nxt = torch.multinomial(F.softmax(logits / temperature, -1), 1).squeeze(-1)
            else:
                nxt = logits.argmax(-1)
            for j in range(B):
                if not done[j]:
                    t = int(nxt[j]); gen[j].append(t)
                    if t == END:
                        done[j] = True
            cur = torch.cat([cur, nxt.unsqueeze(1)], 1)
            if done.all():
                break
        for j, i in enumerate(idxs):
            comps[i] = decode(gen[j])
    return comps


@torch.no_grad()
def accuracy(model, n=400, seed=1234, temperature=0.0):
    rng = random.Random(seed)
    probs = [(rng.randint(0, MAXN), rng.randint(0, MAXN)) for _ in range(n)]
    gens = sample_batch(model, probs, temperature=temperature)
    return sum(is_correct(a, b, g) for (a, b), g in zip(probs, gens)) / n


# ---------------------------------------------------- sequence log-probs (RL/DPO)
def seq_from(a, b, completion):
    """Token ids for a full prompt+completion, plus a mask over completion tokens."""
    p = encode(prompt_str(a, b))
    c = encode(completion)
    ids = p + c
    mask = [0] * len(p) + [1] * len(c)
    return ids, mask


def completion_token_logps(model, seqs, masks):
    """Per-token log-probs over the completion, right-padded to a rectangle.

    seqs: list[list[int]] (prompt+completion), masks: list[list[int]] (1 on completion).
    Returns (tok_lp (B,T-1), mask (B,T-1)). The workhorse for DPO/PPO/GRPO.
    """
    T = max(len(s) for s in seqs)
    B = len(seqs)
    x = torch.full((B, T), END, dtype=torch.long)
    m = torch.zeros(B, T)
    for i, (s, mk) in enumerate(zip(seqs, masks)):
        x[i, :len(s)] = torch.tensor(s)
        m[i, :len(mk)] = torch.tensor(mk, dtype=torch.float)
    logits, _ = model(x[:, :-1])
    logp = F.log_softmax(logits, dim=-1)
    tok_lp = logp.gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1)   # (B, T-1)
    return tok_lp, m[:, 1:]


def completion_logprobs(model, seqs, masks):
    """Sum of per-token completion log-probs — a (B,) tensor."""
    tok_lp, m = completion_token_logps(model, seqs, masks)
    return (tok_lp * m).sum(-1)
