"""Shared toy reasoning stack for Phase 6 — multi-step addition with a scratchpad.

Every Phase-6 project needs the same ingredients: a task where *thinking in tokens*
(chain-of-thought) beats answering directly, whose answers and whose intermediate
steps can both be checked exactly. We use the sum of four numbers:

    prompt   "3+7+9+2="
    direct    "21;"                          answer straight away (hard for a char model)
    CoT       "3+7=10,10+9=19,19+2=21;"      running partial sums (easy 2-number adds)

Direct answering forces the model to compute a 4-way sum in one shot (~3% accuracy);
the chain-of-thought decomposes it into three easy 2-number additions (~40%+). And
because every step "A+B=C" can be checked (C == A+B), the same task supports process
reward models, tree search, and RLVR.

Imported by projects 37-42 via sys.path. Reuses the GPT skeleton from project 08.
"""

import random
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "08-nanogpt-reproduction"))
sys.path.insert(0, str(HERE.parent / "01-train-a-bpe-from-scratch"))
from model import Config, GPT   # noqa: E402

K = 4                            # number of operands
LO, HI = 0, 20                  # operand range
CHARS = "0123456789+=,;"
STOI = {c: i for i, c in enumerate(CHARS)}
ITOS = {i: c for c, i in STOI.items()}
VOCAB = len(CHARS)
BLOCK = 44
END = STOI[";"]


def encode(s):
    return [STOI[c] for c in s]


def decode(ids):
    return "".join(ITOS[int(i)] for i in ids)


def new_model(n_layer=4, n_head=4, n_embd=128, seed=0):
    torch.manual_seed(seed); torch.set_num_threads(12)
    return GPT(Config(vocab_size=VOCAB, n_layer=n_layer, n_head=n_head,
                      n_embd=n_embd, block_size=BLOCK))


# ------------------------------------------------------------------ task / formats
def problem(rng):
    xs = [rng.randint(LO, HI) for _ in range(K)]
    return xs, sum(xs)


def prompt_str(xs):
    return "+".join(map(str, xs)) + "="


def direct_str(xs, ans):
    return f"{ans};"


def cot_str(xs):
    s = xs[0]; steps = []
    for x in xs[1:]:
        steps.append(f"{s}+{x}={s + x}"); s += x
    return ",".join(steps) + ";"


def final_answer(completion):
    """The answer a completion asserts = the last number before the terminator."""
    body = completion.split(";")[0]
    nums = re.findall(r"\d+", body)
    return nums[-1] if nums else ""


def is_correct(xs, ans, completion):
    return final_answer(completion) == str(ans)


def cot_steps(completion):
    """Parse a CoT scratchpad into (a, b, c) triples, one per 'A+B=C' step."""
    out = []
    for part in completion.split(";")[0].split(","):
        m = re.match(r"^(\d+)\+(\d+)=(\d+)$", part)
        if m:
            out.append(tuple(int(g) for g in m.groups()))
    return out


def step_correct(a, b, c):
    return a + b == c


# ------------------------------------------------------------------ SFT
def sft_batch(bs, rng, mode="cot"):
    xs_x, ys, masks = [], [], []
    for _ in range(bs):
        xs, ans = problem(rng)
        comp = direct_str(xs, ans) if mode == "direct" else cot_str(xs)
        s = prompt_str(xs) + comp
        ids = encode(s) + [END] * (BLOCK + 1 - len(s))
        eq = s.index("=")
        m = [1.0 if i >= eq else 0.0 for i in range(BLOCK)]
        xs_x.append(ids[:BLOCK]); ys.append(ids[1:BLOCK + 1]); masks.append(m)
    return torch.tensor(xs_x), torch.tensor(ys), torch.tensor(masks)


def masked_ce(logits, y, mask):
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
    return (loss * mask.reshape(-1)).sum() / mask.sum().clamp(min=1)


def train_sft(model, steps, mode="cot", lr=3e-3, seed=0, opt=None):
    rng = random.Random(seed)
    opt = opt or torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    for _ in range(steps):
        x, y, mask = sft_batch(64, rng, mode=mode)
        logits, _ = model(x)
        loss = masked_ce(logits, y, mask)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


def cot_model(steps=800, seed=0):
    m = new_model(seed=seed)
    train_sft(m, steps, mode="cot", seed=seed)
    return m


# ------------------------------------------------------------------ generation
@torch.no_grad()
def sample_batch(model, xs_list, max_new=None, temperature=1.0):
    """Batched completions, bucketed by prompt length. Returns completion strings."""
    from collections import defaultdict
    max_new = max_new or (BLOCK - 6)
    by_len = defaultdict(list)
    for i, xs in enumerate(xs_list):
        by_len[len(prompt_str(xs))].append(i)
    comps = [None] * len(xs_list)
    for idxs in by_len.values():
        cur = torch.tensor([encode(prompt_str(xs_list[i])) for i in idxs])
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
def accuracy(model, n=300, seed=1234, temperature=0.0):
    rng = random.Random(seed)
    probs = [problem(rng) for _ in range(n)]
    comps = sample_batch(model, [xs for xs, _ in probs], temperature=temperature)
    return sum(is_correct(xs, ans, c) for (xs, ans), c in zip(probs, comps)) / n


# ---------------------------------------------------- sequence log-probs (RL, project 41)
def seq_from(xs, completion):
    p = encode(prompt_str(xs)); c = encode(completion)
    return p + c, [0] * len(p) + [1] * len(c)


class RewardModel(nn.Module):
    """A GPT body with a scalar head, giving a score at every token position.

    Used two ways: an OUTCOME reward model reads the score at the last token (one
    number per solution, project 38); a PROCESS reward model reads the score at each
    step boundary (one number per reasoning step, projects 39-40)."""
    def __init__(self, n_layer=4, n_head=4, n_embd=128):
        super().__init__()
        self.gpt = new_model(n_layer, n_head, n_embd)
        self.head = nn.Linear(n_embd, 1, bias=False)

    def token_scores(self, x):
        g = self.gpt
        h = g.tok(x)
        rope = g._rope_for(x.size(1), x.device) if g.cfg.pos == "rope" else None
        for blk in g.blocks:
            h = blk(h, rope)
        return self.head(g.norm_f(h)).squeeze(-1)           # (B, T)


def pad_batch(seqs):
    """Right-pad a list of id-lists to a (B,T) tensor plus the true lengths."""
    T = max(len(s) for s in seqs)
    x = torch.full((len(seqs), T), END, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
    return x, [len(s) for s in seqs]


def completion_token_logps(model, seqs, masks):
    T = max(len(s) for s in seqs); B = len(seqs)
    x = torch.full((B, T), END, dtype=torch.long); m = torch.zeros(B, T)
    for i, (s, mk) in enumerate(zip(seqs, masks)):
        x[i, :len(s)] = torch.tensor(s); m[i, :len(mk)] = torch.tensor(mk, dtype=torch.float)
    logits, _ = model(x[:, :-1])
    logp = F.log_softmax(logits, dim=-1)
    tok_lp = logp.gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1)
    return tok_lp, m[:, 1:]
