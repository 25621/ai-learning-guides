"""Shared Phase-7 tool/agent stack — model, tool-call grammar, orchestrator.

The agent projects (47-50) all use the same pattern:

  * a tiny char-level GPT (project 08's skeleton) emits **segments** ended by
    ';' — e.g. `C:47*82;` (call the calculator), `L:fr;` (look up a fact),
    `A:3854;` (final answer);
  * an **orchestrator** outside the model parses each finished segment,
    executes the tool, and appends the observation (`R:3854;`) to the
    sequence — the model never computes tool results, it reads them;
  * training loss and RL log-probs are **masked to model-emitted tokens
    only**: `Q:`/`R:` text is environment output, not something the policy
    should be trained to predict.

That last point is the multi-turn version of project 29's loss-masking
lesson, and it is what makes the RL in project 50 correct: environment
tokens carry no policy gradient.

Imported by projects 48, 49 and 50 via sys.path.
"""

import os
import sys

import torch
import torch.nn.functional as F

torch.set_num_threads(12)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "08-nanogpt-reproduction"))
from model import GPT, Config  # noqa: E402

# One shared vocabulary for every Phase-7 agent task.
CHARS = ("0123456789abcdefghijklmnopqrstuvwxyz"
         "ABCDEFGHIJKLMNOPQRSTUVWXYZ:;+*-=?(), .\n")
STOI = {c: i for i, c in enumerate(CHARS)}
ITOS = {i: c for c, i in STOI.items()}
VOCAB = len(CHARS)
SEG_END = STOI[";"]


def encode(s):
    return [STOI[c] for c in s]


def decode(ids):
    return "".join(ITOS[i] for i in ids)


def new_model(block=96, n_layer=4, n_head=4, n_embd=128, seed=0):
    torch.manual_seed(seed)
    cfg = Config(vocab_size=VOCAB, n_layer=n_layer, n_head=n_head,
                 n_embd=n_embd, block_size=block)
    return GPT(cfg)


# ------------------------------------------------------------------- SFT

def collate(traces, block):
    """traces: list of (ids, mask) with mask=1 on model-emitted tokens.

    Returns (x, y, loss_mask) where loss_mask selects targets the model is
    responsible for producing (next-token positions whose target is a
    model token)."""
    B = len(traces)
    x = torch.zeros(B, block, dtype=torch.long)
    y = torch.zeros(B, block, dtype=torch.long)
    m = torch.zeros(B, block)
    for i, (ids, mask) in enumerate(traces):
        ids, mask = ids[:block + 1], mask[:block + 1]
        T = len(ids) - 1
        x[i, :T] = torch.tensor(ids[:-1])
        y[i, :T] = torch.tensor(ids[1:])
        m[i, :T] = torch.tensor(mask[1:], dtype=torch.float)
    return x, y, m


def masked_ce(logits, y, mask):
    lp = F.log_softmax(logits, dim=-1)
    nll = -lp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    return (nll * mask).sum() / mask.sum().clamp(min=1.0)


def train_sft(model, sample_traces, steps, *, bs=32, lr=3e-3, block=96,
              log_every=100, tag=""):
    """sample_traces(bs, rng) -> list of (ids, mask). Cosine LR, AdamW."""
    import time
    from model import cosine_lr
    rng = torch.Generator().manual_seed(0)
    import random
    py_rng = random.Random(0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    t0 = time.time()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, steps, lr)
        x, y, m = collate(sample_traces(bs, py_rng), block)
        logits, _ = model(x)
        loss = masked_ce(logits, y, m)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            print(f"  {tag}step {step}/{steps} loss {loss.item():.3f} "
                  f"({step / (time.time() - t0 + 1e-9):.1f} it/s)",
                  flush=True)
    return model


# ------------------------------------------- orchestrated (interactive) rollout

@torch.no_grad()
def run_episodes(model, prompts, step_fn, states, *, max_model_tokens=48,
                 temperature=0.0, block=96, batch_size=64):
    """Run tool-using episodes, batched.

    The model generates until ';' closes a segment; `step_fn(state, seg)`
    executes it and returns (reply_or_None, done). Replies are appended as
    environment tokens (mask 0). An episode also ends when it runs out of
    model-token budget or sequence room — that's a failure the caller reads
    from its `state`.

    Returns (seqs, masks): token ids and model-token masks per episode,
    ready for `episode_token_logps` (RL) or inspection.
    """
    all_seqs, all_masks = [], []
    for lo in range(0, len(prompts), batch_size):
        chunk = prompts[lo:lo + batch_size]
        st = states[lo:lo + batch_size]
        seqs = [encode(p) for p in chunk]
        masks = [[0] * len(s) for s in seqs]
        seg_start = [len(s) for s in seqs]
        done = [False] * len(chunk)
        used = [0] * len(chunk)
        while not all(done):
            T = max(len(s) for s in seqs)
            x = torch.zeros(len(seqs), T, dtype=torch.long)
            for i, s in enumerate(seqs):
                x[i, :len(s)] = torch.tensor(s)
            logits, _ = model(x)
            for i in range(len(seqs)):
                if done[i]:
                    continue
                lg = logits[i, len(seqs[i]) - 1]
                if temperature and temperature > 0:
                    nxt = int(torch.multinomial(
                        F.softmax(lg / temperature, -1), 1))
                else:
                    nxt = int(lg.argmax())
                seqs[i].append(nxt)
                masks[i].append(1)
                used[i] += 1
                if nxt == SEG_END:
                    seg = decode(seqs[i][seg_start[i]:-1])
                    reply, fin = step_fn(st[i], seg)
                    if reply:
                        seqs[i] += encode(reply)
                        masks[i] += [0] * len(encode(reply))
                    seg_start[i] = len(seqs[i])
                    done[i] = fin
                if used[i] >= max_model_tokens or len(seqs[i]) >= block - 1:
                    done[i] = True
        all_seqs += seqs
        all_masks += masks
    return all_seqs, all_masks


# ------------------------------------------------- the QA task (47 and 50)
# Three question kinds. 'calc' needs exact 2-digit multiplication/addition;
# 'lookup' asks for a fact the model cannot know (the value is rolled fresh
# per episode — it exists only in the orchestrator's database); 'compose'
# chains two lookups and an addition. Tools make all three trivial copy
# jobs; without tools, 'calc' is hard and the other two are impossible.

import random  # noqa: E402
import re  # noqa: E402

KINDS = ("calc", "lookup", "compose")
_CALC_RE = re.compile(r"(\d{1,4})([+*])(\d{1,4})")


def _name(rng):
    return "".join(rng.choice("abcdefghijkmnpqrstuvwxyz") for _ in range(2))


def make_question(rng, kind):
    if kind == "calc":
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        op = rng.choice("+*")
        return {"kind": kind, "q": f"Q:{a}{op}{b}=?", "db": {},
                "ans": a + b if op == "+" else a * b}
    if kind == "lookup":
        n, v = _name(rng), rng.randint(10, 99)
        return {"kind": kind, "q": f"Q:val({n})?", "db": {n: v}, "ans": v}
    n1 = _name(rng)
    n2 = _name(rng)
    while n2 == n1:
        n2 = _name(rng)
    v1, v2 = rng.randint(10, 99), rng.randint(10, 99)
    return {"kind": kind, "q": f"Q:val({n1})+val({n2})?",
            "db": {n1: v1, n2: v2}, "ans": v1 + v2}


def expert_pieces(qd):
    """The tool-using gold trace as (text, is_model_token) pieces."""
    q, db, ans = qd["q"], qd["db"], qd["ans"]
    pieces = [(q, 0)]
    if qd["kind"] == "calc":
        expr = q[2:-2]
        pieces += [(f"C:{expr};", 1), (f"R:{ans};", 0)]
    elif qd["kind"] == "lookup":
        (n, v), = db.items()
        pieces += [(f"L:{n};", 1), (f"R:{v};", 0)]
    else:
        (n1, v1), (n2, v2) = db.items()
        pieces += [(f"L:{n1};", 1), (f"R:{v1};", 0),
                   (f"L:{n2};", 1), (f"R:{v2};", 0),
                   (f"C:{v1}+{v2};", 1), (f"R:{ans};", 0)]
    pieces.append((f"A:{ans};", 1))
    return pieces


def direct_pieces(qd):
    """No-tools gold trace: just blurt the final answer."""
    return [(qd["q"], 0), (f"A:{qd['ans']};", 1)]


def pieces_to_trace(pieces):
    ids, mask = [], []
    for text, is_model in pieces:
        t = encode(text)
        ids += t
        mask += [is_model] * len(t)
    return ids, mask


def qa_traces(kind_mix=KINDS, expert=True):
    def sample(bs, rng):
        out = []
        for _ in range(bs):
            qd = make_question(rng, rng.choice(kind_mix))
            pieces = expert_pieces(qd) if expert else direct_pieces(qd)
            out.append(pieces_to_trace(pieces))
        return out
    return sample


def qa_step_fn(state, seg):
    """Orchestrator for the QA task. Mutates state; returns (reply, done)."""
    if seg.startswith("A:"):
        state["answer"] = seg[2:]
        return None, True
    if seg.startswith("C:"):
        state["tool_calls"] += 1
        m = _CALC_RE.fullmatch(seg[2:])
        if m:
            a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
            return f"R:{a + b if op == '+' else a * b};", False
    elif seg.startswith("L:"):
        state["tool_calls"] += 1
        if seg[2:] in state["db"]:
            return f"R:{state['db'][seg[2:]]};", False
    state["malformed"] += 1
    return "R:?;", False


def qa_eval(model, kind, n, *, seed=1234, temperature=0.0, block=96):
    """Run n fresh questions of one kind; returns metrics + sample states."""
    rng = random.Random(seed)
    qds = [make_question(rng, kind) for _ in range(n)]
    states = [{"db": qd["db"], "ans": qd["ans"], "answer": None,
               "malformed": 0, "tool_calls": 0} for qd in qds]
    seqs, masks = run_episodes(model, [qd["q"] for qd in qds], qa_step_fn,
                               states, block=block, temperature=temperature)
    acc = sum(s["answer"] == str(s["ans"]) for s in states) / n
    malformed = sum(s["malformed"] > 0 for s in states) / n
    toks = sum(sum(m) for m in masks) / n
    return {"acc": acc, "malformed": malformed, "model_tokens": toks,
            "states": states, "seqs": seqs, "masks": masks, "qds": qds}


def episode_token_logps(model, seqs, masks, pad=SEG_END):
    """Per-token log-probs over model-emitted tokens (right-padded rectangle).

    Returns (tok_lp (B,T-1), mask (B,T-1)) — the RL workhorse; environment
    tokens are excluded by the mask so they carry no gradient."""
    T = max(len(s) for s in seqs)
    B = len(seqs)
    x = torch.full((B, T), pad, dtype=torch.long)
    m = torch.zeros(B, T)
    for i, (s, mk) in enumerate(zip(seqs, masks)):
        x[i, :len(s)] = torch.tensor(s)
        m[i, :len(mk)] = torch.tensor(mk, dtype=torch.float)
    logits, _ = model(x[:, :-1])
    logp = F.log_softmax(logits, dim=-1)
    tok_lp = logp.gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1)
    return tok_lp, m[:, 1:]
