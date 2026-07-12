"""A miniature vLLM: paged KV blocks + continuous batching, in ~250 lines.

vLLM needs a CUDA GPU and is not installed on this CPU-only box, so instead of
driving someone else's engine we build the two ideas that make it fast, on top of
the KV cache from project 58. That is the better trade: an engine you can read is
worth more than a server you can only curl.

  PagedAttention      the KV cache is a pool of fixed-size blocks. A sequence
                      holds a *block table* (a list of block ids), not one
                      contiguous slab, so nobody has to reserve room for the
                      longest generation they might ever do.

  Continuous batching the scheduler runs one decode step at a time and re-decides
                      the batch at *every step*: finished sequences leave
                      immediately and waiting ones take their place, instead of
                      the whole batch waiting for its slowest member.

  Prefix caching      (used by project 62) blocks whose contents are identical
                      are shared between sequences instead of recomputed.

Imported by project 62.
"""

import hashlib
import os
import sys
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "58-kv-cache-from-scratch"))

from kv_lib import apply_rope, rope_tables  # noqa: E402


# --------------------------------------------------------------------- the pool
class BlockPool:
    """The KV cache as a pool of fixed-size pages, one storage tensor per layer.

    This is the whole PagedAttention idea: memory is handed out in blocks, so the
    only waste is the unused tail of a sequence's last block (< block_size tokens)
    rather than the entire unused remainder of a preallocated max-length slab.
    """

    def __init__(self, cfg, n_blocks, block_size, device="cpu"):
        d_head = cfg.n_embd // cfg.n_head
        shape = (n_blocks, cfg.n_kv_heads, block_size, d_head)
        self.k = [torch.zeros(shape, device=device) for _ in range(cfg.n_layer)]
        self.v = [torch.zeros(shape, device=device) for _ in range(cfg.n_layer)]
        self.block_size = block_size
        self.n_blocks = n_blocks
        self.free = list(range(n_blocks))
        self.refcount = [0] * n_blocks          # >1 when a block is shared (prefix cache)
        self.peak_used = 0

    def available(self):
        return len(self.free)

    def alloc(self, n):
        if n > len(self.free):
            return None                          # out of KV memory: the request must wait
        ids = [self.free.pop() for _ in range(n)]
        for b in ids:
            self.refcount[b] = 1
        self.peak_used = max(self.peak_used, self.n_blocks - len(self.free))
        return ids

    def share(self, block_id):
        self.refcount[block_id] += 1             # a second sequence now points here
        return block_id

    def release(self, ids):
        for b in ids:
            self.refcount[b] -= 1
            if self.refcount[b] == 0:
                self.free.append(b)

    def write(self, layer, block_ids, start, k, v):
        """Write T tokens starting at logical position `start` into the block table."""
        T = k.shape[1]                           # k: (n_kv, T, d_head)
        for t in range(T):
            pos = start + t
            b = block_ids[pos // self.block_size]
            off = pos % self.block_size
            self.k[layer][b, :, off] = k[:, t]
            self.v[layer][b, :, off] = v[:, t]

    def gather(self, layer, block_ids, length):
        """Read a sequence's K/V back as one contiguous tensor.

        A real CUDA kernel reads the blocks in place; gathering them here costs a
        copy we would not pay on a GPU, but it leaves the *allocation* semantics —
        which is what this project measures — exactly right.
        """
        k = torch.cat([self.k[layer][b] for b in block_ids], dim=1)[:, :length]
        v = torch.cat([self.v[layer][b] for b in block_ids], dim=1)[:, :length]
        return k, v                              # (n_kv, length, d_head)


@dataclass
class Sequence:
    sid: int
    prompt: torch.Tensor                          # (T,) token ids
    max_new: int
    arrival: float = 0.0
    blocks: list = field(default_factory=list)
    tokens: list = field(default_factory=list)   # generated ids
    cached_len: int = 0                          # tokens whose K/V are already written
    t_admit: float = None
    t_first: float = None                        # -> TTFT
    t_end: float = None
    reused_blocks: int = 0                       # prefix-cache hits

    @property
    def length(self):
        return len(self.prompt) + len(self.tokens)

    @property
    def done(self):
        return len(self.tokens) >= self.max_new


# ------------------------------------------------------------------- the engine
class Engine:
    """Schedules sequences over a BlockPool. `policy` picks the scheduling rule."""

    def __init__(self, model, block_size=16, n_blocks=512, max_batch=16,
                 policy="continuous", prefix_cache=False, device="cpu"):
        self.model, self.cfg = model, model.cfg
        self.pool = BlockPool(model.cfg, n_blocks, block_size, device)
        self.max_batch, self.policy = max_batch, policy
        self.prefix_cache = prefix_cache
        self.prefix_index = {}                   # hash(prefix tokens) -> block id
        self.waiting, self.running, self.finished = [], [], []
        self.d_head = self.cfg.n_embd // self.cfg.n_head
        self.cos, self.sin = rope_tables(self.d_head, 2048, device)
        self.prefill_tokens = 0                  # work actually done (for the cache study)
        self.saved_tokens = 0                    # work skipped thanks to the prefix cache

    # -- admission ---------------------------------------------------------- #
    def _blocks_needed(self, seq):
        # Only what the prompt needs (plus room for the first generated token).
        # Blocks for the rest of the generation are taken *on demand* as the
        # sequence grows — a request that stops early never pays for its tail.
        return -(-(len(seq.prompt) + 1) // self.pool.block_size)  # ceil

    def _grow(self, seq, upto):
        """Make sure a block exists for logical position `upto`."""
        while upto // self.pool.block_size >= len(seq.blocks):
            nb = self.pool.alloc(1)
            if nb is None:
                raise RuntimeError("KV pool exhausted — a real engine would preempt here")
            seq.blocks += nb

    def _prefix_key(self, ids):
        return hashlib.sha1(bytes(ids)).hexdigest()

    def _admit(self, seq):
        need = self._blocks_needed(seq)
        table = []
        bs = self.pool.block_size

        if self.prefix_cache:
            # Walk the prompt block by block. A block whose *entire* token history
            # matches one we have seen can be pointed at instead of recomputed —
            # this is vLLM's automatic prefix caching, hashed on the prefix, not
            # just the block, so identical text in a different context never
            # collides.
            n_full = len(seq.prompt) // bs
            for i in range(n_full):
                key = self._prefix_key(seq.prompt[: (i + 1) * bs].tolist())
                if key in self.prefix_index and i == len(table):
                    table.append(self.pool.share(self.prefix_index[key]))
                    seq.reused_blocks += 1
                else:
                    break
            seq.cached_len = len(table) * bs     # these tokens need no prefill

        fresh = self.pool.alloc(need - len(table))
        if fresh is None:
            if table:                            # roll back the shares we just took
                self.pool.release(table)
                seq.cached_len, seq.reused_blocks = 0, 0
            return False
        seq.blocks = table + fresh

        if self.prefix_cache:
            # Register the full prompt blocks we are about to fill, and *pin* each
            # one (an extra reference) so it survives this sequence's release. That
            # pin is the honest price of a prefix cache: it holds KV memory hostage
            # for reuse, which is why real engines evict it under pressure.
            for i in range(len(seq.prompt) // bs):
                key = self._prefix_key(seq.prompt[: (i + 1) * bs].tolist())
                if key not in self.prefix_index:
                    self.prefix_index[key] = seq.blocks[i]
                    self.pool.refcount[seq.blocks[i]] += 1
        seq.t_admit = time.perf_counter()
        return True

    # -- the model, run over a batch of sequences --------------------------- #
    @torch.no_grad()
    def _forward(self, seqs, is_prefill):
        """One model pass. Prefill: seq.prompt[cached_len:]. Decode: one token each."""
        model = self.model
        if is_prefill:
            seq = seqs[0]                        # prefills run one sequence at a time
            ids = seq.prompt[seq.cached_len:].unsqueeze(0)
            starts = [seq.cached_len]
        else:
            ids = torch.tensor([[s.tokens[-1] if s.tokens else s.prompt[-1].item()]
                                for s in seqs])
            starts = [s.length - 1 for s in seqs]

        B, T = ids.shape
        for i, s in enumerate(seqs):                    # grow block tables on demand
            self._grow(s, starts[i] + T - 1)

        x = model.tok(ids)
        for li, blk in enumerate(model.blocks):
            h = blk.n1(x)
            q = blk.attn.q(h).view(B, T, blk.attn.nh, blk.attn.dh).transpose(1, 2)
            k = blk.attn.k(h).view(B, T, blk.attn.nkv, blk.attn.dh).transpose(1, 2)
            v = blk.attn.v(h).view(B, T, blk.attn.nkv, blk.attn.dh).transpose(1, 2)

            # RoPE: every sequence in the batch sits at a different absolute
            # position, so each row gets its own slice of the angle tables.
            q = torch.stack([apply_rope(q[i:i + 1], self.cos[s0:s0 + T],
                                        self.sin[s0:s0 + T])[0]
                             for i, s0 in enumerate(starts)])
            k = torch.stack([apply_rope(k[i:i + 1], self.cos[s0:s0 + T],
                                        self.sin[s0:s0 + T])[0]
                             for i, s0 in enumerate(starts)])

            outs = []
            for i, s in enumerate(seqs):
                self.pool.write(li, s.blocks, starts[i], k[i], v[i])
                kk, vv = self.pool.gather(li, s.blocks, starts[i] + T)
                if blk.attn.nkv != blk.attn.nh:
                    rep = blk.attn.nh // blk.attn.nkv
                    kk = kk.repeat_interleave(rep, 0); vv = vv.repeat_interleave(rep, 0)
                qi = q[i]                                   # (nh, T, dh)
                if T == 1:
                    y = F.scaled_dot_product_attention(qi.unsqueeze(0), kk.unsqueeze(0),
                                                       vv.unsqueeze(0))
                else:
                    p0 = kk.shape[1] - T
                    mask = torch.ones(T, kk.shape[1], dtype=torch.bool).tril(diagonal=p0)
                    y = F.scaled_dot_product_attention(qi.unsqueeze(0), kk.unsqueeze(0),
                                                       vv.unsqueeze(0), attn_mask=mask)
                outs.append(y[0])
            y = torch.stack(outs).transpose(1, 2).contiguous().view(B, T, -1)
            x = x + blk.attn.o(y)
            x = x + blk.mlp(blk.n2(x))
        return model.head(model.norm_f(x))[:, -1]           # (B, vocab)

    def _prefill(self, seq):
        n = len(seq.prompt) - seq.cached_len
        self.prefill_tokens += n
        self.saved_tokens += seq.cached_len
        logits = self._forward([seq], is_prefill=True)
        seq.tokens.append(int(logits[0].argmax()))
        seq.t_first = time.perf_counter()

    def _decode(self, seqs):
        logits = self._forward(seqs, is_prefill=False)
        for i, s in enumerate(seqs):
            s.tokens.append(int(logits[i].argmax()))

    # -- the schedulers ----------------------------------------------------- #
    def run(self, requests, now=None):
        """Serve every request; return per-request records.

        static:      fill a batch, run it to completion, then start the next one.
                     Every sequence in the batch is held hostage by the longest.
        continuous:  re-form the batch every decode step.
        """
        self.waiting = sorted(requests, key=lambda s: s.arrival)
        self.peak_running = self.peak_tokens = 0
        t0 = time.perf_counter()
        origin = self.waiting[0].arrival

        while self.waiting or self.running:
            wall = time.perf_counter() - t0

            if self.policy == "static" and self.running:
                pass                                        # no admissions mid-batch
            else:
                # admit anything that has arrived and fits in KV memory
                while (self.waiting and len(self.running) < self.max_batch
                       and self.waiting[0].arrival - origin <= wall):
                    seq = self.waiting[0]
                    if not self._admit(seq):
                        break                               # pool full: wait for a release
                    self.waiting.pop(0)
                    self._prefill(seq)                      # prefill on admission
                    self.running.append(seq)

            if not self.running:
                if self.waiting:                            # clock hasn't reached next arrival
                    time.sleep(0.001)
                    continue
                break

            active = [s for s in self.running if not s.done]
            if active:
                self._decode(active)

            # what the KV cache is actually holding right now
            self.peak_running = max(self.peak_running, len(self.running))
            self.peak_tokens = max(self.peak_tokens, sum(s.length for s in self.running))

            still = []
            for s in self.running:
                if s.done:
                    s.t_end = time.perf_counter()
                    self.pool.release(s.blocks)
                    self.finished.append(s)
                else:
                    still.append(s)
            self.running = still

        return self.finished
