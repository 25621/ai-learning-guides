"""Weight quantization from scratch: RTN, GPTQ and AWQ.

`auto-gptq`, `awq` and `bitsandbytes` are not installed here, which is a gift:
all three algorithms are short enough to write out, and writing them out is the
only way to see *why* the data-aware ones beat plain rounding.

The three recipes, in one sentence each:

  RTN   round each weight to the nearest representable level. Data-free.
  GPTQ  round the weights *one column at a time*, and after each column push the
        rounding error you just made into the columns you have not done yet, so
        the error cancels in the layer's output. Needs the Hessian X^T X of the
        layer's inputs.
  AWQ   the weights multiplied by large activations matter most, so scale those
        input channels up before rounding (and fold the scale back out after).
        Needs only the average activation magnitude per input channel.

Everything here is *simulated* quantization: we round the weights to the INT4/INT8
grid and store the dequantized values back in fp32, because CPUs have no int4
kernels. That measures quality exactly (the arithmetic the model sees is
identical) and lets `packed_bytes` account for memory analytically.

Imported by project 65 (which swaps the integer grid for FP8).
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "51-mmlu-re-run"))

import eval_lib as E  # noqa: E402  (Model wrapper, load_parquet, MMLU helpers)

torch.set_num_threads(12)


# ------------------------------------------------------------------ the grid
def quantize_grid(W, bits, group_size, sym=False):
    """Round W to a `bits`-wide integer grid, per group of `group_size` inputs.

    W is (out, in). Groups run along the input dimension: each group of columns
    gets its own scale and zero-point, which is why group-wise 4-bit beats
    per-tensor 4-bit — one outlier column can only poison its own group.
    """
    out, cin = W.shape
    g = cin if group_size in (None, -1) else group_size
    assert cin % g == 0, f"group_size {g} must divide in-features {cin}"
    Wg = W.reshape(out, cin // g, g)

    if sym:
        scale = Wg.abs().amax(-1, keepdim=True) / (2 ** (bits - 1) - 1)
        scale = scale.clamp(min=1e-8)
        zero = torch.zeros_like(scale)
        q = torch.clamp(torch.round(Wg / scale), -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    else:
        lo, hi = Wg.amin(-1, keepdim=True), Wg.amax(-1, keepdim=True)
        scale = ((hi - lo) / (2 ** bits - 1)).clamp(min=1e-8)
        zero = torch.round(-lo / scale)
        q = torch.clamp(torch.round(Wg / scale) + zero, 0, 2 ** bits - 1)

    return ((q - zero) * scale).reshape(out, cin)


def packed_bytes(numel, bits, group_size, scale_bits=16, zero_bits=None):
    """Bytes a real int-quantized kernel would store: weights + scales + zeros."""
    zero_bits = bits if zero_bits is None else zero_bits
    g = numel if group_size in (None, -1) else group_size
    n_groups = numel / g
    return (numel * bits + n_groups * (scale_bits + zero_bits)) / 8


# ------------------------------------------------------------------------ GPTQ
class GPTQ:
    """One linear layer's worth of GPTQ (Frantar et al., 2022).

    Accumulate H = 2 X^T X from calibration inputs, then quantize column by
    column, compensating the remaining columns for each rounding error with
    H^-1. The compensation is the whole algorithm: RTN makes the same rounding
    decisions but never tells the other columns about them.
    """

    def __init__(self, linear: nn.Linear):
        self.lin = linear
        self.cin = linear.in_features
        self.H = torch.zeros(self.cin, self.cin)
        self.n = 0

    def add_batch(self, x):
        x = x.reshape(-1, self.cin).float()              # (tokens, in)
        self.H *= self.n / (self.n + x.shape[0])
        self.n += x.shape[0]
        self.H += (2.0 / self.n) * (x.T @ x)

    def quantize(self, bits, group_size, blocksize=128, damp=0.01):
        W = self.lin.weight.data.clone().float()        # (out, in)
        H = self.H.clone()

        dead = torch.diag(H) == 0                       # input channels calibration never lit up
        H[dead, dead] = 1.0
        W[:, dead] = 0

        H += torch.eye(self.cin) * (damp * torch.diag(H).mean())
        # Hinv upper-Cholesky: the standard trick that turns the sequential
        # error-feedback recursion into two triangular solves.
        Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)),
                                     upper=True)

        g = self.cin if group_size in (None, -1) else group_size
        scale = zero = None
        Q = torch.zeros_like(W)

        for b0 in range(0, self.cin, blocksize):
            b1 = min(b0 + blocksize, self.cin)
            W1, Q1 = W[:, b0:b1].clone(), torch.zeros_like(W[:, b0:b1])
            E1 = torch.zeros_like(W1)
            Hb = Hinv[b0:b1, b0:b1]

            for i in range(b1 - b0):
                col, c = W1[:, i], b0 + i

                if c % g == 0:
                    # A new group starts: fix its scale from the weights as they
                    # stand *now* — already carrying the compensation from every
                    # column quantized before them.
                    grp = W[:, c:c + g]
                    lo, hi = grp.amin(1, keepdim=True), grp.amax(1, keepdim=True)
                    scale = ((hi - lo) / (2 ** bits - 1)).clamp(min=1e-8)
                    zero = torch.round(-lo / scale)

                q = torch.clamp(torch.round(col.unsqueeze(1) / scale) + zero,
                                0, 2 ** bits - 1)
                qcol = ((q - zero) * scale).squeeze(1)
                Q1[:, i] = qcol

                # The error we just made, pre-divided by H^-1_ii, is handed to the
                # columns we have not quantized yet. This line *is* GPTQ.
                err = (col - qcol) / Hb[i, i]
                W1[:, i:] -= err.unsqueeze(1) @ Hb[i, i:].unsqueeze(0)
                E1[:, i] = err

            Q[:, b0:b1] = Q1
            W[:, b1:] -= E1 @ Hinv[b0:b1, b1:]          # same feedback, block-to-block

        self.lin.weight.data = Q.to(self.lin.weight.dtype)
        return Q


# ------------------------------------------------------------------------- AWQ
class AWQ:
    """Activation-aware Weight Quantization (Lin et al., 2023).

    Collect the mean |activation| per input channel, then search a per-channel
    scale s = mean|x|^alpha. Quantizing W·diag(s) and dividing by s afterwards
    leaves the layer's function unchanged in fp32 but moves the salient (large
    activation) channels onto a finer part of the grid.
    """

    def __init__(self, linear: nn.Linear):
        self.lin = linear
        self.cin = linear.in_features
        self.act = torch.zeros(self.cin)
        self.n = 0
        self.sample = None                              # a slice of real inputs for the search

    def add_batch(self, x):
        x = x.reshape(-1, self.cin).float()
        self.act = (self.act * self.n + x.abs().sum(0)) / (self.n + x.shape[0])
        self.n += x.shape[0]
        if self.sample is None:
            self.sample = x[:256].clone()               # enough to rank candidate scales

    def quantize(self, bits, group_size, grid=11):
        W = self.lin.weight.data.clone().float()
        X = self.sample
        ref = X @ W.T                                   # the fp32 output we must preserve
        act = self.act.clamp(min=1e-5)

        best, best_loss = None, float("inf")
        for i in range(grid):
            alpha = i / (grid - 1)                      # 0 = plain RTN, 1 = fully activation-scaled
            s = act.pow(alpha)
            s = (s / (s.max() * s.min()).sqrt()).clamp(min=1e-4)   # keep the scale centred at 1
            Wq = quantize_grid(W * s, bits, group_size) / s
            loss = (X @ Wq.T - ref).pow(2).mean().item()
            if loss < best_loss:
                best, best_loss, self.alpha = Wq, loss, alpha

        self.lin.weight.data = best.to(self.lin.weight.dtype)
        return best


# ---------------------------------------------------------- the layer-by-layer driver
def _hidden(out):
    """A decoder block returns a bare Tensor in transformers 5.x and a tuple in 4.x."""
    return out[0] if isinstance(out, tuple) else out


def _block_inputs(hf_model, ids):
    """Capture the hidden states (and the kwargs) entering the first decoder block."""
    layers = hf_model.model.layers
    caught = {}
    inps = []

    class Catcher(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, hidden_states, **kw):
            inps.append(hidden_states.detach())
            caught.update(kw)
            raise StopIteration                        # we only wanted the inputs

    layers[0] = Catcher(layers[0])
    for i in range(ids.shape[0]):
        try:
            with torch.no_grad():
                hf_model(ids[i:i + 1])
        except StopIteration:
            pass
    layers[0] = layers[0].inner
    return torch.cat(inps, 0), caught


def quantize_model(hf_model, method, bits, group_size, calib_ids=None, verbose=True):
    """Quantize every linear inside the transformer blocks, in place.

    Embeddings and the LM head are left alone — that is what production
    quantizers do too, because they are read once per token, not once per layer.

    GPTQ and AWQ walk the blocks *in order*, quantizing block i using activations
    produced by the already-quantized blocks 0..i-1, so each block also corrects
    for the error its predecessors introduced.
    """
    layers = hf_model.model.layers

    if method == "rtn":                                 # data-free: no calibration at all
        for blk in layers:
            for lin in [m for m in blk.modules() if isinstance(m, nn.Linear)]:
                lin.weight.data = quantize_grid(lin.weight.data.float(), bits, group_size)
        return {}

    inps, kwargs = _block_inputs(hf_model, calib_ids)
    stats = {}
    for li, blk in enumerate(layers):
        lins = {n: m for n, m in blk.named_modules() if isinstance(m, nn.Linear)}
        algos = {n: (GPTQ(m) if method == "gptq" else AWQ(m)) for n, m in lins.items()}

        handles = [m.register_forward_pre_hook(
            lambda mod, args, n=n: algos[n].add_batch(args[0]))
            for n, m in lins.items()]
        with torch.no_grad():
            for j in range(inps.shape[0]):
                blk(inps[j:j + 1], **kwargs)
        for h in handles:
            h.remove()

        for n, a in algos.items():
            a.quantize(bits, group_size)
            if method == "awq":
                stats.setdefault("alpha", []).append(a.alpha)

        with torch.no_grad():                           # propagate through the *quantized* block
            outs = [_hidden(blk(inps[j:j + 1], **kwargs)) for j in range(inps.shape[0])]
        inps = torch.cat(outs, 0)
        if verbose:
            print(f"    {method}: block {li + 1}/{len(layers)}", end="\r", flush=True)
    if verbose:
        print()
    return stats


def model_bytes(hf_model, bits=16, group_size=128):
    """Serving footprint: quantized blocks + fp16 embeddings/head."""
    block_params = sum(m.weight.numel()
                       for blk in hf_model.model.layers
                       for m in blk.modules() if isinstance(m, nn.Linear))
    total = sum(p.numel() for p in hf_model.parameters())
    other = total - block_params                        # embeddings, norms, (tied) head
    quant = block_params * 2 if bits == 16 else packed_bytes(block_params, bits, group_size)
    return quant + other * 2, block_params, total


# --------------------------------------------------------------------- evaluation
def wikitext(n_chars=600_000):
    d = E.load_parquet("Salesforce/wikitext", "wikitext-2-raw-v1", "test")
    return "\n\n".join(d["text"])[:n_chars]


@torch.no_grad()
def perplexity(model, tok, text, seqlen=512, n_seq=16):
    """Standard fixed-window perplexity — the metric every quantization paper reports."""
    ids = tok(text, return_tensors="pt").input_ids[0]
    nlls = []
    for i in range(n_seq):
        chunk = ids[i * seqlen:(i + 1) * seqlen].unsqueeze(0)
        if chunk.shape[1] < seqlen:
            break
        out = model(chunk, labels=chunk)
        nlls.append(out.loss.float() * (seqlen - 1))
    return float(torch.exp(torch.stack(nlls).sum() / (len(nlls) * (seqlen - 1))))


def mmlu_accuracy(wrapper: E.Model, items):
    prompts = [E.mmlu_cloze(it["question"], it["choices"]) for it in items]
    preds, _ = wrapper.mc_score(prompts, batch_size=16)
    correct = sum(int(p == it["answer"]) for p, it in zip(preds, items))
    return correct / len(items), correct, len(items)
