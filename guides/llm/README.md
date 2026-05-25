# Large Language Models: From Beginner to Advanced

A comprehensive guide to large language models as a *subject* — not just as the API you call when you need to summarize a PDF. The goal is to take you from "I have used ChatGPT" to "I can read a 2026 frontier-model paper, understand why every architectural and training decision was made, and reproduce a small version of the system end-to-end." The guide is biased toward what matters in practice now: the engineering, the data, and the training recipes that actually ship.

> **An honest framing.** Most of the "magic" of modern LLMs is not a single breakthrough — it is the compounding of a few simple ideas (next-token prediction, the transformer, scaling) executed at enormous cost with enormous care. The math is mostly undergraduate; the engineering is not. This guide teaches the architecture and the objectives because you cannot debug what you do not understand, and then teaches the data, infrastructure, alignment, and inference work that the textbook chapters skip.

---

## Table of Contents

1. [Phase 0: Prerequisites](#phase-0-prerequisites)
2. [Phase 1: Tokenization and Embeddings](#phase-1-tokenization-and-embeddings)
3. [Phase 2: The Transformer Architecture](#phase-2-the-transformer-architecture)
4. [Phase 3: Pretraining — Data, Objectives, and the Loss Curve](#phase-3-pretraining--data-objectives-and-the-loss-curve)
5. [Phase 4: Scaling Laws and Training Infrastructure](#phase-4-scaling-laws-and-training-infrastructure)
6. [Phase 5: Post-training — SFT, RLHF, DPO, GRPO, RLVR](#phase-5-post-training--sft-rlhf-dpo-grpo-rlvr)
7. [Phase 6: Reasoning and Inference-Time Compute](#phase-6-reasoning-and-inference-time-compute)
8. [Phase 7: Retrieval, Tools, and Agents](#phase-7-retrieval-tools-and-agents)
9. [Phase 8: Evaluation](#phase-8-evaluation)
10. [Phase 9: Efficient Inference and Deployment](#phase-9-efficient-inference-and-deployment)
11. [Phase 10: Safety, Interpretability, and Frontier Topics](#phase-10-safety-interpretability-and-frontier-topics)
12. [Suggested Timeline](#suggested-timeline)
13. [Key Advice](#key-advice)
14. [Common Pitfalls](#common-pitfalls)
15. [Additional Resources](#additional-resources)
16. [Glossary](/shared/glossary/)

---

## Phase 0: Prerequisites

LLMs combine probability, deep learning, distributed systems, and a surprising amount of plumbing. You can get started without mastering everything below, but if more than a couple are unfamiliar, slow down before continuing.

### Concepts to Know

- **Probability**: random variables, conditional probability, expectation, cross-entropy, KL divergence
- **Linear algebra**: matrix multiplication, broadcasting, attention is `softmax(QKᵀ/√d)V` and you should be able to read that without flinching
- **Calculus**: gradients, chain rule, the difference between `log p(x)` and `log p(x | context)`
- **Optimization**: SGD, Adam/AdamW, learning-rate schedules, gradient clipping, mixed-precision training
- **Deep learning**: training loops, `nn.Module`, layer normalization, residual connections, dropout. If shaky, do the [PyTorch Deep Dive](../pytorch-deep-dive/) first
- **Programming maturity**: you will read other people's training code at 3 a.m. trying to figure out which of 14 norms is misplaced

### The One Equation Everything Comes Back To

```
              ┌────────────────────────────────────────────┐
              │   p(x_1, x_2, ..., x_n)                    │
              │       =  Π_t  p( x_t | x_{<t} )            │
              └────────────────────────────────────────────┘

The probability of a sequence is the product of the probabilities of each token
given everything that came before. An LLM is a function that estimates the
right-hand-side factor: p(next token | context).

Everything in this guide — every architectural choice, every loss function,
every training trick, every alignment algorithm — is some clever way of
estimating, shaping, or sampling from this one conditional distribution.
```

If that sentence is fuzzy now, it will be sharp by the end of Phase 3.

### What You Need Installed

- **Python 3.10+**, PyTorch, NumPy
- **Hugging Face `transformers`, `datasets`, `tokenizers`, `accelerate`, `peft`, `trl`** — the de facto stack for everything that isn't a frontier lab's internal codebase
- **`nanoGPT`** (Karpathy) — the cleanest "build it yourself" reference implementation
- **`vllm` or `sglang`** — production inference engines; read their code
- **A GPU** — anything with ≥ 16 GB VRAM is enough for Phases 1–4 toy models. For real pretraining or 8B-scale fine-tuning, multi-GPU access is required

### Resources

- [Karpathy — *Let's build GPT: from scratch, in code, spelled out* (YouTube)](https://www.youtube.com/watch?v=kCc8FmEb1nY) — the single best on-ramp; watch before reading anything else
- [Karpathy — *nanoGPT*](https://github.com/karpathy/nanoGPT) — the code that goes with it
- [Jay Alammar — *The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/) — the visual intuition
- [*Speech and Language Processing* (Jurafsky & Martin), 3rd ed. draft](https://web.stanford.edu/~jurafsky/slp3/) — the NLP textbook, free
- [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course) — practical tooling
- [Stanford CS336 — *Language Models from Scratch*](https://stanford-cs336.github.io/spring2024/) — the modern course

---

## Phase 1: Tokenization and Embeddings

Before the model sees a single matrix multiply, your text becomes a sequence of integer token IDs. Tokenization is the most boring-looking part of the stack and the source of an astonishing fraction of real production bugs.

### Concepts to Learn

- **Why not characters? Why not words?** Character models are too long; word models have unbounded vocabulary and no way to handle rare words. Subword tokenization is the compromise everyone settled on
- **Byte-Pair Encoding (BPE)** — start with bytes (or characters), greedily merge the most frequent adjacent pair, repeat until you hit a target vocabulary size
- **Byte-level BPE** — start from raw UTF-8 bytes; guarantees no out-of-vocabulary token can ever occur. The GPT-2/3/4 family choice
- **WordPiece, SentencePiece, Unigram LM** — the cousins; SentencePiece (used by Llama, Mistral, Gemma) is BPE with whitespace treated as a normal symbol so you can tokenize a string verbatim
- **Vocabulary size trade-off**: bigger vocab → fewer tokens per document (faster training and inference) but bigger embedding matrix (more parameters, more memory)
- **Special tokens**: `<bos>`, `<eos>`, `<pad>`, `<unk>`, plus the chat-template tokens (`<|im_start|>`, `<|im_end|>`, `<|system|>`, `<|user|>`, `<|assistant|>`, …). These are the source of half of all subtle inference bugs
- **The embedding matrix** `E ∈ ℝ^{V × d}`: token IDs index into this matrix to produce vectors. Usually tied to the output projection (the "unembedding")
- **Tokenization pathologies** that bite production systems:
  - The "leading space" problem: `"Paris"` and `" Paris"` are different tokens
  - Numerals: `"1234"` may be one token in one tokenizer and four in another, which is why early LLMs were bad at arithmetic
  - Multilingual coverage: a tokenizer trained on English bytes uses 4× more tokens for Chinese, which is both expensive and a quality problem

### The Tokenizer's Job, Annotated

```
"Hello, world!"                                    raw string
       │
       ▼   (1) normalize: NFC, lowercase?, strip?  ← lossy step, choose carefully
"hello, world!"
       │
       ▼   (2) pre-tokenize: split on whitespace / punctuation
["hello", ",", "world", "!"]
       │
       ▼   (3) apply learned BPE merges (or unigram model)
["hel", "lo", ",", "wor", "ld", "!"]
       │
       ▼   (4) map to integer IDs via the vocabulary
[15043, 1018, 11, 1734, 359, 0]
       │
       ▼   (5) prepend / append special tokens for the task
[<bos>, 15043, 1018, 11, 1734, 359, 0, <eos>]

That last step is where chat templates live, and where 90% of fine-tuning
bugs are born. If training tokens don't match inference tokens, exactly,
the model silently underperforms.
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [Train a BPE from scratch](projects/01-train-a-bpe-from-scratch/README.md) | 1 MB of text, 5 000-token vocab; implement merge selection by hand; serialize and reload | ⭐⭐ |
| [Tokenizer compression study](projects/02-tokenizer-compression-study/README.md) | Tokenize the same Wikipedia paragraph in English, French, Mandarin, Hindi, Bengali with `tiktoken`, Llama 3, and a Gemma tokenizer; plot tokens-per-byte | ⭐⭐ |
| [Numeral tokenization audit](projects/03-numeral-tokenization-audit/README.md) | For 1 000 integers from 0–10 000, count tokens per number across tokenizers; explain the digit-grouping decisions | ⭐⭐⭐ |
| [Chat-template debugger](projects/04-chat-template-debugger/README.md) | Take a fine-tuned chat model, render the chat template by hand, byte-compare against `tokenizer.apply_chat_template`. Find one off-by-one whitespace bug | ⭐⭐⭐ |
| [Custom vocab extension](projects/05-custom-vocab-extension/README.md) | Add 256 new tokens (e.g. for chemical SMILES) to an existing tokenizer; resize embeddings; verify the model still generates valid English | ⭐⭐⭐⭐ |

### Sample Code: A Minimal BPE

```python
from collections import Counter

def get_pairs(tokens):
    return Counter(zip(tokens, tokens[1:]))

def merge(tokens, pair, new_id):
    out, i = [], 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
            out.append(new_id); i += 2
        else:
            out.append(tokens[i]); i += 1
    return out

text = open("input.txt").read().encode("utf-8")
tokens = list(text)
merges, next_id = {}, 256

for _ in range(500):                                  # target 256 + 500 = 756 vocab
    pairs = get_pairs(tokens)
    if not pairs: break
    top, _ = pairs.most_common(1)[0]
    tokens = merge(tokens, top, next_id)
    merges[top] = next_id
    next_id += 1

print(f"vocab = {next_id}, compressed tokens = {len(tokens)} (was {len(text)})")
```

### Key Insight

The tokenizer is part of the model. It is trained on data, has parameters (the merge table and the embedding matrix), and its decisions silently constrain everything the model can ever do. A model whose tokenizer assigns one token to `"strawberry"` cannot count its `r`s by introspection — it never sees them. Tokenization quirks explain a startling fraction of "the model is bad at X" mysteries. Always include the tokenizer in your debugging mental model.

### Resources

- [Sennrich et al. — *Neural Machine Translation of Rare Words with Subword Units* (2016)](https://arxiv.org/abs/1508.07909) — the BPE paper
- [Kudo — *SentencePiece* (2018)](https://arxiv.org/abs/1808.06226)
- [Karpathy — *Let's build the GPT Tokenizer* (YouTube)](https://www.youtube.com/watch?v=zduSFxRajkE) — best single resource on tokenization
- [Hugging Face tokenizers docs](https://huggingface.co/docs/tokenizers)

---

## Phase 2: The Transformer Architecture

The transformer is the architecture every modern LLM is built on. Understanding it deeply — not the cartoon version, the actual matrix shapes — is non-negotiable. Almost every line of pretraining-loop code is about feeding data into a stack of transformer blocks and computing the loss on what falls out.

### Concepts to Learn

- **The decoder-only transformer**: a stack of identical blocks, each with self-attention and an MLP, plus a final linear head to vocabulary. The architecture used by GPT, Llama, Mistral, Qwen, DeepSeek, Gemma, Claude
- **Self-attention** as a soft, content-addressable lookup:
  - Project the input to **queries**, **keys**, **values**: `Q = XW_Q`, `K = XW_K`, `V = XW_V`
  - Compute attention weights: `A = softmax(QKᵀ / √d_k)`
  - Mix values: `O = AV`
  - **Causal mask**: zero out the upper triangle so position `t` can only attend to positions `≤ t`
- **Multi-head attention**: split the model dimension into `h` heads, do attention in parallel, concatenate, project back. Heads specialize during training
- **Multi-Query (MQA) and Grouped-Query Attention (GQA)**: share keys/values across query heads to shrink the KV cache. Llama 2 70B onwards uses GQA; nearly every model since 2024 uses it
- **The MLP / FFN block**: two linear layers with a nonlinearity in between. Modern models use **SwiGLU** or **GeGLU** — gated variants that empirically beat plain GELU
- **Residual connections and pre-norm**: `x = x + Attn(Norm(x))` then `x = x + FFN(Norm(x))`. Pre-norm (norm inside the residual branch) trains stably without learning-rate warmup gymnastics; post-norm (original transformer) does not. Every modern model is pre-norm
- **Normalization**: **LayerNorm** for older models, **RMSNorm** for newer ones (Llama family). RMSNorm drops the mean-centering and bias, saves a few FLOPs, no quality loss
- **Positional information** — the transformer is permutation-invariant unless you add it:
  - **Absolute learned embeddings** — GPT-2 era; doesn't extrapolate beyond training length
  - **Sinusoidal** — original "Attention is All You Need"; extrapolates poorly in practice
  - **ALiBi** — bias attention scores by `-m·distance`; cheap and decent
  - **RoPE (Rotary Position Embedding)** — rotate Q and K vectors by position-dependent angles. The current default. Llama, Mistral, Qwen, DeepSeek all use RoPE
  - **YaRN, NTK-aware, position interpolation** — tricks to extend RoPE context window post-training
- **Attention variants you must know exist**:
  - **FlashAttention 2/3** — same math, IO-aware implementation, 2–10× speed, the only way to train at modern scale
  - **Sliding-window attention** — local attention with a fixed window (Mistral)
  - **Sparse / linear / state-space alternatives** — Mamba, RWKV, Hyena — the perennial "what if not attention" thread
- **Mixture-of-Experts (MoE)**: replace the MLP with `N` "experts" and a router that picks the top-`k` per token. Same compute per token, much more total parameters. Mixtral, DeepSeek-V3, Qwen3-Coder are MoEs

### The Transformer Block, Annotated

```
                 ┌────────────────── input x  (B, T, d) ──────────────────┐
                 │                                                         │
                 ▼                                                         │
            RMSNorm                                                        │
                 │                                                         │
                 ▼                                                         │
            ┌─────────┐                                                    │
            │  Self-  │  ← Q, K, V projections                             │
            │ Attn(h, │  ← RoPE applied to Q and K                         │
            │ causal) │  ← softmax(QKᵀ/√d) V, then out-projection          │
            └─────────┘                                                    │
                 │                                                         │
                 + ◄────────────────────────────────────────────────────── │  residual
                 │
                 ▼
            RMSNorm
                 │
                 ▼
            ┌─────────┐
            │  SwiGLU │  ← x_proj · σ(g_proj)  (two parallel linears, then *)
            │   MLP   │  ← typically 8/3 × d hidden width
            └─────────┘
                 │
                 + ◄────────────────────────────────────────────────────── residual
                 │
                 ▼
               output x'  (B, T, d)        — feed into the next block

Stack 32 of these (7B model), 80 of these (70B model), 96+ of these (frontier).
Final RMSNorm, then a single Linear(d → vocab_size) gives logits.
```

### Why √d_k in the Softmax

```
Dot products of two random d-dimensional vectors have variance ~d.
Without the √d_k scale, large d means large logits, softmax saturates,
gradients vanish, training stalls. The scale keeps logit variance ≈ 1
regardless of head dimension.
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [Single attention head](projects/06-single-attention-head/README.md) | Implement scaled dot-product attention with a causal mask; verify against `F.scaled_dot_product_attention` | ⭐⭐ |
| [Multi-head attention](projects/07-multi-head-attention/README.md) | Add multi-head splitting, output projection, GQA option; verify equivalence to PyTorch's `nn.MultiheadAttention` | ⭐⭐⭐ |
| [nanoGPT reproduction](projects/08-nanogpt-reproduction/README.md) | Type out Karpathy's nanoGPT from scratch; train on tiny Shakespeare; sample text | ⭐⭐⭐ |
| [Pre-norm vs post-norm](projects/09-pre-norm-vs-post-norm/README.md) | Train two 6-layer models, identical except for norm placement; observe training stability with and without warmup | ⭐⭐⭐ |
| [RoPE from scratch](projects/10-rope-from-scratch/README.md) | Implement RoPE, including the half-rotation trick; test by checking that `<q, k>` depends only on relative position | ⭐⭐⭐⭐ |
| [GQA ablation](projects/11-gqa-ablation/README.md) | Train identical 100M models with MHA, GQA-4, MQA; measure KV-cache size and validation loss | ⭐⭐⭐⭐ |
| [Mini-MoE](projects/12-mini-moe/README.md) | Add an 8-expert top-2 MoE MLP to nanoGPT; verify routing balances; observe the loss curve | ⭐⭐⭐⭐⭐ |
| [Long-context extension](projects/13-long-context-extension/README.md) | Take a 4k-context model and apply position interpolation or YaRN to extend to 16k; measure needle-in-a-haystack | ⭐⭐⭐⭐⭐ |

### Sample Code: Causal Multi-Head Self-Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalMHA(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads=None):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads        # GQA: n_kv_heads < n_heads
        self.d_head = d_model // n_heads
        self.W_q  = nn.Linear(d_model, n_heads      * self.d_head, bias=False)
        self.W_kv = nn.Linear(d_model, 2 * self.n_kv_heads * self.d_head, bias=False)
        self.W_o  = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.W_q(x).view(B, T, self.n_heads,    self.d_head).transpose(1, 2)
        kv = self.W_kv(x).view(B, T, 2, self.n_kv_heads, self.d_head)
        k, v = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)

        # Repeat KV heads to match Q heads (GQA broadcasting)
        if self.n_kv_heads != self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        # FlashAttention under the hood; causal mask is free
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(y)
```

### Key Insight

The transformer is not magic. It is a **stack of two operations**: attention (mix information across positions) and an MLP (process information at each position independently). Everything else — RoPE, GQA, SwiGLU, RMSNorm — is a small refinement that earned its place by improving loss-per-FLOP at scale, not by being theoretically clean. Read the architecture as a recipe of practical optimizations layered on a stunningly simple base, and the field stops feeling like a magic show.

### Resources

- [Vaswani et al. — *Attention Is All You Need* (2017)](https://arxiv.org/abs/1706.03762) — the original paper
- [Karpathy — *nanoGPT*](https://github.com/karpathy/nanoGPT) — read it line by line
- [*The Annotated Transformer* (Harvard NLP)](http://nlp.seas.harvard.edu/annotated-transformer/) — paper-with-code walkthrough
- [Llama 3 paper (Meta, 2024)](https://arxiv.org/abs/2407.21783) — modern reference architecture, exhaustive details
- [DeepSeek-V3 technical report (2024)](https://arxiv.org/abs/2412.19437) — MoE done at scale
- [Dao et al. — *FlashAttention-2* (2023)](https://arxiv.org/abs/2307.08691)
- [Su et al. — *RoFormer* / RoPE (2021)](https://arxiv.org/abs/2104.09864)
- [Shazeer — *GLU Variants Improve Transformer* (2020)](https://arxiv.org/abs/2002.05202)

---

## Phase 3: Pretraining — Data, Objectives, and the Loss Curve

Pretraining is where a randomly-initialized transformer becomes a model that knows English, code, math, and a lot of facts. The loss function is trivially simple. Everything interesting happens in the data, the schedule, and the operations.

### Concepts to Learn

- **The next-token-prediction objective**: cross-entropy on the next token, averaged across every position in every sequence:
  - `L = -E_{x ~ data}[ Σ_t log p_θ(x_t | x_{<t}) ]`
- **Why this works at all**: predicting the next token forces the model to learn syntax, semantics, world knowledge, reasoning shortcuts, and the conditional distribution over continuations. Compression is understanding
- **Teacher forcing**: at training time, the model sees the true previous tokens, not its own predictions. This is what makes training parallel across positions (all token losses computed in one forward pass)
- **Causal masking** matches teacher forcing: position `t` predicts position `t+1` using only positions `≤ t`
- **Data is the model.** A 7B model trained on Common Crawl and a 7B model trained on curated code+math+textbook data are different models with different abilities. Modern pretraining is **80% a data problem and 20% an architecture problem**
- **Data pipeline stages**:
  - **Sourcing**: web crawl (CommonCrawl, C4, RefinedWeb, FineWeb, FineWeb-Edu), code (GitHub, The Stack v2), books, scientific papers (arXiv, PubMed), reference (Wikipedia)
  - **Deduplication**: exact dedup, MinHash near-dedup. Removing duplicates is one of the highest-ROI moves in pretraining
  - **Filtering**: language ID, perplexity filters, classifier-based quality filters, toxicity filters, PII removal
  - **Mixing**: weighted sampling across sources. Decisions here have huge effect (see *DoReMi*, the Llama 3 data mix ablations)
  - **Tokenization and shuffling**: pack into fixed-length sequences, document-attention masks, deterministic shuffles for replayability
- **Training schedule**:
  - **AdamW** with weight decay; `β1 = 0.9`, `β2 = 0.95`, `ε = 1e-8` are typical
  - **Warmup** (~1% of steps) then **cosine decay** to ~10% of peak LR
  - Some recent work uses **WSD (Warmup-Stable-Decay)** — constant LR, then a short decay at the end; enables continued pretraining without re-warming
  - **Mixed precision** (`bf16` everywhere, sometimes `fp8`); master weights and optimizer states in `fp32`
  - **Gradient clipping** at norm 1.0
  - **Batch size** ramp-up — start small, grow over training
- **Curriculum / data ordering**: simple → complex, or all-uniform? The honest answer is "mostly uniform, with quality-weighted upsampling near the end of training"
- **The loss curve as a diagnostic**: a healthy run goes down log-linearly in tokens, with periodic spikes when the optimizer hits something rough. *Loss spikes* are normal; *loss divergence* is not
- **Mid-training and continued pretraining**: shift the data mix late in training (more code, math, instructions) to "anneal" toward useful skills before post-training

### The Pretraining Loss, Implemented

```python
# Given input_ids of shape (B, T+1), the standard pattern is:
inputs  = input_ids[:, :-1]                            # (B, T)
targets = input_ids[:, 1:]                             # (B, T) — shifted by one
logits  = model(inputs)                                # (B, T, V)
loss    = F.cross_entropy(
    logits.reshape(-1, logits.size(-1)),
    targets.reshape(-1),
    ignore_index=-100,                                 # for padding / masked positions
)
```

That's it. The entire pretraining objective is six lines. The other 50 000 lines of a training repo are data loading, distributed orchestration, checkpointing, logging, and recovery.

### A Modern Pretraining Data Mix (Approximate, Illustrative)

```
   Source                           Weight    Notes
   ────────────────────────────────────────────────────────────────
   Curated web (FineWeb-Edu)          45%     Heavy quality filter
   Code (The Stack v2 + filters)      17%     Permissively licensed
   Books / long-form                  10%     Public-domain + licensed
   Math (arXiv, problem sets)          8%     Latex preserved
   Scientific papers                   7%     arXiv, PubMed-like
   Reference (Wikipedia, etc.)         5%     High-quality, upsampled
   Multilingual web                    6%     CC + filter, per-lang quotas
   Conversational / forum              2%     StackExchange, curated reddit

   Token count: 5–15 trillion tokens for a 7B model (well past Chinchilla),
                 15–30 trillion tokens for a frontier-scale dense model.
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| Train a 10M-parameter LM | nanoGPT on tiny Shakespeare; understand every line of the loop | ⭐⭐ |
| Train a 100M-parameter LM | OpenWebText subset, 8h on one A100; reach <3.5 val loss | ⭐⭐⭐ |
| Dedup ablation | Train two identical 100M models, one on raw CommonCrawl, one after MinHash dedup; compare downstream eval | ⭐⭐⭐⭐ |
| Quality-filter ablation | Repeat with vs. without an educational-quality classifier filter | ⭐⭐⭐⭐ |
| LR schedule sweep | Same model, sweep cosine vs. WSD vs. constant; report val loss + downstream evals | ⭐⭐⭐ |
| Loss-spike forensics | Deliberately cause a loss spike (huge LR, bad data sample); analyze, fix, recover from checkpoint | ⭐⭐⭐⭐ |
| Continued pretraining | Take an open base model; continue-pretrain on 1B tokens of a specialized corpus; measure capability gain and base-task forgetting | ⭐⭐⭐⭐⭐ |

### Sample Code: The Heart of a Pretraining Step

```python
model.train()
for step, batch in enumerate(loader):
    input_ids = batch["input_ids"].to(device, non_blocking=True)

    # Warmup + cosine schedule
    lr = lr_schedule(step, warmup=2000, total=total_steps, peak=3e-4, min_frac=0.1)
    for g in optim.param_groups: g["lr"] = lr

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(input_ids[:, :-1])
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            input_ids[:, 1:].reshape(-1),
        )

    # Scale loss for grad accumulation (effective batch = micro_bs * accum)
    (loss / grad_accum).backward()
    if (step + 1) % grad_accum == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        optim.zero_grad(set_to_none=True)

    if step % log_every == 0:
        log({"loss": loss.item(), "lr": lr, "step": step})
```

### Key Insight

Pretraining is the simplest part of the stack on paper and the hardest in practice. The objective is six lines of code; the data pipeline that feeds it is a hundred thousand. **Frontier models do not differ from open-source models because of the architecture — they share the same architecture.** They differ because they have a better tokenizer, a better data mix, more total compute, more careful schedule choices, and infrastructure that doesn't drop tokens. If you want to push the frontier, push the data and infra; the architecture you can copy.

### Resources

- [Hoffmann et al. — *Training Compute-Optimal Large Language Models* / Chinchilla (2022)](https://arxiv.org/abs/2203.15556)
- [Penedo et al. — *RefinedWeb / FineWeb* (2023–2024)](https://arxiv.org/abs/2406.17557)
- [Llama 3 paper (Meta, 2024)](https://arxiv.org/abs/2407.21783) — the most detailed open pretraining report
- [OLMo & OLMo 2 papers (AllenAI)](https://allenai.org/olmo) — fully open data + code + training logs
- [Xie et al. — *DoReMi: Optimizing Data Mixtures* (2023)](https://arxiv.org/abs/2305.10429)
- [Karpathy — *Let's reproduce GPT-2 (124M)*](https://www.youtube.com/watch?v=l8pRSuU81PU)

---

## Phase 4: Scaling Laws and Training Infrastructure

Why is bigger better? How big is big enough? And how do you train a 70B model on 2 048 GPUs without losing weeks to silent NaNs? This phase is about scale: both the predictive science of it and the engineering of getting there.

### Concepts to Learn

- **The Kaplan scaling laws (2020)**: loss decreases as a power law in parameters, data, and compute. The first formal evidence that "just make it bigger" worked
- **The Chinchilla correction (2022)**: Kaplan over-weighted parameters relative to data. The compute-optimal recipe is **~20 tokens per parameter**, not the much higher parameter:data ratio earlier models used
- **The Chinchilla optimum is not the inference optimum.** If you'll serve a model billions of times, you should **overtrain** (use far more than 20 tokens/param) to get a smaller model with equivalent quality. Llama 2/3 deliberately train past Chinchilla for this reason
- **Compute, in `6 N D` FLOPs**: a forward+backward pass on a dense transformer with `N` parameters and `D` tokens costs approximately `6 N D` FLOPs. This is the back-of-envelope formula you should be able to compute in your head
- **Emergent capabilities** — claims that some capabilities (multi-step arithmetic, in-context learning of new tasks) appear suddenly at scale. The 2023 *Are Emergent Abilities a Mirage?* paper showed many "emergences" are artifacts of discontinuous metrics. The honest story: capabilities improve smoothly with scale, but some downstream metrics step-function as the model crosses a threshold
- **Distributed training parallelism** — the four dimensions that let you fit and feed huge models:
  - **Data parallelism (DP)** — same model on each GPU, different micro-batches, average gradients. The default
  - **Tensor parallelism (TP)** — shard each layer's weights across GPUs in a node (Megatron style). Heavy communication, only viable within a high-bandwidth node
  - **Pipeline parallelism (PP)** — different layers on different GPUs; micro-batches flow as a pipeline. Bubble overhead, but unlocks beyond-one-node models
  - **Expert parallelism (EP)** — for MoE: experts live on different GPUs; routed tokens are sent over the network. The dominant parallelism axis for MoE models
  - **Sequence / context parallelism (SP)** — shard the sequence dimension; essential for long context
- **Optimizer-state sharding**: **ZeRO** (DeepSpeed) and **FSDP** (PyTorch) — partition the optimizer state, gradients, and parameters across DP ranks. Without this, AdamW's state alone (8× param size in fp32) doesn't fit
- **Activation checkpointing**: recompute activations on the backward pass instead of storing them. Trade compute for memory; standard
- **Mixed precision**: `bf16` activations and weights, `fp32` master weights and optimizer state. **FP8** training (H100+, Hopper) is the 2024–2026 frontier
- **The failure modes of scale**:
  - **Loss spikes** — outlier gradients, bad batches, optimizer instability. Mitigations: gradient clipping, skip-batch on spike, lower-precision-friendly initializations
  - **Silent corruption** — a single GPU returning wrong outputs without erroring. Mitigations: checksums, redundant validation runs
  - **Stragglers** — one slow GPU stalls the synchronous step. Mitigations: bucket-aware scheduling
  - **Network partitions** — multi-day jobs need fault-tolerance and rapid checkpoint resume

### The Chinchilla Equation, Practically

```
Compute budget C ≈ 6 N D     (where N = params, D = tokens)

For a fixed C, the optimum is approximately:
    N* ≈ 0.20 × (C / 6)^0.5
    D* ≈ 5.0  × (C / 6)^0.5
    D* / N*  ≈ 20            (the famous ratio)

But you typically serve a model far more than you train it.
The "inference-aware" optimum is much smaller N and much larger D.
Llama 3 8B was trained on ~15T tokens — about 90× past Chinchilla — because the
inference savings over the model's lifetime swamp the training cost.
```

### Parallelism Cheat Sheet

```
                 Best for ...                   Cost
  DP    Throughput; large batches               All-reduce on every step
  TP    Models too big for one GPU              Hi-bandwidth comm; intra-node only
  PP    Models too big for one node             Pipeline bubble; harder to debug
  EP    MoE routing                             All-to-all between experts
  SP    Very long sequences                     Comm in the attention itself
  ZeRO-3 / FSDP  Optimizer + grad + param shard Comm during step; standard in 2026

Most real training jobs combine 3–5 of these axes.
A frontier-scale run might be DP × TP × PP × EP × SP, with FSDP underneath.
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| Reproduce a mini-Chinchilla plot | Train 7 models from 10M to 500M params with appropriate token counts; plot iso-FLOP loss curves | ⭐⭐⭐⭐ |
| Compute calculator | Implement `6 N D` and check your last run's wall-time and FLOPs against the theoretical | ⭐⭐ |
| FSDP from scratch (toy) | 2 GPUs, shard model weights manually, train a tiny model, verify equivalence to DP | ⭐⭐⭐⭐⭐ |
| Activation checkpointing study | Same model, with and without checkpointing; measure memory and step time | ⭐⭐⭐ |
| BF16 vs FP8 ablation | Train a 100M model in BF16 and FP8 (Transformer Engine if available); compare loss and stability | ⭐⭐⭐⭐⭐ |
| Loss-spike recovery drill | Catch a spike via grad-norm threshold, roll back to last checkpoint, skip the offending batch, resume | ⭐⭐⭐⭐ |
| Multi-node training | 2 nodes × 8 GPUs with `torchrun` + FSDP; reach >70% MFU | ⭐⭐⭐⭐⭐ |

### Sample Code: A Compute Budget in Three Lines

```python
def chinchilla_optimum(compute_flops):
    """Returns (params, tokens) for compute-optimal training."""
    side = (compute_flops / 6) ** 0.5
    return 0.20 * side, 5.0 * side

# Example: 10^22 FLOPs (a small frontier-era pretraining run)
N, D = chinchilla_optimum(1e22)
print(f"Chinchilla-optimal: {N/1e9:.1f}B params, {D/1e12:.1f}T tokens")
# → roughly 8B params, 200B tokens — though most labs would push tokens higher
```

### Key Insight

Scaling laws are the closest thing ML has to a physical theory: they predict the loss of a model that doesn't exist yet from the size of one you've already trained. Believing them — really believing them — is what enables frontier-scale bets. The corollary is that **most architectural innovations don't matter at scale**: their effect is dwarfed by changing N, D, or the data mix. Test new ideas on scaling curves, not on absolute numbers. If it doesn't improve loss-per-FLOP across two or three orders of magnitude, it's noise.

### Resources

- [Kaplan et al. — *Scaling Laws for Neural Language Models* (2020)](https://arxiv.org/abs/2001.08361)
- [Hoffmann et al. — *Chinchilla* (2022)](https://arxiv.org/abs/2203.15556)
- [Rae et al. — *Gopher* (2021)](https://arxiv.org/abs/2112.11446)
- [Narayanan et al. — *Megatron-LM* (2021)](https://arxiv.org/abs/2104.04473)
- [Rajbhandari et al. — *ZeRO* (2019)](https://arxiv.org/abs/1910.02054)
- [PyTorch FSDP docs](https://pytorch.org/docs/stable/fsdp.html)
- [NVIDIA *Transformer Engine* (FP8 training)](https://github.com/NVIDIA/TransformerEngine)
- [HuggingFace *Performance and Scalability* guide](https://huggingface.co/docs/transformers/perf_train_gpu_many)

---

## Phase 5: Post-training — SFT, RLHF, DPO, GRPO, RLVR

A base pretrained model is a brilliant autocomplete. It will continue a prompt with whatever pattern matches; it will not "follow instructions" in the sense users expect. Post-training is the family of techniques that turns autocomplete into an assistant. This phase overlaps heavily with **Phase 9 of the [Reinforcement Learning Guide](../reinforcement-learning/)** — read that for the RL theory, this for the LLM-specific recipe.

### Concepts to Learn

- **The standard post-training pipeline**:
  1. **Pretraining** (Phase 3) → a base model
  2. **Supervised fine-tuning (SFT)** on demonstration data → a helpful but not yet calibrated model
  3. **Preference learning** (RLHF or DPO-family) → an aligned, helpful, harmless model
  4. **Reasoning RL** (RLVR, GRPO) → for math, code, agentic tasks where answers can be checked
- **SFT data**: high-quality instruction-response pairs. Sources include human-written demonstrations, model-generated and human-edited data, distilled outputs from stronger models, and synthetic data with rejection sampling
- **Chat templates**: the structured format the model is fine-tuned on (system / user / assistant turns with explicit boundary tokens). The template is part of the contract — inference must match training byte-for-byte
- **Loss masking in SFT**: only compute loss on assistant tokens, not on system or user tokens. The model should imitate replies, not prompts
- **Reward modeling (RM)**:
  - Collect human pairwise preferences: "given prompt x, response A is preferred to response B"
  - Train a model `R_φ(x, y)` (a transformer with a scalar head) under the Bradley-Terry loss:
    `L = -log σ(R_φ(x, y_w) - R_φ(x, y_l))`
  - RM accuracy is typically 65-75%; humans disagree with each other ~25% of the time on the same pair
- **The RLHF objective** (from PPO/InstructGPT lineage):
  - `J(π) = E_{x,y~π}[ R_φ(x, y) ] - β · KL(π || π_SFT)`
  - The KL term is the most important hyperparameter in the entire stack. Without it, you reward-hack the RM within hundreds of steps
- **Direct Preference Optimization (DPO)**:
  - Algebraic identity: the optimal RLHF policy has closed form in terms of `π_SFT` and `R_φ`. Substitute, rearrange, and the whole RL loop collapses into a *supervised* loss on preference pairs:
    `L_DPO = -log σ( β · [ log π_θ(y_w|x)/π_SFT(y_w|x) − log π_θ(y_l|x)/π_SFT(y_l|x) ] )`
  - No reward model, no rollouts, no PPO. The default in many open-source recipes since 2024
- **DPO-family variants** that fix specific failure modes:
  - **IPO** — replace `log σ` with a squared loss for less overfitting
  - **KTO** — works with single-sided judgments (thumbs up / down) rather than pairs
  - **ORPO** — combines SFT and preference loss in one objective
  - **SimPO** — drops the reference model entirely; length-normalized
- **GRPO (Group Relative Policy Optimization)** — DeepSeek's value-function-free PPO variant:
  - For each prompt, sample a *group* of `G` completions
  - The advantage of each completion is its reward minus the group's mean reward (sometimes normalized by std)
  - Apply PPO-style clipped policy updates with KL to the SFT reference
  - Memory-efficient (no value head), simple to implement, and the backbone of recent reasoning-model training
- **RLVR (RL with Verifiable Rewards)** — when answers can be checked programmatically (math equality, unit tests, formal proofs, regex on a sandbox output), skip the reward model and use the verifier directly. The training signal is exact and unhackable. This is the technique behind R1-style reasoning models
- **The relationship between SFT and RL**: SFT is the warm start; RL is the polish. RL on top of a bad SFT model goes nowhere. Most of the gain in modern post-training comes from better SFT data; RL adds the last 5-15% on specific axes

### The Three Post-Training Recipes, Side-by-Side

```
┌──────────────────────────────────────────────────────────────────────────┐
│  RLHF (the InstructGPT recipe)                                           │
│    SFT  →  Train RM on preferences  →  PPO with KL to SFT                │
│    Pros: well-trodden, works with any reward                             │
│    Cons: complex, fiddly, reward-hacking, RM quality dominates           │
├──────────────────────────────────────────────────────────────────────────┤
│  DPO (the closed-form shortcut)                                          │
│    SFT  →  DPO loss on (chosen, rejected) pairs                          │
│    Pros: one model, no rollouts, no RM, two GPUs is enough               │
│    Cons: harder to mix in new on-policy data, length bias                │
├──────────────────────────────────────────────────────────────────────────┤
│  GRPO / RLVR (the reasoning-model recipe)                                │
│    SFT  →  Generate G completions per prompt  →  Verifier scores them    │
│           →  Group-relative advantage  →  Clipped policy update          │
│    Pros: scales to reasoning tasks, no reward model needed when verifier │
│          exists, the engine behind DeepSeek-R1 and successors            │
│    Cons: needs a verifiable task, expensive (G× more generation)         │
└──────────────────────────────────────────────────────────────────────────┘
```

### The KL Penalty — Why It Matters

```
Without a KL penalty, the policy moves toward whatever the RM scores high,
which after a few hundred steps is gibberish that exploits some quirk of
the reward model:

    "The answer is yes yes yes yes yes ..." with RM score 9.8 / 10
    (real example from early RLHF runs)

The KL penalty
    β · KL( π_θ || π_SFT )
keeps the policy tethered to a known-good distribution. β too high:
the policy doesn't move and you wasted the RLHF compute. β too low:
reward-hacking. β ≈ 0.01–0.1 is the typical range; schedule it.
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| SFT a 1B base model | Take an open base model, fine-tune on Alpaca / UltraChat / Tulu mix using TRL; measure on MT-Bench | ⭐⭐⭐ |
| Loss-masking bug hunt | Run SFT with loss on the full sequence vs. only assistant tokens; observe and explain the difference | ⭐⭐⭐ |
| LoRA / QLoRA | Repeat SFT with LoRA adapters; measure quality and VRAM savings vs. full fine-tune | ⭐⭐⭐ |
| Train a reward model | Use HH-RLHF or UltraFeedback; report pairwise accuracy; analyze where the RM disagrees with humans | ⭐⭐⭐⭐ |
| PPO RLHF loop | Wire SFT + RM + PPO with `trl`; observe KL, reward, and downstream quality; intentionally lower β until reward-hacking | ⭐⭐⭐⭐⭐ |
| DPO from scratch | Implement DPO without `trl`; verify against `trl`'s reference loss | ⭐⭐⭐⭐ |
| GRPO on a math task | Sample G=8 per prompt on GSM8K; verifier = exact answer match; train with GRPO; report accuracy gain over SFT | ⭐⭐⭐⭐⭐ |
| Reward-hacking forensics | Deliberately produce a reward-hacked model; trace the failure to RM, β, or rollout distribution | ⭐⭐⭐⭐ |

### Sample Code: The DPO Loss

```python
def dpo_loss(policy_logits_w, policy_logits_l, ref_logits_w, ref_logits_l,
             labels_w, labels_l, beta=0.1):
    """policy_logits_*: (B, T, V) from the policy being trained.
       ref_logits_*:    (B, T, V) from the frozen SFT reference.
       labels_*:        (B, T) with -100 on masked positions."""
    def seq_logp(logits, labels):
        logp = F.log_softmax(logits, dim=-1)
        mask = (labels != -100)
        # gather the log-prob of each target token, sum over the sequence
        gathered = logp.gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        return (gathered * mask).sum(-1)

    pi_w  = seq_logp(policy_logits_w, labels_w)
    pi_l  = seq_logp(policy_logits_l, labels_l)
    ref_w = seq_logp(ref_logits_w,    labels_w)
    ref_l = seq_logp(ref_logits_l,    labels_l)

    logits = beta * ((pi_w - ref_w) - (pi_l - ref_l))
    return -F.logsigmoid(logits).mean()
```

### Sample Code: A GRPO Step (Sketch)

```python
# For a batch of prompts, sample G completions each, score with a verifier.
prompts, completions, rewards = sample_group_rollouts(policy, prompts, G=8)
# rewards: (B, G); standardize within each prompt's group
adv = (rewards - rewards.mean(dim=1, keepdim=True))
adv = adv / (rewards.std(dim=1, keepdim=True) + 1e-6)        # (B, G)

new_logp = policy.token_logprobs(prompts, completions)        # (B, G, T)
old_logp = old_policy.token_logprobs(prompts, completions).detach()
ratio    = (new_logp - old_logp).exp()                        # token-level ratio

# Token-level clipped objective (PPO style), masked over completion tokens
clip = ratio.clamp(1 - eps, 1 + eps)
pg = -torch.min(ratio * adv[..., None], clip * adv[..., None])
pg = (pg * mask).sum() / mask.sum()

kl = kl_to_reference(policy, ref_policy, prompts, completions)
loss = pg + beta * kl
loss.backward()
```

### Key Insight

The post-training stack does *not* teach the model facts — those came from pretraining. It teaches the model how to **expose** what it already knows in the format users want, and how to **suppress** what it should not say. The implication: if a model "can't do X" after post-training, the answer is rarely more RLHF. It's almost always more or better pretraining data, then a small SFT pass to surface the new ability. RL polish has high leverage but a low ceiling. The bottleneck is upstream.

### Resources

- [Ouyang et al. — *InstructGPT* (2022)](https://arxiv.org/abs/2203.02155) — the original RLHF-for-chat paper
- [Rafailov et al. — *DPO* (2023)](https://arxiv.org/abs/2305.18290)
- [Shao et al. — *GRPO / DeepSeekMath* (2024)](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1 (2025)](https://arxiv.org/abs/2501.12948) — the RLVR-on-reasoning recipe
- [Lambert et al. — *Tülu 3* (2024)](https://arxiv.org/abs/2411.15124) — the open frontier post-training recipe, end to end
- [Lambert — *RLHF Book*](https://rlhfbook.com/) — the modern reference
- [HuggingFace TRL](https://github.com/huggingface/trl) — the canonical post-training library

---

## Phase 6: Reasoning and Inference-Time Compute

Through 2023 the standard assumption was: more pretraining compute → smarter model. The big shift in 2024–2025 was the discovery that a *fixed* model gets much better at hard tasks if it is allowed to **think for longer at inference time** — and that we can train models specifically to use that thinking time well.

### Concepts to Learn

- **Chain of Thought (CoT)** — prompting the model to "think step by step" before answering. The original 2022 result; works because intermediate tokens give the model more compute and serial depth to work with
- **Self-consistency** — sample many CoT solutions; majority-vote the final answer. Big wins on math without any training
- **Best-of-N / rejection sampling** — sample many completions, pick the one scored highest by a verifier or reward model. Cheap, often surprisingly strong
- **Process reward models (PRMs)** — score the model's *reasoning steps*, not just the final answer. Train on stepwise human or synthetic annotations. Used as scorers in tree-search and Best-of-N
- **Outcome reward models (ORMs)** — score only the final answer. Simpler, often almost as good
- **Search at inference time**:
  - **Tree-of-Thoughts** — expand a tree of partial reasoning paths, prune with a PRM
  - **MCTS-style decoding** — apply Monte Carlo Tree Search over reasoning trajectories
  - **Look-ahead / re-rank** — generate, verify, regenerate
- **Inference-time scaling laws** (the 2024 OpenAI o1 / DeepSeek-R1 era): plotting `accuracy vs. tokens-of-thought` produces clean log-linear curves. A model trained to use its thinking time efficiently beats a 10× larger model that isn't
- **The "long-CoT" RL recipe** — the major recent breakthrough:
  - Start with a strong base model and a small SFT pass on reasoning traces
  - Run RLVR (Phase 5) on verifiable tasks (math, code) with the long-CoT prompt
  - The model spontaneously learns to backtrack, verify itself, try multiple approaches, and say "wait, let me reconsider"
  - DeepSeek-R1 (2025) was the first open demonstration of the recipe at scale
- **The trade-off**: long-thinking models are great on reasoning, but slow and expensive at inference. Production systems use a routing layer ("does this query need thinking?") on top
- **Hybrid / thinking modes**: a single model that can answer normally (short) or think first (long). Claude 3.7+, GPT-5, Gemini 2.x and the Qwen3 family all expose this

### The Inference-Time Scaling Curve

```
Accuracy on hard math (AIME-style)
   100% │                               ╭───────  (long-CoT RL model)
        │                          ╭────╯
        │                     ╭────╯
    75% │                ╭────╯
        │           ╭────╯
        │      ╭────╯                  ╭──────  (CoT prompted, no RL)
    50% │ ╭────╯                  ╭────╯
        │─╯                  ╭────╯
        │              ╭─────╯
    25% │         ╭────╯
        │    ╭────╯                ←  direct-answer baseline (~flat)
        │────╯─────────────────────
     0% └────────────────────────────────────────►
           10²       10³       10⁴      10⁵
                Reasoning tokens per problem (log scale)

The two left-most points are the same model; the only thing that changed is
how many tokens of "thinking" we gave it. Training the model with RL on
verifiable rewards (the top curve) shifts the whole frontier up.
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| CoT vs. direct on GSM8K | Compare a base 7B model with and without "let's think step by step"; report accuracy gain | ⭐⭐ |
| Self-consistency sweep | Same model, vary `n_samples ∈ {1, 4, 16, 64}`; plot accuracy vs. cost | ⭐⭐⭐ |
| Best-of-N with a reward model | Reward-model-scored selection on a math benchmark; compare to self-consistency majority | ⭐⭐⭐ |
| Process reward model | Train a small PRM on PRM800K; use it to rescore generations; measure win-rate | ⭐⭐⭐⭐ |
| Tree-of-Thoughts on a logic puzzle | Implement a small tree-search over partial solutions with a PRM heuristic | ⭐⭐⭐⭐ |
| Mini R1 recipe | Take Qwen 7B base; SFT on a small reasoning trace set; GRPO with `is_correct` reward on math problems; observe long-CoT behavior emerging | ⭐⭐⭐⭐⭐ |
| Length-budget controller | Train a model to obey an explicit "think for at most N tokens" instruction; measure quality-vs-budget curve | ⭐⭐⭐⭐⭐ |

### Sample Code: Self-Consistency

```python
from collections import Counter

def self_consistent_answer(prompt, n=16, temperature=0.7):
    answers = []
    for _ in range(n):
        completion = model.generate(prompt, temperature=temperature, max_new_tokens=512)
        # parse the final answer out of the CoT — task-specific
        ans = extract_final_answer(completion)
        if ans is not None:
            answers.append(ans)
    if not answers:
        return None
    return Counter(answers).most_common(1)[0][0]
```

### Key Insight

For two decades, "smarter model" meant "bigger model." The reasoning-RL recipe shifted that: you can also get smarter by giving the model **serial depth at inference time** and training it to use that depth. This is the most important capability development of 2024–2025 and the reason every major lab now ships a "thinking" mode. The bitter lesson holds — scale wins — but scale now has two axes (training compute, inference compute), and the inference axis is much cheaper to push.

### Resources

- [Wei et al. — *Chain of Thought* (2022)](https://arxiv.org/abs/2201.11903)
- [Wang et al. — *Self-Consistency* (2022)](https://arxiv.org/abs/2203.11171)
- [Lightman et al. — *Let's Verify Step by Step* / PRM800K (2023)](https://arxiv.org/abs/2305.20050)
- [Yao et al. — *Tree of Thoughts* (2023)](https://arxiv.org/abs/2305.10601)
- [OpenAI — *Learning to Reason with LLMs* (o1 blog, 2024)](https://openai.com/o1/)
- [DeepSeek-R1 (2025)](https://arxiv.org/abs/2501.12948) — the open recipe
- [Snell et al. — *Scaling LLM Test-Time Compute Optimally* (2024)](https://arxiv.org/abs/2408.03314)

---

## Phase 7: Retrieval, Tools, and Agents

A pretrained LLM is a closed system: it can only return information that was in its weights as of its data cutoff, and it cannot take real actions. Retrieval and tool use turn the LLM into an *interface* to external knowledge and capability. Agents are the natural endpoint: LLMs in a loop, deciding what to do next.

### Concepts to Learn

- **Retrieval-Augmented Generation (RAG)** — fetch relevant documents at query time and stuff them into the context window before generating. The standard pattern for "ask the LLM about *my* data"
- **The RAG pipeline**:
  - **Chunking** — split documents into ~200–800 token pieces with structure-aware boundaries
  - **Embedding** — encode each chunk with a sentence-embedding model (e.g. E5, BGE, GTE, Cohere Embed, OpenAI text-embedding-3)
  - **Indexing** — store vectors in a vector DB (FAISS, Qdrant, pgvector, Pinecone)
  - **Retrieval** — at query time, embed the query, find top-k by cosine similarity
  - **Reranking** — pass candidates through a cross-encoder reranker (much more accurate, much slower) to pick the final k
  - **Generation** — concatenate the retrieved chunks into the prompt and generate
- **Hybrid retrieval**: combine dense (embedding) and sparse (BM25) retrieval. Hybrid almost always beats either alone for retrieval over real documents
- **Long-context vs. RAG**: if your model has a 1M-token context, do you still need RAG? Yes — context is expensive, latency-sensitive, and quality degrades far from the recent past ("needle in a haystack" failures). RAG retrieves the right needles
- **Tool use / function calling**:
  - Model emits a structured call (`{"tool": "search", "query": "..."}`) that an external orchestrator executes
  - Result is returned as the next user message
  - Model continues, possibly with more tool calls
  - Most modern chat APIs have native tool-calling formats (JSON schema, "tools" parameter)
- **The agent loop** — a model in a control flow:
  1. Receive a task
  2. Plan / think
  3. Choose a tool (or "final answer")
  4. Execute, observe result
  5. Repeat until done or budget exhausted
- **Agentic frameworks** — LangGraph, OpenAI Agents SDK, Anthropic's `claude-code` and SWE-agent style harnesses, MCP (Model Context Protocol) for standardized tool exposure. These are scaffolding, not science; the model is doing the work
- **The hard parts of agents**:
  - **Long-horizon reliability** — error rates compound. A 95%-per-step agent succeeds <60% of the time over 10 steps
  - **Cost and latency** — agents can spend tens of thousands of tokens per task
  - **Tool grounding** — the model has to know what a tool does, not just hallucinate calls
  - **Verification and recovery** — what does the agent do when a tool returns an error?
- **Training models for tools and agents**:
  - SFT on tool-call traces
  - RL (often GRPO/RLVR) with task-completion reward — the model learns to plan, recover, and use its tools well
  - Agentic post-training is one of the most active frontiers in 2025–2026

### The Agent Loop, Visualized

```
   User task: "Find the population of every G7 country and put them in a CSV"
        │
        ▼
   ┌─────── Plan ──────────────────────────────────────────────────────┐
   │  Model: "I'll search for each country's population, then format."  │
   └─────────────────────────────────────────────────────────────────────┘
        │
        ▼
   ┌── Choose & call tool ──┐    ┌── Observe ──┐
   │ search("UK population")│ →  │  68.3M ...  │ ───┐
   └────────────────────────┘    └─────────────┘    │
                                                    ▼
   ┌── Choose & call tool ──┐    ┌── Observe ──┐    │
   │ search("US population")│ →  │  335M ...   │ ───┤  (Up to N iterations)
   └────────────────────────┘    └─────────────┘    │
                                                    ▼
                                              ... repeat ...
                                                    │
        ▼                                           │
   ┌────────── Synthesize ──────────────────────────┘
   │ Model: produces a CSV file as the final answer.
   └──────────────────────────────────────────────────

Failure modes: wrong query, stale data, tool error, the model loops forever,
the model "hallucinates" a tool that doesn't exist, the budget runs out.
Robust agents need verification, retry, and budget control at every step.
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| Minimal RAG | Index 1 000 Wikipedia paragraphs with a sentence-embedding model; build a retrieve-then-answer pipeline | ⭐⭐⭐ |
| Chunking ablation | Same corpus, 200 vs. 800 vs. 1 600 token chunks, with and without overlap; measure answer quality | ⭐⭐⭐ |
| Reranker effect | Add a cross-encoder reranker to the pipeline; measure nDCG and end-to-end answer quality | ⭐⭐⭐⭐ |
| Hybrid retrieval | Combine dense + BM25 via reciprocal rank fusion; verify the consistent win | ⭐⭐⭐⭐ |
| Tool-using chatbot | Build a model that can call a calculator and a search tool; evaluate on a synthetic benchmark | ⭐⭐⭐ |
| Agent on a sandbox task | Build a tiny ReAct-style agent on a deterministic task (e.g., spreadsheet edits); evaluate over many seeds | ⭐⭐⭐⭐ |
| SWE-style coding agent | A model + a shell + a file editor; solve a few easy issues from a sample bug benchmark | ⭐⭐⭐⭐⭐ |
| RL fine-tune for tools | GRPO on tool-call success; verifier checks tool output matches expected | ⭐⭐⭐⭐⭐ |

### Sample Code: A Minimal RAG Function

```python
import numpy as np

def retrieve(query, chunks, embeddings, embed_fn, k=5):
    q = embed_fn([query])[0]
    sims = embeddings @ q / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q) + 1e-8)
    top_k = sims.argsort()[-k:][::-1]
    return [chunks[i] for i in top_k]

def rag_answer(query, chunks, embeddings, embed_fn, llm):
    ctx = retrieve(query, chunks, embeddings, embed_fn, k=5)
    prompt = (
        "Use ONLY the following context to answer. If the answer is not in the "
        "context, say you don't know.\n\n"
        + "\n\n---\n\n".join(ctx)
        + f"\n\nQuestion: {query}\nAnswer:"
    )
    return llm.generate(prompt)
```

### Key Insight

RAG and tools are the moment LLMs stop being closed-book test-takers and start being open-ended *problem-solvers*. But every new connection to the outside world is also a new attack surface: prompt injection through retrieved documents, tool-call hallucination, and runaway cost. The shift from "single-turn chatbot" to "multi-step agent" multiplies both capability and risk, and almost every production failure mode shifts from "the model said something wrong" to "the system did something wrong in a loop." Treat the harness as part of the model.

### Resources

- [Lewis et al. — *Retrieval-Augmented Generation* (2020)](https://arxiv.org/abs/2005.11401)
- [Yao et al. — *ReAct* (2023)](https://arxiv.org/abs/2210.03629)
- [Anthropic — *Building Effective Agents* (2024)](https://www.anthropic.com/news/building-effective-agents)
- [LangChain, LlamaIndex, LangGraph](https://www.langchain.com/) — the practical glue
- [Anthropic — *Model Context Protocol (MCP)*](https://modelcontextprotocol.io/)
- [BEIR benchmark](https://github.com/beir-cellar/beir) — retrieval evaluation
- [SWE-Bench](https://www.swebench.com/) — the agent benchmark

---

## Phase 8: Evaluation

Evaluating LLMs is harder than training them. Loss curves keep going down while real-world quality plateaus, or vice versa. Public benchmarks get gamed within months. Human evaluation is expensive and noisy. This is the messy frontier where most teams quietly fail.

### Concepts to Learn

- **Capability vs. alignment**:
  - **Capability evals** — can the model do the task at all? (MMLU, GSM8K, MATH, HumanEval, GPQA, AIME)
  - **Alignment / behavior evals** — does it answer the way we want? (MT-Bench, AlpacaEval, Arena, IFEval)
  - **Safety evals** — does it refuse what it should and not refuse what it shouldn't? (HarmBench, XSTest)
- **The benchmark canon of 2026**:
  - **Knowledge**: MMLU, MMLU-Pro, GPQA, GPQA-Diamond
  - **Math**: GSM8K (saturated), MATH (saturating), AIME, FrontierMath
  - **Code**: HumanEval, MBPP, SWE-Bench (verified), LiveCodeBench
  - **Reasoning**: ARC-AGI, BIG-Bench Hard, MUSR, ZebraLogic
  - **Long context**: RULER, "needle in a haystack" (deprecated, too easy), LongBench
  - **Instruction following**: IFEval (programmable checks), AlpacaEval 2 (LLM-judged)
  - **Agentic**: SWE-Bench Verified, OSWorld, WebArena, GAIA, τ-Bench
  - **Honesty / hallucination**: TruthfulQA, SimpleQA, HaluEval
- **Pitfalls every team eventually hits**:
  - **Train-test contamination** — your pretraining data accidentally includes the eval set. Probably more common than published numbers suggest
  - **Prompt sensitivity** — accuracy can swing 10 points based on the prompt template. Always evaluate with the model's intended chat template
  - **Metric brittleness** — exact-match misses semantically correct answers; BLEU/ROUGE correlate poorly with quality
  - **Saturation** — once an eval is >90%, ranking is noise. Move to harder benchmarks
- **LLM-as-judge**: use a strong LLM to grade open-ended generations. Cheap, fast, surprisingly well-calibrated; biased toward verbose answers and toward the judge's own style
- **Arena-style evaluation**: pairwise comparisons with humans (or strong LLM judges) computing Elo. The de facto frontier ranking in 2024–2026
- **Per-capability decomposition**: a single score hides a model's shape. Always report by capability (knowledge, math, code, instruction-following, refusal calibration) so regressions are visible
- **Eval-as-product** — what the company actually optimizes for. Public benchmarks are a sanity check; the real eval is a custom suite that matches your users' tasks

### Choosing a Decoding Strategy for Eval

```
Task type                       Recommended decoding
─────────────────────────────────────────────────────────────
Deterministic (math, code)      Greedy or temperature=0; pass@1
                                Sometimes temperature 0.6 with pass@k
Open-ended (chat, writing)      temperature 0.7, top-p 0.95
Reasoning with self-consistency Sample n=32+, take majority
LLM-as-judge consumer            temperature 0; deterministic output
Long-context retrieval           Greedy; deterministic by design

⚠ Always report your decoding settings. Two papers' "GSM8K = 82%" can
  be a 10-point real gap once decoding is held fixed.
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| MMLU re-run | Evaluate an open model on MMLU; report by category; verify your number against published | ⭐⭐ |
| Prompt sensitivity sweep | Same model, same eval, five different prompt formats; plot the spread | ⭐⭐⭐ |
| Contamination probe | For a model you trained, search its pretraining data for exact and near-duplicate eval items | ⭐⭐⭐⭐ |
| LLM-as-judge pipeline | Build a judge that compares model A and model B on your own prompts; check inter-judge agreement | ⭐⭐⭐ |
| Capability profile | Run a model through 8 evals across capabilities; produce a one-page radar chart | ⭐⭐⭐ |
| Custom eval | Define a 100-prompt eval that matches your application; collect 3 grades per prompt and analyze noise | ⭐⭐⭐⭐ |
| Arena reproduction | Run a small Elo tournament on 5 open models with an LLM judge; check stability vs. seed and judge model | ⭐⭐⭐⭐ |

### Sample Code: An LLM-as-Judge Loop

```python
JUDGE_PROMPT = """You compare two assistant responses to the same user prompt.
Rate which is better on: helpfulness, accuracy, and clarity.

User: {q}
Response A: {a}
Response B: {b}

Reply with exactly one of: A, B, or TIE."""

def judge_pair(q, a, b, judge_model):
    forward  = judge_model.generate(JUDGE_PROMPT.format(q=q, a=a, b=b)).strip()
    backward = judge_model.generate(JUDGE_PROMPT.format(q=q, a=b, b=a)).strip()
    # Swap to detect position bias; require agreement
    if   forward == "A" and backward == "B": return "A"
    elif forward == "B" and backward == "A": return "B"
    else:                                     return "TIE"
```

### Key Insight

The state of LLM evaluation in 2026 is that **no single number tells you which model is better**. Saturated benchmarks lie; new ones get gamed; LLM judges have biases; humans are slow and inconsistent. The healthiest practice is to maintain **a portfolio of small, targeted evals** that match your actual use case, refresh them constantly, hide them from contamination, and treat any single benchmark result with suspicion. If your team can't articulate exactly what regression an eval would catch, it's noise.

### Resources

- [Liang et al. — *HELM* (2022)](https://arxiv.org/abs/2211.09110) — the holistic eval framework
- [LMSys Chatbot Arena](https://lmarena.ai/) — public pairwise leaderboard
- [`lm-evaluation-harness` (EleutherAI)](https://github.com/EleutherAI/lm-evaluation-harness) — the standard eval runner
- [Zheng et al. — *MT-Bench and LLM-as-a-Judge* (2023)](https://arxiv.org/abs/2306.05685)
- [IFEval (Zhou et al., 2023)](https://arxiv.org/abs/2311.07911)
- [SWE-Bench (Jimenez et al., 2024)](https://arxiv.org/abs/2310.06770)

---

## Phase 9: Efficient Inference and Deployment

Training a model is a fixed cost. Serving it is a forever cost. A modest production system serves 10× more tokens in its first month than the model saw during fine-tuning. This phase is about making that economically and latency-feasibly.

### Concepts to Learn

- **The two-phase nature of LLM inference**:
  - **Prefill** — process the whole prompt in one forward pass; compute-bound, parallel
  - **Decode** — generate one token at a time, each step reading the KV cache; memory-bandwidth-bound, serial
  - These have wildly different performance characteristics and must be optimized separately
- **The KV cache**:
  - For every previous token, cache its keys and values across all layers
  - Avoids recomputing attention over the whole prefix each step
  - Memory cost: `2 · n_layers · n_kv_heads · d_head · seq_len · batch · 2 bytes`
  - For a 70B model at 8k context, multi-tenant serving, the KV cache dominates GPU memory
- **Continuous batching**: dynamically merge requests of different lengths into a single forward pass at each decode step. The single biggest serving throughput win. Implemented in vLLM, TGI, sglang
- **PagedAttention (vLLM)** — manage the KV cache like virtual memory; allocate physical blocks lazily; share blocks across requests with common prefixes. Eliminates fragmentation
- **Prefix caching** — common prompt prefixes (a long system message, retrieved docs) are cached and reused across requests
- **Speculative decoding**:
  - A small "draft" model proposes the next k tokens
  - The big "target" model verifies them in one parallel forward pass
  - Accept the longest verified prefix, reject the rest
  - 2–4× speedup at no quality cost. Standard since 2023
  - **Medusa / EAGLE / self-speculation** — train extra heads on the target model itself instead of a separate draft. Much higher acceptance rates
- **Quantization** — reduce weight/activation precision to save memory and bandwidth:
  - **INT8 weight-only** (GPTQ, AWQ) — near-lossless for weights; activations stay BF16
  - **INT4 weight-only** — small quality cost, big memory win; the standard for consumer-GPU deployment
  - **FP8** (E4M3 / E5M2 on Hopper / Blackwell) — full FP8 forward (weights + activations + KV cache), increasingly the production default for serving
  - **W4A8, W4A4** — increasingly aggressive; quality cost grows below 4-bit weights
- **Distillation** — train a smaller student model to imitate a larger teacher's output distribution. Works well for capabilities the teacher already has; doesn't conjure new abilities
- **Structured / constrained generation** — at decode time, mask out logits that violate a regex / JSON schema / grammar (Outlines, lm-format-enforcer, sglang's regex). Critical for tool-calling reliability
- **Hardware** — H100/H200/B100/B200, MI300X/MI355X, TPU v5/v6/Trillium, Groq/Cerebras/SambaNova for inference. Different sweet spots; choose for your workload's prefill/decode ratio

### The Cost Math You Should Be Able To Do In Your Head

```
For a dense decoder model with N parameters serving in bf16:

  Memory for weights:                2 · N bytes
  Memory per KV-cache token:         2 · n_layers · n_kv_heads · d_head · 2 bytes
  FLOPs per decoded token:           2 · N   (forward only, dominant term)
  Decode is memory-bandwidth bound:
      tok/s ≈ HBM_bandwidth / (2 · N)
      e.g. H100 (~3 TB/s HBM3), 70B model:
           tok/s ≈ 3e12 / (2 · 7e10) ≈ 21 tokens/sec/replica  (single user)

Throughput at scale comes from BATCHING, not from faster per-token decode.
Continuous batching turns this into hundreds of tokens/sec per replica
across many concurrent users.
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| KV cache from scratch | Add a KV cache to a transformer; measure speedup vs. naive recomputation | ⭐⭐⭐ |
| Quantize a 7B model | INT8 with GPTQ; INT4 with AWQ; measure quality (a few benchmarks) and VRAM | ⭐⭐⭐⭐ |
| Speculative decoding | Pair a 1B draft with a 7B target; measure acceptance rate and wall-clock speedup | ⭐⭐⭐⭐ |
| Serve with vLLM | Stand up a vLLM server; load-test with realistic concurrent requests | ⭐⭐⭐ |
| Prefix-cache study | Same model in vLLM, same workload, with prefix cache on/off; report tail latency | ⭐⭐⭐ |
| Constrained JSON generation | Force JSON-schema-valid output with Outlines or sglang; measure overhead | ⭐⭐⭐ |
| Distill 7B → 1B | Distill a strong 7B model into a 1B student on a task domain; report quality retention | ⭐⭐⭐⭐⭐ |
| FP8 serving | Convert a model to FP8 with TransformerEngine; verify quality and speedup | ⭐⭐⭐⭐⭐ |

### Sample Code: A Minimal Speculative-Decoding Loop

```python
def speculative_decode(target, draft, prompt_ids, k=4, max_new_tokens=128):
    out = prompt_ids[:]
    while len(out) < len(prompt_ids) + max_new_tokens:
        # 1. Draft k tokens autoregressively
        draft_ids = draft.greedy(out, k)                  # length k

        # 2. Target verifies in ONE forward pass
        target_logits = target.forward(out + draft_ids)
        target_ids    = target_logits.argmax(-1)[len(out)-1 : len(out)-1+k]

        # 3. Accept the longest matching prefix
        accept = 0
        for i in range(k):
            if draft_ids[i] == target_ids[i]:
                accept += 1
            else:
                break
        # 4. Append accepted tokens + one fresh target token (the corrected one)
        out += draft_ids[:accept]
        if accept < k:
            out.append(int(target_ids[accept]))
        else:
            out.append(int(target.greedy(out, 1)[0]))
    return out
```

### Key Insight

The economics of LLMs are determined at inference, not at training. A model that's 5% better on benchmarks but 2× slower to serve will lose to the slightly-worse one every time outside of a research lab. This is why so much of the post-2023 work has been about inference engineering — KV-cache layouts, paged attention, continuous batching, speculative decoding, quantization, FP8. The next decade of "what wins" will be at least as much about cost-per-token as about capability-per-parameter.

### Resources

- [Kwon et al. — *vLLM / PagedAttention* (2023)](https://arxiv.org/abs/2309.06180)
- [Leviathan et al. — *Speculative Decoding* (2023)](https://arxiv.org/abs/2211.17192)
- [Cai et al. — *Medusa* (2024)](https://arxiv.org/abs/2401.10774)
- [Frantar et al. — *GPTQ* (2022)](https://arxiv.org/abs/2210.17323)
- [Lin et al. — *AWQ* (2023)](https://arxiv.org/abs/2306.00978)
- [Hugging Face *Text Generation Inference* docs](https://huggingface.co/docs/text-generation-inference)
- [sglang](https://github.com/sgl-project/sglang) — fast structured-generation runtime

---

## Phase 10: Safety, Interpretability, and Frontier Topics

Once you've built and shipped a model, you discover the problems that aren't on any benchmark: it lies, it leaks training data, it can be jailbroken, and nobody fully understands why it works. This phase is the open frontier — what people are actively researching now.

### Concepts to Learn

- **Hallucination** — confidently asserting false facts. Caused by the training objective itself (max likelihood rewards fluent continuation, not truth) and not directly fixable by more data. Mitigations: RAG, calibrated abstention training, verifier-in-the-loop, RLVR on factuality
- **Prompt injection** — adversarial text in user input (or retrieved documents, tool outputs, images) that overrides the system instructions. The hardest unsolved security problem in deployed LLMs. The model can't reliably tell "instructions" apart from "data"
- **Jailbreaks** — prompts that bypass safety training. Techniques include role-play scaffolding, low-resource-language translation, gradient-based suffixes (GCG), many-shot jailbreaking. The field has shifted to *defense-in-depth*: don't expect any single layer to be jailbreak-proof
- **Memorization and privacy** — LLMs verbatim-reproduce ~0.1–1% of training data on appropriate prompts. Implications for copyright, PII, and security. Mitigations: deduplication, training-data filtering, differential privacy (expensive and lossy)
- **Reward hacking** — at the RL stage, the model exploits a quirk of the reward signal rather than doing what was intended. The deeper version, **specification gaming**, is endemic to RL and is a major argument for keeping the KL leash short
- **Mechanistic interpretability** — opening the black box one circuit at a time:
  - **Probing classifiers** — does a layer encode "country of capital city"? Train a linear probe and check
  - **Activation patching / causal tracing** — overwrite an activation from a different run, see what the output does. Locates *where* in the network a behavior lives
  - **Sparse autoencoders (SAEs)** — decompose a layer's activations into thousands of monosemantic "features" (e.g., "Golden Gate Bridge", "is a JSON key", "negation in a clause"). The major 2024–2025 line of work, including Anthropic's Golden Gate Claude and feature-steering demos
  - **Circuits-level analysis** — induction heads, attention head specialization, residual-stream features. The toolkit for the question "why did the model do that?"
- **Constitutional AI / AI feedback (RLAIF)** — replace some human preference labels with a "constitution" the AI follows when self-evaluating. Cheaper and often higher quality than human-only feedback at scale
- **Scalable oversight** — when models are smarter than the humans evaluating them, how do you align them? Debate, recursive reward modeling, weak-to-strong generalization. The open theoretical problem of the field
- **The frontier threads to watch in 2026**:
  - **Long-horizon agents** — keeping success rates high over thousands of steps
  - **Continual learning** — updating models without retraining from scratch and without catastrophic forgetting
  - **Multimodal-native models** — vision + audio + action as first-class inputs and outputs (see the multimodal guide)
  - **World models for embodied AI** — pretraining LLMs on grounded action-and-observation streams
  - **Provable / verifiable alignment** — combining formal methods with neural systems
  - **The economics of frontier training** — when does the compute curve break? What happens when one company controls 10× more inference than the next?

### A Map of the Open Problems

```
       Capability ──── lots of headroom; scaling + RL still working
            │
            ▼
       Reliability ─── error rates compound in agents; long-horizon is hard
            │
            ▼
       Honesty ────── hallucination unsolved; calibration partly solved
            │
            ▼
       Safety ─────── jailbreaks unsolved; prompt injection unsolved
            │
            ▼
       Alignment ──── reward hacking real; scalable oversight open
            │
            ▼
       Understanding ─ interpretability is making progress; we still can't
                       fully audit a frontier model
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| Hallucination triage | Build a 100-prompt eval where the model should say "I don't know"; measure refusal rate vs. confident wrong answers | ⭐⭐⭐ |
| Memorization probe | For an open model with known training data, find verbatim regurgitation with prefix prompts | ⭐⭐⭐⭐ |
| Prompt-injection red team | Build a tool-using agent; try 20 injection attacks via retrieved documents; document which defenses help | ⭐⭐⭐⭐ |
| GCG jailbreak | Reproduce a small GCG attack on an aligned open model; understand what the optimization is doing | ⭐⭐⭐⭐⭐ |
| Linear probes for facts | Train probes on hidden states to recover "is this statement true"; measure across layers | ⭐⭐⭐⭐ |
| Tiny SAE | Train a sparse autoencoder on a small model's residual stream; visualize a few features | ⭐⭐⭐⭐⭐ |
| RLAIF on a small task | Replace human preferences with AI judgments; train DPO; compare quality and cost | ⭐⭐⭐⭐ |

### Key Insight

The capabilities frontier is well-understood and well-funded. The hard problems are the *non-capabilities* problems: making the model honest, making it safe under adversarial input, making its reasoning auditable, and making it cooperate with humans we can no longer fully evaluate. None of these are solved. Most of them are not even rigorously formalized. If you want to do research that matters in the late 2020s, this is where to look.

### Resources

- [Bai et al. — *Constitutional AI* (2022)](https://arxiv.org/abs/2212.08073)
- [Zou et al. — *GCG / Universal Jailbreaks* (2023)](https://arxiv.org/abs/2307.15043)
- [Carlini et al. — *Extracting Training Data* (2021)](https://arxiv.org/abs/2012.07805)
- [Anthropic — *Scaling Monosemanticity* (2024)](https://transformer-circuits.pub/2024/scaling-monosemanticity/)
- [Anthropic Interpretability publications](https://transformer-circuits.pub/)
- [Greenblatt et al. — *Alignment Faking* (2024)](https://arxiv.org/abs/2412.14093)
- [Hendrycks et al. — *Aligning AI with Human Values* (survey)](https://arxiv.org/abs/2109.13916)

---

## Suggested Timeline

| Phase | Duration | Outcome |
|-------|----------|---------|
| 0. Prerequisites | 0–2 weeks | PyTorch + HF stack installed; Karpathy's GPT video watched |
| 1. Tokenization | 3–5 days | Trained a BPE; understand chat templates and their pitfalls |
| 2. Transformer architecture | 1–2 weeks | nanoGPT typed from scratch; RoPE, GQA, SwiGLU understood |
| 3. Pretraining | 2–3 weeks | 100M model trained end-to-end; understand the data pipeline |
| 4. Scaling and infra | 2 weeks | Chinchilla math is reflex; FSDP / multi-GPU run done |
| 5. Post-training | 3 weeks | SFT + DPO + a GRPO loop run; reward-hacking diagnosed once |
| 6. Reasoning | 1–2 weeks | Self-consistency, Best-of-N, mini long-CoT RL working |
| 7. Retrieval and agents | 2 weeks | RAG + reranker shipped; small tool-using agent built |
| 8. Evaluation | 1–2 weeks | A 6-eval portfolio reporting honestly with seeds |
| 9. Inference / deployment | 2 weeks | vLLM in prod; spec decode + quantization measured |
| 10. Safety + frontier | Ongoing | Picked one thread (interpretability, agents, safety) and going deep |

**Total to "comfortable practitioner":** ~3–4 months of focused study, longer if combined with real projects (recommended). To "research-comfortable on one frontier thread": 6–12 months beyond that.

---

## Key Advice

1. **Watch Karpathy's GPT video before reading anything else.** It is the single highest-bandwidth introduction to the entire field. Pause and re-implement.
2. **Implement from scratch at least once.** nanoGPT, a BPE, attention, RoPE, a DPO loss. Not "use the library version"; from scratch. The exercise teaches you which knobs are real and which are accidents.
3. **The tokenizer is part of the model.** Half your "weird model behaviors" are tokenizer issues. Always inspect tokens, not strings.
4. **Match training and inference exactly.** Chat templates, special tokens, EOS handling, BOS prepending — all must be byte-identical between training and serving. Most "the model is dumb in prod" stories trace here.
5. **Data is the model.** If your fine-tuned model isn't doing X, the answer is almost never "more epochs" or "different learning rate" — it's "the data doesn't have X in the form you want."
6. **Seed and run multiple times.** Especially for RL and small fine-tuning. A single curve tells you almost nothing.
7. **Always run a strong baseline.** Few-shot prompting of the base model. SFT-only. A simple retrieval pipeline. Many "RLHF works" claims collapse when compared against a well-tuned simpler approach.
8. **Evaluate at every step.** Don't wait until the end. Build a small held-out eval and check it after SFT, after DPO, after RL. Regressions are cheap to fix when caught early and expensive otherwise.
9. **Profile before scaling.** Most pretraining and inference systems spend more time on data movement and synchronization than on the math. Always profile.
10. **Read CleanRL, nanoGPT, and TRL source.** These three repositories teach more about modern LLM engineering than any textbook.
11. **In the post-training era, ask: do I have a verifier?** If yes, use RLVR / GRPO. If no, decide between DPO (cheap, supervised) and full RLHF (expensive, flexible). The decision dominates everything downstream.
12. **Track inference cost from day one.** A model is a *product* once it ships. Tokens-per-second-per-dollar will determine whether it survives contact with users.

---

## Common Pitfalls

- ❌ Training chat template ≠ inference chat template → mysterious quality drop in deployment
- ❌ Loss computed on user / system tokens during SFT → model imitates user turns
- ❌ Forgetting `attention_mask` in batched generation → garbage on the padded side
- ❌ Wrong BOS / EOS handling → model never stops, or stops immediately
- ❌ Training in `fp16` instead of `bf16` → silent overflow in attention logits
- ❌ Skipping deduplication of pretraining data → model memorizes and hallucinates more
- ❌ Tokenizer trained on a different corpus than the model → poor compression, weird tokens
- ❌ KL penalty too low in RLHF → reward hacking within hundreds of steps
- ❌ KL penalty too high → policy doesn't move, RLHF compute wasted
- ❌ Reward model overfit / out-of-distribution → policy moves in a direction the RM "scored high" but humans don't like
- ❌ Reporting a single seed in a paper → results don't replicate
- ❌ Benchmark contamination → publicly reporting numbers your data has memorized
- ❌ Treating a 70-on-MMLU model and another 70-on-MMLU model as equivalent → they probably differ by 10+ points on what you actually care about
- ❌ Forgetting the `<|im_end|>` token in fine-tuning data → the model runs on forever in chat
- ❌ Building an agent before measuring per-step reliability → 5% per-step error becomes 60% task-level failure over 10 steps
- ❌ Optimizing for training perplexity and ignoring real eval → loss goes down, users complain
- ❌ Quantizing a model and not re-evaluating → silent capability regression

---

## Additional Resources

### Books
- [Jurafsky & Martin — *Speech and Language Processing* (3rd ed., draft)](https://web.stanford.edu/~jurafsky/slp3/) — the NLP textbook
- [Lambert — *Reinforcement Learning from Human Feedback*](https://rlhfbook.com/) — post-training reference
- [Raschka — *Build a Large Language Model (From Scratch)*](https://www.manning.com/books/build-a-large-language-model-from-scratch) — practical end-to-end
- [Bishop & Bishop — *Deep Learning: Foundations and Concepts*](https://www.bishopbook.com/) — for the math-minded

### Courses
- [Stanford CS336 — *Language Models from Scratch*](https://stanford-cs336.github.io/spring2024/) — the modern definitive course
- [Stanford CS224N — *NLP with Deep Learning*](https://web.stanford.edu/class/cs224n/) — the canonical NLP course
- [Karpathy — *Neural Networks: Zero to Hero* (YouTube)](https://karpathy.ai/zero-to-hero.html) — the best intro on the internet
- [Hugging Face NLP and LLM Courses](https://huggingface.co/learn) — practical, free

### Code You Should Read
- [`nanoGPT`](https://github.com/karpathy/nanoGPT) — single-file GPT-2 reproduction
- [`llm.c`](https://github.com/karpathy/llm.c) — raw-C pretraining, fast and educational
- [`transformers` (Hugging Face)](https://github.com/huggingface/transformers) — the encyclopedia
- [`trl`](https://github.com/huggingface/trl) — RLHF / DPO / GRPO reference
- [`vllm`](https://github.com/vllm-project/vllm) — production inference engine
- [`sglang`](https://github.com/sgl-project/sglang) — structured-generation runtime
- [`gpt-neox`](https://github.com/EleutherAI/gpt-neox) — open large-scale training stack
- [`OLMo`](https://github.com/allenai/OLMo) — fully open training pipeline

### Papers Everyone Cites
- [Vaswani et al. — *Attention Is All You Need* (2017)](https://arxiv.org/abs/1706.03762)
- [Radford et al. — GPT-2 (2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Brown et al. — GPT-3 (2020)](https://arxiv.org/abs/2005.14165)
- [Hoffmann et al. — Chinchilla (2022)](https://arxiv.org/abs/2203.15556)
- [Ouyang et al. — InstructGPT (2022)](https://arxiv.org/abs/2203.02155)
- [Touvron et al. — Llama 1 / Llama 2 (2023)](https://arxiv.org/abs/2307.09288)
- [Llama 3 paper (2024)](https://arxiv.org/abs/2407.21783)
- [DeepSeek-V3 (2024)](https://arxiv.org/abs/2412.19437)
- [DeepSeek-R1 (2025)](https://arxiv.org/abs/2501.12948)

### Communities
- [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/) — the most active practitioner forum
- [EleutherAI Discord](https://www.eleuther.ai/) — open-source frontier work
- [GPU Mode Discord](https://github.com/gpu-mode/lectures) — the inference / kernel side
- [LMSys Discord](https://lmarena.ai/) — eval and serving

### Talks Worth Watching
- [Karpathy — *Intro to LLMs* (1 hour)](https://www.youtube.com/watch?v=zjkBMFhNj_g)
- [Karpathy — *State of GPT*](https://www.youtube.com/watch?v=bZQun8Y4L2A)
- [Yann LeCun — *Self-supervised learning and world models*](https://www.youtube.com/watch?v=8m4hO8C8YxA)
- [Jared Kaplan — *Scaling laws*](https://www.youtube.com/watch?v=h1QO0lFmwzU)

---

## Quick Start Checklist

- [ ] Have watched Karpathy's *Let's build GPT* end to end
- [ ] Have implemented a BPE tokenizer and understand why `" Paris"` ≠ `"Paris"`
- [ ] Have implemented multi-head causal attention with RoPE from scratch
- [ ] Have trained a 10–100M parameter model on real text and watched the loss curve behave
- [ ] Can compute, in your head, the FLOPs to train an `N`-parameter model on `D` tokens
- [ ] Know what Chinchilla's `~20 tokens / param` ratio means and when to ignore it
- [ ] Have done SFT on an open base model with correct loss masking
- [ ] Have implemented DPO and verified the loss against `trl`'s reference
- [ ] Have run a small GRPO or RLVR loop on a verifiable task
- [ ] Have built a RAG pipeline with a reranker and measured retrieval quality
- [ ] Have built a tool-using or agentic loop and measured per-step reliability
- [ ] Have served a model with vLLM and measured tokens/sec vs. concurrency
- [ ] Have quantized a model to INT4 and re-run your evals
- [ ] Have set up spec-decoding and measured the speedup
- [ ] Have diagnosed at least one reward-hacking incident or chat-template bug
- [ ] Can read a contemporary LLM paper and explain which design choices solve which problems

---

## License

MIT License. See the [LICENSE](https://github.com/25621/ai-learning-guides/blob/main/LICENSE) file for details.
