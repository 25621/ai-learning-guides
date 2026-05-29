# Sampling Kernel

---

> Every decode step ends with the same small pile of math — and on a tight inner loop, "small" still adds up.

---

## Key Insight

This project implements [temperature](/shared/glossary/#temperature), [top-k](/shared/glossary/#top-k), and [top-p](/shared/glossary/#top-p) [sampling](/shared/glossary/#sampling) as a single pass over the model's [logits](/shared/glossary/#logits), then profiles it against the naive composition of `torch.topk` plus `torch.multinomial`. The three transforms all reshape the same probability distribution before the random draw, so a careful kernel can fuse them and call the GPU once instead of three times per token.

## Why This Matters

[Decode](/shared/glossary/#decode) runs the sampling path *every* generated token, often hundreds of times per request, so a kernel launch that looked harmless in isolation can become a real share of [ITL](/shared/glossary/#itl--tpot). Production engines like [vLLM](/shared/glossary/#vllm) fuse the whole sampling pipeline into one kernel for exactly this reason — and seeing the speedup yourself trains the instinct to look for the same pattern elsewhere.
