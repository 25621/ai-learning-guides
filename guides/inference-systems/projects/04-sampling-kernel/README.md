# Sampling Kernel

---

> Every decode step ends with the same small pile of math — and on a tight inner loop, "small" still adds up.

---

## Key Insight

This project implements [temperature](/shared/glossary/#temperature), [top-k](/shared/glossary/#top-k), and [top-p](/shared/glossary/#top-p) [sampling](/shared/glossary/#sampling) as a single pass over the model's [logits](/shared/glossary/#logits), then profiles it against the naive composition of `torch.topk` plus [`torch.multinomial`](/shared/glossary/#torchmultinomial). The three transforms all reshape the same probability distribution before the random draw, so a careful [kernel](/shared/glossary/#kernel) can fuse them and call the GPU once instead of three times per token.

## A Concrete Example

Imagine generating a 500-token reply. The sampling math — temperature, top-k, top-p, then the random draw — runs once per token, so it fires 500 times for this *single* request. Suppose those steps are three separate GPU [kernel](/shared/glossary/#kernel) launches, and each launch costs ~5 microseconds of fixed overhead before any real work begins:

- **Unfused:** `500 tokens × 3 launches × 5µs = 7.5 ms` spent *just starting kernels* — invisible in a one-off microbenchmark, but a measurable slice of [ITL](/shared/glossary/#itl--tpot) once it repeats hundreds of times.
- **Fused into one kernel:** `500 × 1 × 5µs = 2.5 ms`.

That's a 5 ms saving per request that the user never has to wait for — and the effect only grows with longer replies and more concurrent requests. A cost that "looked harmless in isolation" turns into real latency precisely because it sits on the hot path that repeats every token.

## Why This Matters

[Decode](/shared/glossary/#decode) runs the sampling path *every* generated token, often hundreds of times per request, so a [kernel](/shared/glossary/#kernel) launch that looked harmless in isolation can become a real share of [ITL](/shared/glossary/#itl--tpot). Production engines like [vLLM](/shared/glossary/#vllm) fuse the whole sampling pipeline into one kernel for exactly this reason — and seeing the speedup yourself trains the instinct to look for the same pattern elsewhere.
