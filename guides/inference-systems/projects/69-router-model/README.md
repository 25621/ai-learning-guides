# Router Model

---

> Most questions don't need your biggest model — a good [router](/shared/glossary/#router-model) is the cheapest speedup in the whole stack.

---

## Key Insight

This project builds a tiny [router model](/shared/glossary/#router-model) ([trained or just prompted](/shared/glossary/#trained-or-just-prompted)) that looks at each incoming request and decides whether to send it down a fast 1B "easy path" or escalate it to a slow, expensive 70B "hard path." You then measure the two things that matter: answer quality and [cost per token](/shared/glossary/#cost-per-million-tokens), to see how much you save without users noticing.

## Why This Matters

In real traffic, the large majority of queries are easy and never needed a frontier-size [LLM](/shared/glossary/#llm) at all. A router captures that fact directly — paying the big model's cost only for the requests that actually require it — which is often a larger cost win than any kernel or quantization trick, for a fraction of the engineering effort.
