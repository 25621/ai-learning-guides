# Greedy Speculative Decoding

---

> A small model guesses the next few words; the big model checks them all in a single glance.

---

## Key Insight

This project pairs a small [draft model](/shared/glossary/#draft-model) that cheaply guesses the next few tokens with a big [target model](/shared/glossary/#target-model) that checks all those guesses in one [decode](/shared/glossary/#decode) pass — keeping every token the target agrees with (its [argmax](/shared/glossary/#argmax)) and measuring the [acceptance rate](/shared/glossary/#acceptance-rate) and real speedup. This is [speculative decoding](/shared/glossary/#speculative-decoding) in its simplest, greedy form.

## Why This Matters

Because [decode](/shared/glossary/#decode) spends most of its time waiting on memory, the target can verify several guesses for almost the price of generating one token — so when the draft guesses well you get 2–4× faster generation with identical output. Building the greedy version first makes the trickier sampling version far easier to reason about.
