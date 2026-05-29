# Prompt Sensitivity Sweep

---

> Same model, same questions, five wordings — five different scores.

---

## Key Insight

This project holds the model and the [benchmark](/shared/glossary/#benchmark) fixed and changes only how the question is phrased, running a [sweep](/shared/glossary/#sweep) over five prompt formats and plotting how far the accuracy moves.

## Why This Matters

If rewording a prompt swings the score by ten points, then comparing two models is meaningless unless both are tested with the same template — a trap that has produced many bogus "model A beats model B" claims.
