# Custom Grammar

---

> Write the rules of a valid answer once, and the model can never break them.

---

## Key Insight

This project builds a regex-based grammar for a domain-specific output format (for example, valid SQL) and enforces it during [decode](/shared/glossary/#decode): at each step the grammar decides which next tokens are legal, and the rest are masked out of the [logits](/shared/glossary/#logits) before sampling. It is [constrained generation](/shared/glossary/#constrained-generation) applied to a format you define yourself.

## Why This Matters

Many production outputs must follow a strict shape — a query language, a config file, a command for another system — and a single malformed character makes the whole thing unusable. A custom grammar guarantees the model stays inside the lines, turning a flaky text generator into a dependable structured-output component.
