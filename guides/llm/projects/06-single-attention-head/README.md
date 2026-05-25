# Single Attention Head

---

> Attention is a weighted average where each token decides how much to listen to every earlier token.

---

## Key Insight

A single [attention](/shared/glossary/#attention) head computes `softmax(QKᵀ/√d)·V`: it scores how well each token's query matches every token's key, turns those scores into weights with [softmax](/shared/glossary/#softmax), and mixes the values accordingly. A [causal mask](/shared/glossary/#causal-mask) hides future positions so a token can only attend to itself and what came before.

## Why This Matters

This one operation is the heart of every [transformer](/shared/glossary/#transformer). Building it by hand and checking it against PyTorch's `F.scaled_dot_product_attention` turns the formula every modern LLM relies on from a mystery into something you can write from memory.
