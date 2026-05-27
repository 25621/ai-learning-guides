# RoPE from Scratch

---

> Encode position by spinning each token's vectors — and the angle between two tokens becomes their distance.

---

## Key Insight

[RoPE](/shared/glossary/#rope) encodes a token's position by *rotating* its query and key vectors by an angle proportional to that position. Because rotations compose, the [attention](/shared/glossary/#attention) score between two tokens ends up depending only on their relative distance, not their absolute positions.

## Why This Matters

RoPE is the default positional scheme in Llama, Mistral, Qwen, and DeepSeek. Implementing it — including the [half-rotation](/shared/glossary/#half-rotation) trick — and confirming that `⟨q, k⟩` depends only on relative position makes the most widely used position embedding concrete rather than magical.
