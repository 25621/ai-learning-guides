# Speculative Decoding

## Key Insight

Language model inference is often bottlenecked by memory bandwidth during the token generation phase. [Speculative decoding](/shared/glossary/#speculative-decoding) bypasses this constraint by utilizing a small, fast draft model to propose a sequence of candidate tokens, which the larger target model verifies in parallel. Because a single forward pass of the target model can evaluate multiple tokens simultaneously, this technique significantly improves generation speed without changing the final output distribution.
