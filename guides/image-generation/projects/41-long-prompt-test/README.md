# Long-Prompt Test

## Key Insight

The text encoder a model uses decides how well it understands a long, detailed prompt. The original [Stable Diffusion](/shared/glossary/#stable-diffusion) used [CLIP](/shared/glossary/#clip)'s text encoder (the "CLIP-L" variant), which was trained only to match images to short captions and tops out around 77 tokens — so it tends to drop or blur details in a paragraph-long prompt. [T5](/shared/glossary/#t5), trained on general language tasks, tracks word order and long-range detail far better, which is why newer models feed it through [cross-attention](/shared/glossary/#cross-attention) for stronger adherence. Running the same 200-token prompts through each and comparing the images makes the gap visible: T5 follows compositional, multi-clause descriptions that CLIP-L quietly collapses.
