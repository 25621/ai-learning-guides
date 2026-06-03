# Negative Prompts Study

## Key Insight

A [negative prompt](/shared/glossary/#negative-prompt) is a second prompt naming what you *don't* want ("ugly, blurry, low quality"), and it costs nothing extra because it simply replaces the blank unconditional input that [classifier-free guidance (CFG)](/shared/glossary/#cfg-classifier-free-guidance) already runs — the model is pushed away from the negative prompt's prediction and toward your real one. This project measures whether that actually helps by generating 50 prompts with and without negatives and scoring both sets: [FID](/shared/glossary/#fid) for realism, a [CLIP](/shared/glossary/#clip) score for prompt adherence, and an aesthetic score for human-judged appeal. The result builds honest intuition for when negatives genuinely improve images versus when they merely shift the style.
