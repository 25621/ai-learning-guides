# Reproduce a Leaderboard Result

## Key Insight

A single number on a [leaderboard](/shared/glossary/#leaderboard) hides a hundred quiet choices — the exact prompt template, how multiple-choice answers are parsed, the image resolution, whether a few example questions were shown first. Picking a published [MMBench](/shared/glossary/#mmbench) score and trying to reproduce it almost always leaves a gap, and chasing that gap is the fastest way to learn where multimodal evaluation silently leaks or inflates accuracy. A *wide* gap usually means your prompt or parser differs from the paper's; a suspiciously *small* gap (or a score that beats the paper) can mean the [benchmark](/shared/glossary/#benchmark)'s questions leaked into the model's training data ([contamination](/shared/glossary/#contamination)). The discipline you build here — pinning every setting and documenting exactly why your number differs — is what separates a trustworthy result from a cherry-picked one.
