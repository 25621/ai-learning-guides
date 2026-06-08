# Watermarking

## Key Insight

As AI video becomes hard to tell apart from real footage, being able to prove a clip was machine-made matters for fighting [deepfakes](/shared/glossary/#deepfake) and misinformation. [Watermarking](/shared/glossary/#watermarking) hides an invisible, machine-detectable signal inside the output — either stamped into the pixels after generation or baked into the model's own sampling, the way [SynthID](/shared/glossary/#synthid) does — that a matching detector can later read back out without the mark ever being visible to a human. This project adds such a watermark to a model's outputs and verifies that a detector flags them while leaving real footage unflagged, confronting the core tension head-on: a mark strong enough to survive cropping and compression is harder to keep imperceptible.
