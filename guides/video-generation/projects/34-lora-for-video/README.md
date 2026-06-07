# LoRA for Video

## Key Insight

[LoRA (Low-Rank Adaptation)](/shared/glossary/#lora) fine-tunes a giant model cheaply by freezing all its original [weights](/shared/glossary/#weights) and learning only a tiny pair of [low-rank](/shared/glossary/#low-rank) matrices that nudge the output — small enough to train on a handful of examples and to swap in and out like sticky notes. This project trains such an adapter on roughly 50 clips of one specific visual style or character, teaching a [text-to-video](/shared/glossary/#t2v) model to reproduce that look on demand without retraining the whole network. Because the adapter is only a few megabytes, you can keep a separate LoRA per style and load whichever one a given shot needs — the same workflow that made LoRAs the dominant way to personalize image models.
