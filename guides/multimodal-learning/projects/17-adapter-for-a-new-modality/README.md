# Adapter for a New Modality

## Key Insight

This project shows how cheaply you can teach an existing model a brand-new sense: bolt a small [adapter](/shared/glossary/#adapter) onto a [frozen](/shared/glossary/#frozen) [CLIP](/shared/glossary/#clip) image encoder so it can read [depth maps](/shared/glossary/#depth-map) — grayscale images of how far away each pixel is — even though CLIP never trained on them. Only the adapter's handful of [weights](/shared/glossary/#weights) learn; CLIP's millions stay put, so you reuse all its hard-won visual knowledge instead of paying to retrain it, and you protect that knowledge from being overwritten by your small depth dataset. The takeaway is that adding a [modality](/shared/glossary/#modality) need not mean a new model — a tiny trainable bridge into a frozen backbone is often enough, the same money-saving idea behind [LoRA](/shared/glossary/#lora).
