# Classifier-Free Guidance

## Key Insight

[Classifier-free guidance (CFG)](/shared/glossary/#cfg-classifier-free-guidance) is the single biggest practical trick in modern text-to-image generation: it makes samples follow the prompt far more closely, and unlike the earlier [classifier guidance](/shared/glossary/#classifier-guidance) it needs no separate classifier at all. The recipe is to train *one* model that sometimes sees the real condition and sometimes a "null" (empty) condition — done by randomly dropping the condition about 10% of the time during training — so the same network learns both the conditional and unconditional prediction. At sampling time you run it twice and extrapolate: start from the unconditional prediction and push *away* from it and *toward* the conditional one, with a "guidance scale" controlling how hard you push. This sharpens the output toward modes the prompt strongly implies, but turn the scale too high and you get oversaturated, less diverse images — the trade-off this project maps by sweeping the scale from 1.0 to 12.0.

## FAQ

**Is the "real condition" something you supply directly?**
Yes. The real condition is whatever you feed in as the prompt (e.g., `"a cat"`). At sampling the model runs twice — once on your prompt and once on the "null" condition — but you only supply the prompt. The null is a learned empty token (an empty string or a fixed null embedding), so the unconditional pass is produced automatically by the same network. The guidance formula `pred = uncond + scale * (cond - uncond)` uses both, but only `cond` is your input. In practice the two passes are usually batched into a single forward pass (*CFG fusion*) rather than run separately.

**Why drop the condition ~10% of the time, not 50/50?**
The two passes are not equally important. The conditional prediction is the actual product and must be precise; the unconditional prediction only serves as a baseline to extrapolate away from. A 50/50 split would spend half the model's capacity on the less-important unconditional signal, weakening prompt adherence. Because both share one network, the unconditional pass reaches "good-enough baseline" quality with far less exposure. Empirically (Ho & Salimans, 2022), drop rates around 0.1–0.2 gave the best FID/IS, while 0.5 was worse and very low rates leave the baseline too inaccurate — making ~10% the sweet spot.

**For a complex scene, must I input every attribute separately?**
No. You write one natural sentence (e.g., `"A Golden Retriever wearing an apron and seriously making latte art in front of an espresso machine, fluffy fur, warm lighting, high-quality photo"`). The text encoder turns the whole sentence into a single conditioning embedding, so the conditional pass looks the same whether the prompt is one word or a long phrase — complexity does not require extra passes or split inputs. Unspecified details (background, lighting, exact pose) are sampled from the learned distribution; more detail simply narrows the output. The real limitation is *attribute binding* — colors or traits attaching to the wrong object when several are described — which is a model/text-encoder limit, not something fixed by splitting the prompt. When text alone is insufficient, extra condition channels like ControlNet (pose/edge/depth maps) or IP-Adapter (reference image) are added.

**How is training on diverse images automated?**
The automation lives almost entirely in the data pipeline, not the model code. The core idea is generating each image's condition (caption) without manual labeling:

1. **Collect** (image, text) pairs by scraping the web, using existing `alt`-text as the first-pass caption (e.g., LAION-5B).
2. **Auto-filter** with quality gates — [CLIP](/shared/glossary/#clip) score for image–text match, aesthetic score, plus resolution/NSFW/watermark filters and deduplication.
3. **Recaption** with a VLM (BLIP-2, CogVLM, LLaVA) to replace sparse alt-text with dense synthetic captions (the DALL·E 3 / SD3 approach), usually mixing synthetic and original text. This is the step that most improves condition quality.
4. **Precompute and cache** — aspect-ratio bucketing, [VAE](/shared/glossary/#vae) latents, and frozen text embeddings, so they aren't recomputed every epoch.
5. **Stream** from sharded `.tar` archives via [WebDataset](/shared/glossary/#webdataset) to avoid per-file I/O at billion-sample scale.
6. **Automate the training loop** — condition dropout (~10% null) is handled in the DataLoader collate step, alongside AMP, gradient checkpointing, FSDP/ZeRO sharding, periodic checkpointing, logging, and automatic FID evaluation.

The "alt-text → filter → VLM recaption" stages are what remove manual labeling; the rest (latent caching, WebDataset, distributed training) is infrastructure for running it at scale.