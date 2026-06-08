# Consistency-Model Distillation

## Key Insight

A [diffusion model](/shared/glossary/#diffusion-model) needs around 50 denoising steps to produce a clip, which is far too slow for production use; [distillation](/shared/glossary/#distillation) trains a fast "student" to reproduce that 50-step "teacher" in only about 4 steps. A [consistency model](/shared/glossary/#consistency-model) is the student design that makes this possible: it is trained so that *every* point along the noisy-to-clean denoising path points straight at the same finished clip, so one or a few jumps replace the long walk. This project performs that distillation and measures the bargain directly — the speed you gain (50 ÷ 4 ≈ 12× fewer steps) against the quality you lose. Where the [real-time latency hunt](../44-real-time-latency-hunt/README.md) project chases raw milliseconds-per-frame for interactivity, this one zeroes in on the speed-versus-quality trade of the distillation step itself.
