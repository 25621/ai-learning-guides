# Static Quantization (PTQ)

---

> Measure the activations once, fix the scales, and skip the per-batch guesswork.

---

## Key Insight

[Static quantization (PTQ)](/shared/glossary/#static-quantization-ptq) converts both [weights](/shared/glossary/#weights) and [activations](/shared/glossary/#activations) to [int8](/shared/glossary/#int8) ahead of time. To pick the right activation scales, it first runs a few sample batches through the model — a step called [calibration](/shared/glossary/#calibration).

## Why This Matters

Because the scales are fixed before serving, static quantization avoids the per-batch overhead of [dynamic quantization](/shared/glossary/#dynamic-quantization) and is usually faster, especially for a [CNN](/shared/glossary/#cnn). The cost is the extra calibration step and a little more accuracy tuning.
