# Per-Channel vs Per-Tensor

## Key Insight

Selecting the right [quantization](/shared/glossary/#quantization) granularity determines the balance between model accuracy and hardware efficiency. Comparing [per-channel quantization](/shared/glossary/#per-channel-quantization)—which assigns a unique scaling factor to each filter or channel—with [per-tensor quantization](/shared/glossary/#per-tensor-quantization) reveals that finer granularity preserves quality by accommodating varying range scales across channels. However, the extra scaling operations can introduce minor compute overhead, making it essential to choose the appropriate method based on target hardware capabilities.
