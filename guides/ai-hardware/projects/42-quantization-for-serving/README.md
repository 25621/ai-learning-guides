# Quantization for Serving

## Key Insight

Deploying large language models for production requires balancing hardware throughput with model output quality. By converting model weights to lower-precision numeric formats like [FP8](/shared/glossary/#fp8) or [int4](/shared/glossary/#int4) via [post-training quantization (PTQ)](/shared/glossary/#ptq--qat), this project compares serving efficiency against native [bfloat16](/shared/glossary/#bfloat16) execution. Measuring throughput and perplexity under different quantization strategies illustrates how bit-width reduction directly lowers memory bandwidth pressure during the memory-bound decode phase.
