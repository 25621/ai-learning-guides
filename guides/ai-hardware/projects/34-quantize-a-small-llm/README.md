# Quantize a Small LLM

## Key Insight

Post-training [quantization](/shared/glossary/#quantization) allows developers to compress large language models to [int4](/shared/glossary/#int4) formats using techniques like [GPTQ](/shared/glossary/#gptq). Evaluating the quantized model's accuracy on benchmarks like [MMLU](/shared/glossary/#mmlu) and monitoring [perplexity](/shared/glossary/#perplexity) changes shows how aggressive bit-width reduction affects language understanding. Selecting the right balance between model compression and task quality is key to making models run on resource-constrained hardware.
