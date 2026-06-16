# KV-Cache Quantization

## Key Insight

[Quantizing](/shared/glossary/#quantization) the [KV cache](/shared/glossary/#kv-cache) to [int8](/shared/glossary/#int8) addresses the primary memory bottleneck in long-context language model serving. Reducing the precision of stored attention keys and values halves the memory footprint of the cache, allowing for larger serving batches and significantly higher throughput. However, because keys and values have distinct [activation ranges](/shared/glossary/#activation-range), applying separate [scaling factors](/shared/glossary/#scaling-factor) is crucial to maintain generation accuracy over long contexts.
