# KV-Cache Memory Math

## Key Insight

Serving language models requires understanding how memory consumption scales with batch size and context length. During the [decode](/shared/glossary/#decode) phase, storing the [KV cache](/shared/glossary/#kv-cache) for all active sequences dominates the GPU's memory footprint. Performing the memory arithmetic for a given model architecture, such as a 8B parameter model, reveals how keys and values scale linearly, which is essential to prevent out-of-memory errors and maximize serving throughput.
