# CPU/NVMe Offload

---

> When the GPU runs out of room, push the rarely-touched cache down to cheaper memory instead of dropping the request.

---

## Key Insight

This project adds a second tier to the [KV cache](/shared/glossary/#kv-cache): when GPU [HBM](/shared/glossary/#hbm) fills up, cold (rarely-used) cache blocks are moved out to CPU RAM and loaded back when a request needs them again. It measures the reload cost against the [throughput](/shared/glossary/#throughput) gained on long-running sessions.

## Why This Matters

Very long chats and agent sessions can hold more cache than fits on the GPU. Tiering to cheaper, slower memory lets those sessions survive instead of being dropped — but every reload adds latency, so measuring the trade-off tells you when offload helps and when it hurts.
