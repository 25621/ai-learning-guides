# Synthetic Load Tests

---

> "Send one request and it looks fast; send a realistic crowd and the truth shows up."

---

## Key Insight

This project generates fake-but-realistic traffic — prompts and outputs with lifelike length distributions — and replays it at 1×, 2×, and 5× concurrency to see how [latency](/shared/glossary/#latency) and [throughput](/shared/glossary/#throughput) hold up as load grows.

## Why This Matters

Single-request timing lies about production, where many users hit the server at once. Measuring under realistic concurrent load is the only honest way to learn how much traffic one replica can absorb before it breaks its targets.
