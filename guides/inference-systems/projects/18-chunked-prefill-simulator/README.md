# Chunked Prefill Simulator

---

> One giant prompt should not be allowed to freeze everyone else's stream.

---

## Key Insight

This project builds a small simulator — requests arrive at random times, each [prefill](/shared/glossary/#prefill) takes some time and each [decode](/shared/glossary/#decode) step a little more — to study [chunked prefill](/shared/glossary/#chunked-prefill): slicing a long prefill into pieces so it interleaves with decode steps. Sweeping the chunk size, it plots [tail latency](/shared/glossary/#tail-latency) ([TTFT](/shared/glossary/#ttft) at the 99th percentile) against [throughput](/shared/glossary/#throughput).

## Why This Matters

Without chunking, one 32k-token prompt stalls every other request for hundreds of milliseconds. The simulator lets you find the chunk size that keeps latency smooth without giving up much throughput — and you can explore the whole trade-off curve without ever touching a GPU.
