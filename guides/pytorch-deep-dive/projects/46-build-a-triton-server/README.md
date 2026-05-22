# Build a Triton Server

---

> Wrapping a model in a server turns it into something other programs can call over the network.

---

## Key Insight

[Triton Inference Server](/shared/glossary/#triton-inference-server) is NVIDIA's production server for hosting models. It loads your model, exposes it over HTTP, and handles [batching](/shared/glossary/#batching) and multiple model versions, so clients can send inputs and get predictions back over the network.

## Why This Matters

In production, a model rarely runs in the same process as the application using it. A serving framework like Triton turns your model into a network service with [batching](/shared/glossary/#batching), versioning, and monitoring built in.
