# Bandwidth Measurement

## Key Insight

Measuring the achieved [memory bandwidth](/shared/glossary/#memory-bandwidth) of a simple copy [kernel](/shared/glossary/#kernel) exposes the gap between theoretical hardware specifications and real-world performance. In practice, factors like cache hit rates, memory access patterns, and overhead from the control loop limit how fast data can flow to the [GPU](/shared/glossary/#gpu) cores. Benchmarking this baseline is crucial for diagnosing memory-bound bottlenecks in more complex neural network operations.
