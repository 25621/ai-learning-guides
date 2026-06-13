# CUDA Vector Add

## Key Insight

Writing a basic [kernel](/shared/glossary/#kernel) for vector addition serves as the "Hello World" of [GPU](/shared/glossary/#gpu) programming, introducing the core concepts of [CUDA](/shared/glossary/#cuda) thread hierarchies and launch configurations. By mapping threads to individual data elements, we learn how to parallelize memory-bound operations. Profiling this operation using [Nsight Systems](/shared/glossary/#nsight-systems) reveals how the choice of block dimensions impacts [occupancy](/shared/glossary/#occupancy) and execution efficiency.
