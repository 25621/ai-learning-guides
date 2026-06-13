# Coalesced vs Non-coalesced

## Key Insight

Ensuring that threads within a [warp](/shared/glossary/#warp) perform contiguous [memory coalescing](/shared/glossary/#memory-coalescing) is critical to achieving high [memory bandwidth](/shared/glossary/#memory-bandwidth) on the [GPU](/shared/glossary/#gpu). When adjacent threads access contiguous global memory addresses, the hardware merges these requests into a single, efficient memory transaction. If threads instead perform strided or scattered accesses, the GPU must execute multiple separate transactions, severely stalling the compute units and throttling overall performance.
