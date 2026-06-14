# Tenstorrent Dev

## Key Insight

Programming accelerators designed by [Tenstorrent](/shared/glossary/#tenstorrent) introduces a pipeline-based computation model that maps deep learning graphs directly onto a grid of [RISC-V](/shared/glossary/#risc-v) cores. By utilizing their open-source software development kit, developers can write low-level [kernels](/shared/glossary/#kernel) that stream data through the [network-on-chip](/shared/glossary/#network-on-chip) rather than relying on heavy warp scheduling and complex memory caching. This architectural alternative offers high efficiency for transformer workloads and provides a hands-on look at non-von Neumann computer designs.
