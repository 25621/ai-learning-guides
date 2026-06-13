# Mini FlashAttention

## Key Insight

Writing a simplified [FlashAttention](/shared/glossary/#flashattention) [kernel](/shared/glossary/#kernel) in [Triton](/shared/glossary/#triton) demonstrates how online [softmax](/shared/glossary/#softmax) computation enables tiling of the [attention](/shared/glossary/#attention) mechanism. By calculating softmax scaling factors incrementally, we avoid materializing the massive intermediate attention matrix in global memory, keeping all intermediate data within the SM's fast [SRAM](/shared/glossary/#sram). This project illustrates the fundamental design pattern of modern high-performance deep learning: transforming a memory-bound operation into a compute-bound one.
