# Triton Softmax

## Key Insight

Writing a [softmax](/shared/glossary/#softmax) [kernel](/shared/glossary/#kernel) in [Triton](/shared/glossary/#triton) demonstrates how block-level programming simplifies custom [GPU](/shared/glossary/#gpu) acceleration compared to raw [CUDA](/shared/glossary/#cuda). Softmax is fundamentally memory-bound, meaning performance depends entirely on optimizing memory access and maximizing [memory bandwidth](/shared/glossary/#memory-bandwidth) utilization. Implementing it also highlights the need for numerical techniques to prevent [underflow](/shared/glossary/#underflow) or overflow during exponentiation.
