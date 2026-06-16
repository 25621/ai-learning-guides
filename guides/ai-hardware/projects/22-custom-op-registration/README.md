# Custom Op Registration

## Key Insight

Registering a custom [Triton](/shared/glossary/#triton) [kernel](/shared/glossary/#kernel) as a [custom op](/shared/glossary/#custom-op) allows PyTorch's compiler, [torch.compile](/shared/glossary/#torchcompile), to trace and optimize it within a larger neural network model graph. By defining schema, validation rules, and derivative mappings, we bridge the gap between low-level hardware kernels and high-level graph execution. This integration ensures that custom optimizations do not break auto-differentiation or graph capture, making custom acceleration safe for production training and inference.
