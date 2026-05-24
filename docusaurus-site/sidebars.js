/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  guideSidebar: [
    {
      type: 'doc',
      id: 'README',
      label: 'Introduction',
    },
    {
      type: 'category',
      label: 'Guides',
      collapsed: false,
      items: [
        {
          type: 'category',
          label: 'PyTorch Deep Dive',
          link: {type: 'doc', id: 'guides/pytorch-deep-dive/README'},
          items: [
            {
              type: 'category',
              label: 'Phase 1: Tensors and the Storage Model',
              items: [
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/stride-explorer/README', label: 'Stride explorer'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/view-vs-copy-detective/README', label: 'View vs copy detective'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/manual-indexing/README', label: 'Manual indexing'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/dtype-precision-study/README', label: 'dtype precision study'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/broadcasting-bug-hunt/README', label: 'Broadcasting bug hunt'},
              ],
            },
            {
              type: 'category',
              label: 'Phase 2: Autograd, Inside Out',
              items: [
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/micrograd-in-pytorch-style/README', label: 'Micrograd in PyTorch style'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/manual-backprop/README', label: 'Manual backprop'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/custom-autograd-function/README', label: 'Custom autograd.Function'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/straight-through-estimator/README', label: 'Straight-through estimator'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/gradient-checkpointing/README', label: 'Gradient checkpointing'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/double-backward/README', label: 'Double backward'},
              ],
            },
            {
              type: 'category',
              label: 'Phase 3: nn.Module, Optimizers, and the Training Loop',
              items: [
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/module-introspection/README', label: 'Module introspection'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/hook-based-feature-extractor/README', label: 'Hook-based feature extractor'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/custom-optimizer/README', label: 'Custom optimizer'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/implement-adamw-from-scratch/README', label: 'Implement AdamW from scratch'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/state-dict-surgery/README', label: 'State dict surgery'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/reproducible-training/README', label: 'Reproducible training'},
              ],
            },
            {
              type: 'category',
              label: 'Phase 4: Data Loading and Input Pipelines',
              items: [
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/naive-vs-optimized-loader/README', label: 'Naive vs optimized loader'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/custom-collate/README', label: 'Custom collate'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/weighted-sampler/README', label: 'Weighted sampler'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/streaming-webdataset/README', label: 'Streaming WebDataset'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/memory-mapped-tokens/README', label: 'Memory-mapped tokens'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/profile-and-fix/README', label: 'Profile and fix'},
              ],
            },
            {
              type: 'category',
              label: 'Phase 5: Performance',
              items: [
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/profile-a-training-step/README', label: 'Profile a training step'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/amp-speedup-study/README', label: 'AMP speedup study'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/torch-compile-test/README', label: 'torch.compile test'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/memory-breakdown/README', label: 'Memory breakdown'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/gradient-accumulation/README', label: 'Gradient accumulation'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/bottleneck-fix/README', label: 'Bottleneck fix'},
              ],
            },
            {
              type: 'category',
              label: 'Phase 6: Custom Kernels',
              items: [
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/cpp-extension-for-elementwise-add/README', label: 'C++ extension for elementwise add'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/triton-softmax/README', label: 'Triton softmax'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/triton-matmul/README', label: 'Triton matmul'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/fused-mlp/README', label: 'Fused MLP'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/mini-flashattention/README', label: 'Mini FlashAttention'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/custom-op-registration/README', label: 'Custom op registration'},
              ],
            },
            {
              type: 'category',
              label: 'Phase 7: Distributed Training',
              items: [
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/two-gpu-ddp/README', label: 'Two-GPU DDP'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/implement-gradient-allreduce/README', label: 'Implement gradient AllReduce'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/fsdp-a-transformer/README', label: 'FSDP a transformer'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/tensor-parallel-attention/README', label: 'Tensor parallel attention'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/debug-a-hang/README', label: 'Debug a hang'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/multi-node-setup/README', label: 'Multi-node setup'},
              ],
            },
            {
              type: 'category',
              label: 'Phase 8: Deployment',
              items: [
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/export-to-onnx/README', label: 'Export to ONNX'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/mobile-deployment/README', label: 'Mobile deployment'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/dynamic-quantization/README', label: 'Dynamic quantization'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/static-quantization-ptq/README', label: 'Static quantization (PTQ)'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/build-a-triton-server/README', label: 'Build a Triton server'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/latency-profiling/README', label: 'Latency profiling'},
              ],
            },
            {
              type: 'category',
              label: 'Phase 9: Debugging Hard Problems',
              items: [
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/nan-forensics/README', label: 'NaN forensics'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/memory-leak-hunt/README', label: 'Memory leak hunt'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/determinism-audit/README', label: 'Determinism audit'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/hang-diagnosis/README', label: 'Hang diagnosis'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/eager-vs-compile-diff/README', label: 'Eager vs compile diff'},
              ],
            },
            {
              type: 'category',
              label: 'Phase 10: Reading the PyTorch Source',
              items: [
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/trace-one-op-end-to-end/README', label: 'Trace one op end to end'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/read-native-functions-yaml/README', label: 'Read native_functions.yaml'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/build-pytorch-from-source/README', label: 'Build PyTorch from source'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/patch-and-rebuild/README', label: 'Patch and rebuild'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/fix-a-good-first-issue/README', label: 'Fix a "good first issue"'},
              ],
            },
          ],
        },
        {
          type: 'category',
          label: 'LLM',
          link: {type: 'doc', id: 'guides/llm/README'},
          items: [
            {
              type: 'category',
              label: 'Phase 1: Tokenization and Embeddings',
              items: [
                {type: 'doc', id: 'guides/llm/projects/train-a-bpe-from-scratch/README', label: 'Train a BPE from scratch'},
                {type: 'doc', id: 'guides/llm/projects/tokenizer-compression-study/README', label: 'Tokenizer compression study'},
                {type: 'doc', id: 'guides/llm/projects/numeral-tokenization-audit/README', label: 'Numeral tokenization audit'},
                {type: 'doc', id: 'guides/llm/projects/chat-template-debugger/README', label: 'Chat-template debugger'},
                {type: 'doc', id: 'guides/llm/projects/custom-vocab-extension/README', label: 'Custom vocab extension'},
              ],
            },
          ],
        },
        {type: 'doc', id: 'guides/image-generation/README', label: 'Image Generation'},
        {type: 'doc', id: 'guides/reinforcement-learning/README', label: 'Reinforcement Learning'},
        {type: 'doc', id: 'guides/video-generation/README', label: 'Video Generation'},
        {type: 'doc', id: 'guides/robotics/README', label: 'Robotics'},
        {type: 'doc', id: 'guides/multimodal-learning/README', label: 'Multimodal Learning'},
        {type: 'doc', id: 'guides/inference-systems/README', label: 'Inference Systems'},
        {type: 'doc', id: 'guides/ai-hardware/README', label: 'AI Hardware'},
      ],
    },
    {
      type: 'category',
      label: 'Reference',
      collapsed: true,
      items: [
        {type: 'doc', id: 'shared/glossary', label: 'Glossary'},
      ],
    },
  ],
};

module.exports = sidebars;
