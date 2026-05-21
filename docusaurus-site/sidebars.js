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
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/01-stride-explorer/README', label: 'Stride explorer'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/02-view-vs-copy-detective/README', label: 'View vs copy detective'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/03-manual-indexing/README', label: 'Manual indexing'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/04-dtype-precision-study/README', label: 'dtype precision study'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/05-broadcasting-bug-hunt/README', label: 'Broadcasting bug hunt'},
              ],
            },
            {
              type: 'category',
              label: 'Phase 2: Autograd, Inside Out',
              items: [
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/06-micrograd-in-pytorch-style/README', label: 'Micrograd in PyTorch style'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/07-manual-backprop/README', label: 'Manual backprop'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/08-custom-autograd-function/README', label: 'Custom autograd.Function'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/09-straight-through-estimator/README', label: 'Straight-through estimator'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/10-gradient-checkpointing/README', label: 'Gradient checkpointing'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/11-double-backward/README', label: 'Double backward'},
              ],
            },
            {
              type: 'category',
              label: 'Phase 3: nn.Module, Optimizers, and the Training Loop',
              items: [
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/12-module-introspection/README', label: 'Module introspection'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/13-hook-based-feature-extractor/README', label: 'Hook-based feature extractor'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/14-custom-optimizer/README', label: 'Custom optimizer'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/15-implement-adamw-from-scratch/README', label: 'Implement AdamW from scratch'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/16-state-dict-surgery/README', label: 'State dict surgery'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/17-reproducible-training/README', label: 'Reproducible training'},
              ],
            },
            {
              type: 'category',
              label: 'Phase 4: Data Loading and Input Pipelines',
              items: [
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/18-naive-vs-optimized-loader/README', label: 'Naive vs optimized loader'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/19-custom-collate/README', label: 'Custom collate'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/20-weighted-sampler/README', label: 'Weighted sampler'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/21-streaming-webdataset/README', label: 'Streaming WebDataset'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/22-memory-mapped-tokens/README', label: 'Memory-mapped tokens'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/23-profile-and-fix/README', label: 'Profile and fix'},
              ],
            },
            {
              type: 'category',
              label: 'Phase 5: Performance',
              items: [
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/24-profile-a-training-step/README', label: 'Profile a training step'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/25-amp-speedup-study/README', label: 'AMP speedup study'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/26-torch-compile-test/README', label: 'torch.compile test'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/27-memory-breakdown/README', label: 'Memory breakdown'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/28-gradient-accumulation/README', label: 'Gradient accumulation'},
                {type: 'doc', id: 'guides/pytorch-deep-dive/projects/29-bottleneck-fix/README', label: 'Bottleneck fix'},
              ],
            },
          ],
        },
        {type: 'doc', id: 'guides/llm/README', label: 'LLM'},
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
