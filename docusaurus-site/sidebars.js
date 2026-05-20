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
        {type: 'doc', id: 'guides/pytorch-deep-dive/README', label: 'PyTorch Deep Dive'},
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
