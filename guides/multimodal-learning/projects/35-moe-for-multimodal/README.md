# MoE for Multimodal

## Key Insight

A [Mixture-of-Experts (MoE)](/shared/glossary/#moe) layer replaces one shared [MLP](/shared/glossary/#mlp) with many parallel [expert](/shared/glossary/#expert) MLPs plus a small router that sends each token to only the top few, so the model can hold a huge number of [parameters](/shared/glossary/#weights) while spending only a little compute per token. In a multimodal model this raises a tempting question: left to itself, will the router learn to send image tokens to one set of experts and text tokens to another — experts that *specialize by modality*? Watching which experts fire for which [modality](/shared/glossary/#modality) is a concrete window into how a unified model divides up its capacity, and a partial answer to whether modalities prefer to be processed separately even inside one shared backbone.
