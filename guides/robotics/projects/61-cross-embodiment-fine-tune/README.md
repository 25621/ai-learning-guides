# Cross-Embodiment Fine-Tune

## Key Insight

[Cross-embodiment](/shared/glossary/#cross-embodiment) learning allows a robotic [policy](/shared/glossary/#policy) to generalize across different robot geometries, actuators, and platforms by pretraining on massive, heterogeneous datasets like [Open X-Embodiment](/shared/glossary/#open-x-embodiment). Fine-tuning these large-scale models on a specific target robot requires far fewer demonstrations than training from scratch, because the model's vision encoder and spatial representations are already optimized. By learning a shared action-representation mapping, the robot leverages general physical concepts learned from other platforms to solve its specific task with high sample efficiency.
