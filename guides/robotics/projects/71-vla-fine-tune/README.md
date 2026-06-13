# VLA Fine-Tune

## Key Insight

Fine-tuning a pretrained [Vision-Language-Action (VLA) model](/shared/glossary/#vla) adapts its broad generalist capabilities to a specific robotic platform with a fraction of the demonstrations required for from-scratch [Behavior Cloning (BC)](/shared/glossary/#bc). Because the base model already possesses rich visual features and language understanding, the [fine-tuning](/shared/glossary/#fine-tuning) process only needs to align its tokenized action vocabulary with the joint spaces and kinematic limits of the target robot. This makes VLA adaptation highly sample-efficient and robust to visual clutter compared to training a task-specific policy from scratch.
