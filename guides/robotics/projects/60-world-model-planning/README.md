# World-Model Planning

## Key Insight

[World models](/shared/glossary/#world-model) enable model-based [reinforcement learning](/shared/glossary/#reinforcement-learning) by training a neural network to predict the next observations and rewards of the environment given the current state and action. Using this learned model, an agent can perform [planning](/shared/glossary/#planning) entirely in a [latent space](/shared/glossary/#latent-space) without taking real-world or simulator actions, which is highly sample-efficient. Optimizing action trajectories with the [Cross-Entropy Method (CEM)](/shared/glossary/#cem) over this world model allows the robot to simulate hundreds of possible action sequences, select the top-performing paths, and execute only the first action of the best-planned sequence.
