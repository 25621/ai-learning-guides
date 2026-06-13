# World-Model Rollout

## Key Insight

Planning actions using a learned [world model](/shared/glossary/#world-model) allows a robot to evaluate action sequences in imagination before executing them in the physical world. By performing a [world-model rollout](/shared/glossary/#world-model-rollout) over a sequence of candidate actions, the system generates predicted future video frames or states. A planning agent can then score these trajectories based on their visual similarity to a goal image and select the action sequence that is predicted to yield the most successful outcome.
