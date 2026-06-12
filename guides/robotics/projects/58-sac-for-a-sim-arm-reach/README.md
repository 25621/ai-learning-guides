# SAC for a Sim Arm Reach

## Key Insight

[Soft Actor-Critic (SAC)](/shared/glossary/#sac) is a sample-efficient, [off-policy](/shared/glossary/#off-policy) [actor-critic](/shared/glossary/#actor-critic) [reinforcement learning](/shared/glossary/#reinforcement-learning) algorithm that incorporates [entropy regularization](/shared/glossary/#entropy-regularization) to encourage exploration and robustness. By optimizing the policy to maximize both expected long-term reward and policy entropy, SAC prevents premature convergence to suboptimal deterministic behaviors. A critical implementation detail is tuning the temperature parameter `α` (the [temperature](/shared/glossary/#temperature) parameter) that balances the trade-off between exploitation and exploration, ensuring the robot arm discovers stable trajectories to reach the target location in simulation.
