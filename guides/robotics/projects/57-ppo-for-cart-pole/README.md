# PPO for Cart-Pole

## Key Insight

[Proximal Policy Optimization (PPO)](/shared/glossary/#ppo) is the standard [on-policy](/shared/glossary/#on-policy) [reinforcement learning](/shared/glossary/#reinforcement-learning) algorithm, balancing stable training with ease of implementation through a constrained [policy gradient](/shared/glossary/#policy-gradient-theorem) update. By implementing a [clipping loss](/shared/glossary/#clipping-loss) that limits the size of the policy update step, PPO prevents the policy from drifting into regions of parameter space that degrade performance. Solving the classic [cart-pole](/shared/glossary/#cartpole) task with PPO demonstrates how tracking the policy ratio and using an [actor-critic](/shared/glossary/#actor-critic) [baseline](/shared/glossary/#baseline) keeps updates stable without requiring complex second-order optimization.
