# REINFORCE on CartPole

## Key Insight

[REINFORCE](/shared/glossary/#reinforce) is the most direct way to learn a [policy](/shared/glossary/#policy): instead of estimating action values and acting [greedily](/shared/glossary/#greedy-policy), it nudges the policy's [weights](/shared/glossary/#weights) so that actions taken in successful episodes become more likely and actions from failed ones become less likely. The mechanism is the [policy gradient theorem](/shared/glossary/#policy-gradient-theorem), made computable by the [log-derivative trick](/shared/glossary/#log-derivative-trick), which weights each action by the full [Monte Carlo](/shared/glossary/#monte-carlo-method) [return](/shared/glossary/#return) of its [rollout](/shared/glossary/#rollout). [CartPole](/shared/glossary/#cartpole) is the gentlest place to see this work — but also to feel its flaw: because the weight is a whole noisy trajectory's return rather than a one-step estimate, the gradient is [unbiased but high-variance](/shared/glossary/#bias-variance-tradeoff), so training lurches around and learns slowly. Watching that variance firsthand is the whole point, and it motivates the baseline and [actor-critic](/shared/glossary/#actor-critic) fixes in the projects that follow.

## REINFORCE vs. DQN

[REINFORCE](/shared/glossary/#reinforce) and [DQN (Deep Q-Network)](/shared/glossary/#dqn) represent the two primary branches of reinforcement learning: **policy-gradient** methods and **value-based** methods.

- **DQN (Value-Based):** Learns to estimate the expected future reward (value) of taking each action in a given state (using a [critic](/shared/glossary/#actor-critic) or value network), and then acts [greedily](/shared/glossary/#greedy-policy) based on those estimates. It is [off-policy](/shared/glossary/#off-policy) and relies on an [experience replay](/shared/glossary/#experience-replay) buffer to learn from past data.
- **REINFORCE (Policy-Gradient):** Bypasses estimating action values entirely. It directly outputs a probability distribution over actions (the [policy](/shared/glossary/#policy)) and updates its weights using [Monte Carlo](/shared/glossary/#monte-carlo-method) [returns](/shared/glossary/#return) from complete [rollouts](/shared/glossary/#rollout). It is strictly [on-policy](/shared/glossary/#on-policy).

**Analogy:** Imagine learning to play golf.
- A **DQN** golfer (value-based) tries to calculate the exact expected score or landing position for every possible club and swing angle, and then picks the one with the highest estimated value.
- A **REINFORCE** golfer (policy-gradient) doesn't calculate any values. They just swing. If the ball lands in or near the hole, they try to repeat that exact swing next time. If the ball lands in a pond, they avoid that swing in the future.

While DQN is often more sample-efficient because it can reuse past data, REINFORCE is mathematically simpler, directly optimizes the policy, and works naturally in continuous action spaces where calculating values for infinitely many actions is impossible.

