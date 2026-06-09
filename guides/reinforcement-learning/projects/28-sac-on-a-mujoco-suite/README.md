# SAC on a Mujoco Suite

## Key Insight

[SAC](/shared/glossary/#sac) (Soft Actor-Critic) is the modern default for [continuous control](/shared/glossary/#continuous-control) because it folds an entropy bonus straight into the objective — it follows [maximum-entropy RL](/shared/glossary/#maximum-entropy-rl), maximizing reward *and* keeping the [policy](/shared/glossary/#policy) as random as it can afford — so it explores on its own and stays robust without the hand-tuned action noise [DDPG](/shared/glossary/#ddpg) and [TD3](/shared/glossary/#td3) lean on. Running it across a [MuJoCo](/shared/glossary/#mujoco) suite of increasingly hard bodies — [HalfCheetah](/shared/glossary/#halfcheetah), [Walker2d](/shared/glossary/#walker2d), [Ant](/shared/glossary/#ant), and [Humanoid](/shared/glossary/#humanoid) — tests whether one algorithm with one hyperparameter setting can scale from a simple runner to a 17-joint humanoid, and SAC's reputation rests on the fact that it usually can.
