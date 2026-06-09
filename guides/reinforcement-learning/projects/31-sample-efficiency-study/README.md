# Sample Efficiency Study

## Key Insight

[Sample efficiency](/shared/glossary/#sample-efficiency) asks how much reward an algorithm buys per unit of environment experience, and it cleanly splits the two camps of deep RL: [off-policy](/shared/glossary/#off-policy) methods like [SAC](/shared/glossary/#sac) replay every transition from a [buffer](/shared/glossary/#experience-replay) many times, so they reach good policies in far fewer samples, while [on-policy](/shared/glossary/#on-policy) methods like [PPO](/shared/glossary/#ppo) must throw data away after each update and make up for it with massive parallelism and fast simulation. Running PPO and SAC on the same [MuJoCo](/shared/glossary/#mujoco) task and plotting [return](/shared/glossary/#return) against both samples used and wall-clock time makes the trade-off concrete: SAC usually wins on samples, while PPO often wins on wall-clock time when the simulator is cheap and you can run thousands of environments at once.
