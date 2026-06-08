# SARSA vs Q-Learning on Cliff Walking

## Key Insight

[SARSA](/shared/glossary/#sarsa) and [Q-learning](/shared/glossary/#q-learning) differ in exactly one spot — which next action they put into the [TD target](/shared/glossary/#td-error) — and [Cliff Walking](/shared/glossary/#cliff-walking) is the environment designed to make that one difference visible. SARSA is [on-policy](/shared/glossary/#on-policy): its target uses the action the [ε-greedy](/shared/glossary/#epsilon-greedy) policy *actually* takes next, so it "knows" it will sometimes explore and step off the cliff, and it learns a safer path that gives the edge a wide berth. Q-learning is [off-policy](/shared/glossary/#off-policy): its target uses the *greedy* next action, so it learns the optimal cliff-edge path yet earns *lower* reward during training because exploration occasionally shoves it over the edge.

This highlights a key lesson: the theoretically optimal policy (Q-learning) and the policy that actually performs best *while still exploring* (SARSA) are not always the same. Because Q-learning ignores its own exploration, it hugs the risky edge and frequently falls. SARSA, however, learns to expect its own exploratory steps (like anticipating that it might occasionally "trip") and chooses a safer path further inland. Reproducing Sutton & Barto's Figure 6.5 makes this lesson concrete.
