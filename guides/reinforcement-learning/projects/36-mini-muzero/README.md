# Mini MuZero

## Key Insight

[MuZero](/shared/glossary/#muzero) combines [Monte Carlo Tree Search](/shared/glossary/#mcts) with a learned [dynamics model](/shared/glossary/#dynamics-model) that predicts only what matters for decisions — reward, [value](/shared/glossary/#value-function), and [policy](/shared/glossary/#policy) — without ever reconstructing the actual game board, which is why the same algorithm works on [Atari](/shared/glossary/#atari) pixels as well as board games. 

Implementing it on a tiny game like [Tic-Tac-Toe](/shared/glossary/#tic-tac-toe) or 4×4 [Connect Four](/shared/glossary/#connect-four) exposes its three coupled neural network heads, which collaborate to let the agent plan ahead entirely in its mind:

*   **[Representation function](/shared/glossary/#representation-function-muzero) (or Representation head):** This translates raw inputs (like game screens or board coordinates) into a clean, abstract [latent state](/shared/glossary/#latent-space) containing only details relevant to winning. *Analogy: A chess master looking at a board does not care about the wood grain of the pieces or the glare of the lights; they represent the board abstractly in their mind as "castled king" or "weak center."*
*   **[Dynamics head](/shared/glossary/#dynamics-head-muzero):** This takes a latent state and a proposed action, and predicts the *next* latent state and any immediate [reward](/shared/glossary/#reward-function). *Analogy: When planning a move in your head, you think "if I slide my rook here (action), the position in my mind will change (next state) and I will capture their bishop (reward)."*
*   **[Prediction head](/shared/glossary/#prediction-head-muzero):** This looks at any latent state (real or imagined) and immediately predicts the policy (which moves are best to try) and value (the chance of winning). *Analogy: A seasoned player looks at a board layout and has a gut feeling: "I have about an 80% chance of winning from here (value), and my best next move is probably to advance my pawn (policy)."*

The "Zero" in the name marks its descent from [AlphaZero](/shared/glossary/#alphazero), which ran the same search-plus-learning loop but was handed the real game rules; MuZero learns the rules instead.
