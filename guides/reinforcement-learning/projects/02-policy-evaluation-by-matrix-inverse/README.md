# Policy Evaluation by Matrix Inverse

## Key Insight

When the [policy](/shared/glossary/#policy) is held fixed, the [Bellman equation](/shared/glossary/#bellman-equation) stops being scary: it becomes an ordinary set of linear equations, one per state, that you can solve in a single shot with the [matrix inverse](/shared/glossary/#matrix-inverse) `V = (I − γPπ)⁻¹ rπ`. This [policy evaluation](/shared/glossary/#policy-evaluation) step — computing the [value function](/shared/glossary/#value-function) of a *given* policy — is the easy half of RL; the hard half is improving the policy afterwards. Solving it once by matrix inverse and again by repeated [Bellman backups](/shared/glossary/#bellman-operator) shows that the slow iterative method everyone uses in practice is simply converging to this exact closed-form answer.
