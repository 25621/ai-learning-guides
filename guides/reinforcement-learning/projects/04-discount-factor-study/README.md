# Discount Factor Study

## Key Insight

The [discount factor](/shared/glossary/#discount-factor) γ controls how far into the future the agent bothers to look: a small γ makes it grab immediate reward, while a γ near 1 makes it patient, and the rough rule is that it plans over an [effective horizon](/shared/glossary/#effective-horizon) of about `1/(1−γ)` steps (γ = 0.99 gives `1/0.01` ≈ 100 steps). Sweeping γ on one fixed task and watching the [optimal policy](/shared/glossary/#optimal-policy) flip — from grabbing a nearby small reward to walking to a distant large one — shows that γ is not just a numerical knob for convergence but a genuine part of the problem definition. Change γ and you have literally changed what "optimal" means.
