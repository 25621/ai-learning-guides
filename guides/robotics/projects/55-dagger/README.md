# DAgger

## Key Insight

[DAgger (Dataset Aggregation)](/shared/glossary/#dagger) addresses the [covariate-shift](/shared/glossary/#covariate-shift) problem of [behavior cloning](/shared/glossary/#bc) by turning offline [imitation learning](/shared/glossary/#imitation-learning) into an active, online process. By running the trained policy in the environment to generate [rollouts](/shared/glossary/#rollout), querying an expert to provide the correct actions for those newly visited states, and aggregating this data back into the training set, DAgger dynamically bridges the gap between training and testing distributions. This iterative feedback loop teaches the policy not just how to perform the task perfectly, but how to recover and steer back when it inevitably drifts off the expert trajectory.
