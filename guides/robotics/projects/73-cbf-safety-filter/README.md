# CBF Safety Filter

## Key Insight

Wrapping a learned [policy](/shared/glossary/#policy) with a [Control Barrier Function (CBF)](/shared/glossary/#cbf) safety filter guarantees physical safety without requiring the policy itself to learn complex constraint boundaries. The safety filter monitors the robot's state and projects the policy's raw control inputs onto a mathematically safe set of actions. If the learned policy proposes a command that would lead to a collision or joint limit violation, the filter minimally modifies the command to ensure safety while preserving the original policy's intent as much as possible.
