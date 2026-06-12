# Direct Collocation

## Key Insight

For underactuated systems like the [CartPole](/shared/glossary/#cartpole), finding a trajectory that respects the system's physical dynamics is extremely difficult using geometric planners alone. [Direct collocation](/shared/glossary/#direct-collocation) solves this by discretizing both the robot's [states](/shared/glossary/#state) and control inputs, enforcing the dynamics equations as algebraic constraints at collocation points, and solving the resulting problem using [IPOPT](/shared/glossary/#ipopt). This project plans a cart-pole swing-up maneuver to show how optimization can discover complex dynamic maneuvers, which are then stabilized near the upright [setpoint](/shared/glossary/#setpoint) using a local [LQR](/shared/glossary/#lqr) controller.
