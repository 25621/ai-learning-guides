# Quadruped Trotting MPC

## Key Insight

Implement a convex [Model Predictive Control (MPC)](/shared/glossary/#mpc) controller to generate stable trotting [gaits](/shared/glossary/#gait) on a quadruped robot. The controller simplifies the robot's complex legged [kinematics](/shared/glossary/#kinematics) into a [centroidal dynamics](/shared/glossary/#centroidal-dynamics) model, tracking the momentum of its center of mass. By solving a [quadratic optimization problem](/shared/glossary/#quadratic-program) at each step, the [MPC](/shared/glossary/#mpc) computes the ground reaction forces for the feet in contact, which are then converted to joint torques via a [Whole-Body Control (WBC)](/shared/glossary/#wbc) loop to maintain dynamic balance.
