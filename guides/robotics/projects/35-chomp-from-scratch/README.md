# CHOMP from Scratch

## Key Insight

Instead of separating path search from path smoothing, [trajectory optimization](/shared/glossary/#trajectory-optimization) formulates motion planning as a continuous mathematical minimization problem. [CHOMP](/shared/glossary/#chomp) uses functional gradient descent to optimize an initial trajectory, pulling it away from obstacles using the gradient of a [Signed Distance Field (SDF)](/shared/glossary/#sdf) while simultaneously minimizing joint velocity and acceleration. This project implements CHOMP on a 2D grid to show how gradient descent can smoothly guide a path out of collision, while exploring how the optimizer can get trapped in local minima.
