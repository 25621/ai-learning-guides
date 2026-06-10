# Damped Least-Squares IK

## Key Insight

[Inverse kinematics](/shared/glossary/#inverse-kinematics) runs [forward kinematics](/shared/glossary/#forward-kinematics) backward — given a desired hand pose, find the joint angles that achieve it — and the workhorse method repeatedly inverts the [Jacobian](/shared/glossary/#jacobian) to step the joints toward the target. The catch is that near a [singularity](/shared/glossary/#kinematic-singularity) the Jacobian loses rank, and a plain inverse then demands impossibly fast joint motion; [damped least-squares](/shared/glossary/#damped-least-squares) cures this by adding a small penalty (the "damping") that trades a little tracking accuracy for bounded, stable joint velocities. Watching the arm ease through a wrist singularity instead of flailing is the clearest lesson in why this one regularization trick ships inside real industrial controllers.
