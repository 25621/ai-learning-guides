# Inverse Dynamics from Scratch

## Key Insight

[Inverse dynamics](/shared/glossary/#inverse-dynamics) answers the question a controller actually needs: *what joint torques will produce this exact motion?* The answer is the [manipulator equation](/shared/glossary/#manipulator-equation) `M(q)q̈ + C(q,q̇)q̇ + g(q) = τ`, and the [Recursive Newton-Euler Algorithm (RNEA)](/shared/glossary/#rnea) evaluates it in time that grows only linearly with the number of joints by sweeping outward to accumulate each link's velocity and acceleration, then inward to add up the forces. Coding it by hand for a 2-link arm and checking it against [Pinocchio](/shared/glossary/#pinocchio) is the cleanest way to internalize where the gravity, inertia, and velocity-dependent Coriolis terms each come from — a single sign error shows up immediately as a torque that disagrees with the reference.
