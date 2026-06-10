# Forward Kinematics From Scratch

## Key Insight

[Forward kinematics](/shared/glossary/#forward-kinematics) answers the most basic question in robotics: given the angle of every joint, where in space is the hand? You compute it by multiplying one [homogeneous transform](/shared/glossary/#homogeneous-transform) per joint, walking outward from the robot's base frame to its [end-effector](/shared/glossary/#end-effector), so the whole computation is just a chain of 4×4 matrix products. It is always solvable and always cheap — unlike [inverse kinematics](/shared/glossary/#inverse-kinematics), its much harder mirror image — which is why coding it from raw NumPy and checking the result against a trusted library like [Pinocchio](/shared/glossary/#pinocchio) is the rite of passage that makes [Jacobians](/shared/glossary/#jacobian), control, and planning concrete later on.
