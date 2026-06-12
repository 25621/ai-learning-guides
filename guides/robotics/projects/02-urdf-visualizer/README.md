# URDF Visualizer

## Key Insight

A robot is a tree of [rigid](/shared/glossary/#rigid-body) links joined by movable joints, and a [URDF](/shared/glossary/#urdf--mjcf--usd) file is the standard text description of that tree — each link's shape and mass, each joint's axis and motion limits, and how they nest from the base out to the hand. Loading that file and drawing the robot at random joint angles turns an abstract spec into a picture, which is the fastest way to catch a flipped axis or a mis-parented link before it wastes hours on real hardware. This is also your first hands-on use of a fast rigid-body library like [Pinocchio](/shared/glossary/#pinocchio) — the [kinematics](/shared/glossary/#kinematics)-and-dynamics engine you will lean on for the rest of the guide.
