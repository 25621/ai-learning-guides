# RRT-Connect for an Arm

## Key Insight

Planning collision-free motions for high-dimensional manipulators is challenging because obstacles in the workspace form highly complex shapes in the joint [C-space](/shared/glossary/#c-space). The [RRT-Connect](/shared/glossary/#rrt-connect) algorithm solves this by growing two separate random trees—one from the start pose and one from the goal pose—and attempting to connect them in the middle at each step. This project plans a 7-[DoF (degrees of freedom)](/shared/glossary/#degrees-of-freedom) reach around a table obstacle in the [MuJoCo](/shared/glossary/#mujoco) simulator to show how bidirectional tree growth speeds up planning in cluttered, multi-joint spaces.
