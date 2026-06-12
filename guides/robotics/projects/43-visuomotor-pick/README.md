# Visuomotor Pick

## Key Insight

Rather than dividing a manipulation task into separate perception, planning, and control steps, a [visuomotor policy](/shared/glossary/#visuomotor-policy) maps camera images directly to joint motor commands. Training such policies end-to-end using [reinforcement learning](/shared/glossary/#reinforcement-learning) inside simulators like [MuJoCo](/shared/glossary/#mujoco) allows the robot to adapt to visual variations and contact dynamics. This direct connection between sight and action enables highly reactive pick-and-place behaviors that can handle moving targets and unexpected physical disturbances during execution.
