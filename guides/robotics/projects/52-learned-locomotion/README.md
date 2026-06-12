# Learned Locomotion

## Key Insight

Train a [legged locomotion](/shared/glossary/#legged-locomotion) control [policy](/shared/glossary/#policy) using [reinforcement learning](/shared/glossary/#reinforcement-learning) in the GPU-accelerated [Isaac Lab](/shared/glossary/#isaac-lab) simulation environment. Instead of hand-crafting [gait](/shared/glossary/#gait) sequences or footstep planners, the policy learns to coordinate the robot's joints directly from joint angles and inertial measurements to match a target velocity. By applying [domain randomization](/shared/glossary/#domain-randomization) to physics parameters like friction, mass, and latencies, the learned policy develops robustness for successful [sim-to-real](/shared/glossary/#sim-to-real) transfer.
