# In-Hand Cube Reorientation

## Key Insight

Controlling multi-finger hands to perform [in-hand manipulation](/shared/glossary/#in-hand-manipulation) requires managing continuous contact changes and joint coordination. By training a control policy using [reinforcement learning](/shared/glossary/#reinforcement-learning) in a physics simulator like [MuJoCo](/shared/glossary/#mujoco), the robot can learn to rotate a cube to arbitrary target orientations. Applying [sim-to-real](/shared/glossary/#sim-to-real) transfer techniques ensures the learned policy remains robust against the unmodeled friction, backlash, and sensor noise of physical robotic hands.
