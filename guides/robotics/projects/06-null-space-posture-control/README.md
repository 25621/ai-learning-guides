# Null-Space Posture Control

## Key Insight

An arm with more than six joints is [redundant](/shared/glossary/#kinematic-redundancy) — many different joint configurations place the hand in exactly the same pose — and the set of joint motions that move the joints *without* moving the hand is the [null space](/shared/glossary/#null-space) of the [Jacobian](/shared/glossary/#jacobian). This project uses that spare freedom to do two jobs at once: track an [end-effector](/shared/glossary/#end-effector) trajectory with the primary task while quietly nudging the elbow toward a comfortable "home" posture with the leftover motion. The same trick is the foundation of collision avoidance, joint-limit avoidance, and natural-looking motion on 7-[DoF](/shared/glossary/#degrees-of-freedom) and humanoid arms.
