# Analytic 2D Grasp

## Key Insight

Classical [grasp synthesis](/shared/glossary/#grasp-synthesis) uses geometric models of objects to compute contact points that achieve [force closure](/shared/glossary/#force-closure), ensuring the grasp can resist any external disturbance. In a 2D polygonal setup, this is determined by constructing [friction cones](/shared/glossary/#friction-cone) at each contact and checking if their overlap spans the object's center of mass. Visualizing these friction cones and validating force closure is the foundational step for planning stable grasps before moving to complex, high-dimensional objects.
