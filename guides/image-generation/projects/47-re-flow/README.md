# Re-flow

## Key Insight

A freshly trained [rectified flow](/shared/glossary/#rectified-flow) model already follows fairly straight paths from noise to image, but they are not perfectly straight, so taking very few [sampling](/shared/glossary/#sampling) steps still wobbles off course. "Re-flow" fixes this by using the trained model to generate many (noise, image) pairs, then training a *second* model to map each noise sample directly to its paired image along a straight line — literally teaching it the shortcut the first model discovered. After one or two rounds the trajectories become straight enough to sample in as few as one or two steps, which is the foundation of the fast [distillation](/shared/glossary/#distillation) methods behind real-time image generators.
