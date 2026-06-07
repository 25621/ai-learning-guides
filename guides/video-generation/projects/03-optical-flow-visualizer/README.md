# Optical Flow Visualizer

## Key Insight

[Optical flow](/shared/glossary/#optical-flow) is a per-pixel map of how each point moved between two frames — the raw "motion signal" that video generation ultimately has to model, and that reappears everywhere in the field. This project computes dense flow between adjacent frames with both a classic algorithm ([Farnebäck](/shared/glossary/#farnebäck-optical-flow)) and a modern neural one ([RAFT](/shared/glossary/#raft)), then paints the result with a color wheel where [hue](/shared/glossary/#hue) encodes direction and brightness encodes speed. Seeing motion as color makes it obvious where a clip is calm versus chaotic, and where the classic and neural methods disagree. That same flow signal is exactly what later phases reuse to filter data and to condition models on motion.
