# Camera Trajectory

## Key Insight

To move from "something moves in the clip" to "the *camera* pans left and zooms in", a video model has to be told the camera's path explicitly. This project adds [camera control](/shared/glossary/#camera-control) to an [image-to-video](/shared/glossary/#i2v) model by feeding it [Plücker-coordinate](/shared/glossary/#plücker-coordinates) camera embeddings — a compact six-number description of the ray each pixel looks along, computed per frame from the desired camera trajectory — and then verifies the model honors requests to pan, zoom, and orbit. Encoding the camera as per-pixel rays rather than raw position numbers is what lets the model generalize to trajectories it never saw in training, because every pixel then carries a direct geometric hint about where its content should appear to come from.
