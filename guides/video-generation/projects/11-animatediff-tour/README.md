# AnimateDiff Tour

## Key Insight

[AnimateDiff](/shared/glossary/#animatediff) turns any community [Stable Diffusion](/shared/glossary/#stable-diffusion) image [checkpoint](/shared/glossary/#checkpoint) into a short-video generator without retraining it: it slots a separately trained [motion module](/shared/glossary/#motion-module) — a stack of time-aware layers — between the [frozen](/shared/glossary/#frozen) image model's blocks, and that one module supplies all the motion knowledge. This project plugs that module into an off-the-shelf SD 1.5 checkpoint and generates *animated stills* (subtle, looping motion on an otherwise static scene), letting you swap art styles freely while the same motion module animates each one. The lesson is that motion can be learned *once* and reused across many image models — a clean example of [temporal inflation](/shared/glossary/#temporal-inflation) packaged as a drop-in part rather than a model you must train end to end.
