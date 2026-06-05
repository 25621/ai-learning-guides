# Native Video Model

## Key Insight

Instead of bolting frames onto an image model, a native video model cuts the clip into [spatiotemporal patches](/shared/glossary/#patchification) — small 3D boxes (TubeViT-style) that each span a little image region *and* a few frames — so motion is baked into the tokens from the start rather than inferred later. Feeding those tokens to a [transformer](/shared/glossary/#transformer) lets it relate "this corner, early" to "that corner, later" in a single step, which is exactly what frame-by-frame approaches throw away. The cost is that 3D patches multiply the token count fast, so even a small video classifier forces you to confront the compute-versus-detail trade-off that dominates all video modeling.
