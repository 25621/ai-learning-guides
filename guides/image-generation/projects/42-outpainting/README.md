# Outpainting

## Key Insight

[Outpainting](/shared/glossary/#outpainting) extends an image *beyond* its original borders — turning a portrait into a full scene — and the trick is that it is just [inpainting](/shared/glossary/#inpainting) pointed outward: you paste the original onto a larger blank canvas, mark the new border region as the area to fill, and let the [diffusion model](/shared/glossary/#diffusion-model) generate only there while keeping the original pixels fixed. Because the model conditions on the surviving edge, the new content continues the scene's lighting, texture, and lines naturally instead of starting fresh. This project builds it directly on top of an inpainting loop, which is the whole point: no new training, just a bigger canvas and the right mask.
