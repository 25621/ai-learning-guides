# Diffusion Policy

## Key Insight

[Diffusion policy](/shared/glossary/#diffusion-policy) represents the action distribution of a robot as a [denoising diffusion model](/shared/glossary/#diffusion-model) conditioned on current observations, allowing it to excel at complex, multi-step manipulation tasks. Unlike standard [behavior cloning](/shared/glossary/#bc) which uses deterministic networks that fail when demonstrations show multiple valid paths, diffusion policies handle [multimodal distributions](/shared/glossary/#multimodal-distribution) naturally by gradually refining random noise into smooth action trajectories. This iterative generation process ensures that the robot makes a clear, decisive choice (such as passing an obstacle on the left or the right) instead of outputting a hazardous average of all demonstrations.
