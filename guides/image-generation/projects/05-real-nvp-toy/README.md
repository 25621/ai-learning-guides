# Real NVP Toy

## Key Insight

A [normalizing flow](/shared/glossary/#normalizing-flow) learns to generate data by warping simple random noise through a series of reversible steps, and because every step is reversible it can also report the exact probability of any point — something most generative models cannot do. This project trains a small Real NVP flow on easy 2D shapes like two moons or a swiss roll, then visualizes both the density it learned and the samples it draws. Working in 2D lets you actually *see* the learned distribution as a heatmap, which makes the otherwise abstract idea of "modeling a probability density" concrete. The catch you will notice: keeping every step reversible heavily constrains the architecture, which is why flows lost ground to diffusion on real images.
