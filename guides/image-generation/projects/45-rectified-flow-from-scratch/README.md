# Rectified Flow from Scratch

## Key Insight

[Rectified flow](/shared/glossary/#rectified-flow) is a kind of [flow matching](/shared/glossary/#flow-matching): instead of predicting the noise added at a discrete step the way [DDPM](/shared/glossary/#ddpm) does, you train the model to predict a *velocity* — the straight-line direction `ε - x_0` that points from a half-noised image back toward clean data. Because the training paths are straight lines, [sampling](/shared/glossary/#sampling) simply steps along the predicted arrows by solving an [ODE](/shared/glossary/#ode) (with [Euler](/shared/glossary/#euler-method) or [Heun](/shared/glossary/#heuns-method)), and you reach good images in only 10–50 steps. Re-deriving your earlier diffusion model with this objective shows how little has to change — the same network, but a simpler loss with no [noise schedule](/shared/glossary/#noise-schedule) to tune — yet it trains cleanly and few-step sampling works out of the box.
