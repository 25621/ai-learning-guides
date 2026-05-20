# Straight-Through Estimator

---

> Pretend the non-differentiable is differentiable.

---

## Key Insight

Some operations, like rounding or thresholding, are non-differentiable because their derivative is zero almost everywhere. A [straight-through estimator](/shared/glossary/#straight-through-estimator) (STE) solves this by using the non-differentiable operation in the forward pass, but passing the gradients straight through unchanged during the [backward pass](/shared/glossary/#backward-pass) as if the operation was an identity function.

## Why This Matters

STEs are essential for training models with discrete components, such as [VQ-VAE](/shared/glossary/#vq-vae)s or discrete latent variables. They offer a practical workaround for incorporating hard decision boundaries into continuous [autograd](/shared/glossary/#autograd) pipelines.