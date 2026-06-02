# VP vs VE Comparison

## Key Insight

[VP and VE](/shared/glossary/#vp--ve-sde) are the two classic ways to define the forward noising process of a diffusion model. Variance-Preserving (VP) — the family [DDPM](/shared/glossary/#ddpm) belongs to — shrinks the original signal as it adds noise so the total variance stays around 1 the whole way; Variance-Exploding (VE) — used by the original [score](/shared/glossary/#score)-based papers — leaves the signal untouched and just piles on ever-larger noise, so the variance grows without bound. The two are mathematically interconvertible and reach similar [FID](/shared/glossary/#fid), but they differ in numerical conditioning and in which samplers behave well, which is exactly what makes the comparison instructive. This project trains the same model under both [SDE](/shared/glossary/#sde-stochastic-differential-equation) families and compares quality, training stability, and sampler behavior side by side.
