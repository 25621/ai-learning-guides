# DDIM Inversion and Editing

## ELI5 (Explain Like I'm 5)

- **The Big Idea:** A diffusion model turns noise into a picture. To edit a *real* picture, you first run the process **backwards** to find the exact noise that would have produced it. Then you re-run forwards with a changed description — since you start from almost the same noise, the layout stays put and only the described content changes.
- **Analogy:** Imagine a river flowing downhill from a spring to a lake (noise to image). To move a real boat on the lake, you trace the current back up to the spring it came from, then let it flow down again — but nudge the description of the destination. It follows nearly the same riverbed, so it arrives in almost the same place, slightly changed.
- **Example:** We take real handwritten 4s, invert them to noise, then denoise while asking for a 9. The 9s come out with the 4s' slant, thickness and position — same pose, new digit.

## Key Insight

To edit a *real* photo with a [diffusion model](/shared/glossary/#diffusion-model) you first need the noise that would regenerate it — and [DDIM inversion](/shared/glossary/#ddim-inversion) finds it by running the deterministic [DDIM](/shared/glossary/#ddim) sampler *backwards*, adding noise along the same path the model would later remove. Once you hold that starting noise you change the prompt and denoise forward again: because the trajectory is largely shared, the new image keeps the original's layout and pose while swapping the content you re-described. The imperfection you will see — inversion drifts, so the reconstruction is never pixel-perfect — is exactly the gap that follow-up methods like *null-text inversion* were built to close.

## What's in this directory

| File | Role |
|------|------|
| `ddim_invert.py` | The matched pair: `ddim_invert` (walk the deterministic path *up* to noise) and `ddim_denoise` (walk it back *down*) |
| `edit.py` | Reconstruct and edit real MNIST digits; emit the figures |

Our base is the phase-5 [class-conditional DDPM](../28-class-conditional-ddpm/README.md);
the class label stands in for the text prompt.

```bash
# reuse the shared conditional base (also used by projects 51/52/56):
python ../51-dreambooth/train_cond_base.py --out checkpoints/cond_base.pt
python edit.py     # ~1 min
```

## The subtlety that makes or breaks it

Deterministic [DDIM](/shared/glossary/#ddim) (eta = 0) is an ODE, so inversion
is just the same update run the other way. But the plain sampler (the
[DDIM sampler](../27-ddim-sampler/README.md) project) **clamps** the predicted
clean image to `[-1, 1]` each step for stability — and clamping is a non-linear
op that breaks invertibility. Reconstructing a real image needs the *unclamped*
partner, which is exactly why `ddim_denoise` lives here alongside `ddim_invert`
instead of reusing the earlier sampler. With the matched pair, the round-trip
MSE on real digits is ~0.004.

## Results

**Reconstruction — invert, then denoise with the SAME label.** Top row: real
test digits. Bottom row: invert to noise and back. The two rows are nearly
identical; the small residual differences are the "inversion drift" the Key
Insight names — the gap null-text inversion later closes:

![Reconstruction](outputs/recon.png)

**Editing — invert 4s as class 4, denoise as class 9.** Top row: real 4s.
Bottom row: the same noise denoised toward "9". The 9s inherit each 4's slant,
stroke thickness and position — structure is preserved because the trajectory
is shared; only the identity the label describes changes:

![DDIM inversion edit](outputs/edit.png)

A couple of edits wobble (one keeps too much of the 4, one over-rounds) —
honest evidence that inversion is approximate on a small model, and motivation
for the structure-preserving follow-ups (Prompt-to-Prompt, null-text inversion).

## Things to try

- Raise the step count from 100 toward `T` and watch reconstruction tighten —
  inversion accuracy trades directly against speed.
- Edit along a different pair (7→1, 3→8) and see which share enough structure
  for a clean swap.
- Interpolate between the source and target label embeddings to get a smooth
  4→9 morph from a single inverted noise.
