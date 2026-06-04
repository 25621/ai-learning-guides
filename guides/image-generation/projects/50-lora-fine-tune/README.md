# LoRA Fine-Tune

## Key Insight

[LoRA (Low-Rank Adaptation)](/shared/glossary/#lora) is the reason you can teach a giant [Stable Diffusion](/shared/glossary/#stable-diffusion) checkpoint a brand-new subject on a single consumer GPU. Instead of [fine-tuning](/shared/glossary/#fine-tuning) all of the model's frozen [weights](/shared/glossary/#weights), you leave them untouched and train only a tiny pair of [low-rank](/shared/glossary/#low-rank) matrices that nudge each layer's output — so the result is a few megabytes you can share and swap, not a full multi-gigabyte model. The practical lesson of training on ~20 images of a custom subject is the tension between learning the subject and [overfitting](/shared/glossary/#overfitting): too many steps and the model can only ever redraw your training photos, too few and it never locks onto the subject at all.
