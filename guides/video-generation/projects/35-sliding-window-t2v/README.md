# Sliding-Window T2V

## Key Insight

[Sliding-window generation](/shared/glossary/#sliding-window-generation) is the cheapest way to push a [text-to-video (T2V)](/shared/glossary/#t2v) model past the clip length it was trained on: generate a series of short clips that overlap by a few frames, then blend each overlap so the joins are invisible. This project chains 5-second clips into a 30-second video and fuses the shared frames in the model's [latent space](/shared/glossary/#latent-space) — averaging there is smoother than on raw pixels and avoids ghosting. Because the only thread tying distant moments together is whatever those short overlaps can carry forward, the picture slowly [drifts](/shared/glossary/#sliding-window-generation): colors creep and a character's outfit mutates the further you travel from the opening frame. It is the first long-form trick worth trying, since it needs no retraining — only a clever way to stitch existing outputs.
