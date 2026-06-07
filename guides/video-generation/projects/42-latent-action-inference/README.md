# Latent Action Inference

## Key Insight

Most [action-conditioned](/shared/glossary/#action-conditioning) models need video where every frame is tagged with the action that caused it — but the internet's billions of hours of footage carry no such labels. A [latent action model](/shared/glossary/#latent-action-model), the trick behind [Genie](/shared/glossary/#genie), sidesteps this by *inferring* the action: it learns a small [latent](/shared/glossary/#latent-space) code that best explains how one frame became the next, discovering a reusable vocabulary of "moves" with no labels at all. This project trains exactly that on unlabeled clips, then checks whether the discovered codes line up with real, meaningful actions (does code #3 always mean "move left"?). Succeed and you can build a controllable [world model](/shared/glossary/#world-model) straight from raw footage — the data bottleneck that limits everything else in this phase suddenly disappears.
