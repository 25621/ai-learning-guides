# Video Frame VLM

## Key Insight

The cheapest way to make a [VLM](/shared/glossary/#vlm) "watch" a video is to not treat it as video at all: sample a handful of frames (say 8 evenly spaced ones), hand them to the model as 8 still images, and ask a [video QA](/shared/glossary/#vqa-visual-question-answering) question over the lot. This works surprisingly well for questions about *what* is present ("is there a dog?") because a few snapshots usually contain the answer, and it reuses an existing image VLM with zero new training. Its blind spot is *motion and order* — anything that depends on what happened between the sampled frames (did the cup fall before or after it was touched?) is lost the moment you throw the in-between frames away.
