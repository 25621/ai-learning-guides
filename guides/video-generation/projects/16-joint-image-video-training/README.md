# Joint Image-Video Training

## Key Insight

When you [fine-tune](/shared/glossary/#fine-tuning) an image model on video alone, its still-image quality quietly collapses — the rich appearance knowledge it started with drifts away because video datasets are smaller and more compressed than image datasets. [Joint image-video training](/shared/glossary/#joint-image-video-training) prevents this by mixing the two in every batch — here 90% still images (each treated as a one-frame "video") and 10% real clips — so the model keeps practicing crisp single-frame generation while it learns motion. This project co-trains your inflated model both ways and compares it against a video-only run, so you can see directly how much sharper and more data-efficient the joint recipe is. It works without any change to the architecture because a still image is simply the `T=1` special case of a video: the very same layers process both.
