# Style LoRA

## Key Insight

A style [LoRA](/shared/glossary/#lora) shows that the same low-rank [fine-tuning](/shared/glossary/#fine-tuning) trick used for *subjects* works just as well for a *look* — train on ~30 images sharing a coherent style (a painter's brushwork, a film's color grade) and the adapter learns the style without memorizing any single picture. Success is measured by [generalization](/shared/glossary/#generalization): the style must transfer to prompts that never appeared in the 30 images, proving the LoRA captured *how* things are rendered rather than *what* was in the training set. This is the line between a useful style adapter and an [overfit](/shared/glossary/#overfitting) one — the former restyles anything you ask for, the latter just regurgitates its training images.
