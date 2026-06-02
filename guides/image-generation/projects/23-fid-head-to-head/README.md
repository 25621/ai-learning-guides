# FID Head-to-Head

## Key Insight

[FID (Fréchet Inception Distance)](/shared/glossary/#fid) scores how close a model's images are to real ones by comparing the two sets in the feature space of a pretrained classifier — lower means more realistic. This project trains a [DCGAN](/shared/glossary/#dcgan) and a small [VAE](/shared/glossary/#vae) on the same dataset and puts them head-to-head on FID, training time, and stability. The usual lesson is a clean illustration of the era's central trade-off: the GAN reaches a lower (better) FID with sharper samples but is fiddly to train and can fall into [mode collapse](/shared/glossary/#mode-collapse), while the VAE trains smoothly and reliably yet produces blurrier images.
