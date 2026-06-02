# Cosine vs Linear Schedule

## Key Insight

The [noise schedule](/shared/glossary/#noise-schedule) decides how fast a [DDPM](/shared/glossary/#ddpm) destroys an image as it walks from clean data to pure static — and that single choice quietly controls how well the model trains. A *linear* schedule adds noise at a constant rate, which turns out to wipe out almost all image structure too early, so the model wastes many of its later steps learning from inputs that are already nearly pure static. A *cosine* schedule (Nichol & Dhariwal) instead follows the gentle shoulder of a cosine curve, keeping more signal alive through the middle of the process so every step carries useful information. This project trains two otherwise-identical models and compares their [FID (Fréchet Inception Distance)](/shared/glossary/#fid) and samples, making the schedule's impact concrete rather than theoretical.
