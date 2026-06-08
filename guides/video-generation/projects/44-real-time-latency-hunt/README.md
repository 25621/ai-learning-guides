# Real-Time Latency Hunt

## Key Insight

Interactivity has a hard deadline: to feel real-time at 30 [frames per second](/shared/glossary/#frame-rate-fps) the model must produce each frame in about 33 milliseconds, yet a normal [diffusion model](/shared/glossary/#diffusion-model) needs dozens of denoising steps per frame — far too slow. This project closes that gap with [distillation](/shared/glossary/#distillation): train a fast "student" [consistency model](/shared/glossary/#consistency-model) to reproduce a slow 50-step "teacher" in just 4 steps — or even 1 — then measure the real win in milliseconds per frame. Cutting steps from 50 to 4 is a ~12× speedup (50 ÷ 4 ≈ 12), the difference between a model that renders a clip overnight and one you can actually play. This is the same few-step toolkit behind real-time and [streaming](/shared/glossary/#streaming-video-generation) systems like [CausVid](/shared/glossary/#causvid).
