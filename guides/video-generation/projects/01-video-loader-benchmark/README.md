# Video Loader Benchmark

## Key Insight

Most video-training pipelines are bottlenecked on *decoding* the video — turning a compressed file back into frames — not on the GPU doing math, so the library you pick to read frames off disk quietly sets your training speed. This project times four common readers — [decord](/shared/glossary/#decord), `torchvision.io`, PyAV, and `ffmpeg-python` — on the same folder of `.mp4`s and reports decode time per clip. They differ by large factors because each makes different trade-offs around the underlying [video codec](/shared/glossary/#video-codec), CPU threading, and how directly it hands you a [tensor](/shared/glossary/#tensor) instead of a generic image. Finding the winner for *your* data turns a slow pipeline into a fast one without touching the model at all.
