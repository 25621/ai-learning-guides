# Storage Study

## Key Insight

How you store video on disk is a direct trade between space and speed: raw `.npy` frames decode instantly but are enormous, while compressed formats shrink the files 10–100× at the cost of CPU time to decode them back. This project stores the same 100 clips three ways — raw arrays, [H.264](/shared/glossary/#h264) inside an `.mp4` [container](/shared/glossary/#media-container), and [AV1](/shared/glossary/#av1) inside a `.webm` — and measures both disk footprint and decode speed. The right choice depends on whether your training run is bottlenecked on disk space, on network bandwidth, or on CPU decode, and this project gives you the concrete numbers to decide instead of guessing.
