# Modality Balance

## Key Insight

This project treats the *per-modality [loss](/shared/glossary/#loss-function) curve* as a measuring instrument: by deliberately under- or over-sampling one [modality](/shared/glossary/#modality) in the data mix and watching each modality's loss fall (or flat-line) on its own curve, you can see directly how the sampling ratio controls what the model actually learns. The diagnostic exposes a trap that an averaged loss hides — a single "multimodal loss" can look healthy while one modality the model is silently ignoring sits stuck near its starting value. Where the [Phase 7 version](/shared/glossary/#modality-balancing) frames this as an architectural fix, here it is a *data-pipeline* knob: [modality balancing](/shared/glossary/#modality-balancing) is just choosing each modality's sampling rate (or loss weight) so every per-modality curve descends at a comparable pace.
