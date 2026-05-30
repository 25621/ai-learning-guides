# Calibration Drift Study

---

> The data you calibrated on in January is not the data your users send in April.

---

## Key Insight

This project [calibrates](/shared/glossary/#calibration) a [quantized](/shared/glossary/#quantization) model on traffic from week 0, then re-measures its quality 12 weeks later — after the incoming requests have shifted ([distribution drift](/shared/glossary/#distribution-drift)) — and quantifies how much the gap has grown.

## Why This Matters

Calibration tunes a model to the data it saw, so as real user traffic drifts over time a once-good quantization can quietly get worse. Measuring that decay tells you how often you actually need to re-calibrate.
