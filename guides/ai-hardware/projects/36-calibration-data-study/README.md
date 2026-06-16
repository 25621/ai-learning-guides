# Calibration Data Study

## Key Insight

Post-training [quantization](/shared/glossary/#quantization) depends on representative [calibration](/shared/glossary/#calibration) data to determine the optimal scaling factors for [weights](/shared/glossary/#weights) and [activations](/shared/glossary/#activations). Sweeping the number of calibration samples demonstrates how the quality of a quantized model converges as more representative input variety is observed. Understanding this relationship helps developers optimize the calibration phase, ensuring maximum accuracy recovery with minimal computational overhead.
