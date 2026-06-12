# Latency Budget Instrument

## Key Insight

In closed-loop robot control, the time elapsed between sensor capture and motor actuation—known as the [latency](/shared/glossary/#latency) budget—directly determines the maximum stable control gain and overall system reactivity. By instrumenting and tracing timestamps across the perception, planning, and driver layers, developers can measure and profile the 5th, 50th, and 95th percentile [latencies](/shared/glossary/#latency). Minimizing this delay and its variability ([jitter](/shared/glossary/#jitter)) is crucial to prevent instability and ensure safe, high-frequency physical interactions.
