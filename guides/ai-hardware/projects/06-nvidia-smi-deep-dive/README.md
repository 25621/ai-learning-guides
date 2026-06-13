# nvidia-smi Deep Dive

## Key Insight

Using the [nvidia-smi](/shared/glossary/#nvidia-smi) command-line utility provides direct visibility into the active [GPU](/shared/glossary/#gpu) state, including its [compute capability](/shared/glossary/#compute-capability) version, [SM](/shared/glossary/#sm) count, and [NVLink](/shared/glossary/#nvlink) topology. Learning to parse this output allows developers to diagnose communication bottlenecks and monitor memory allocation in multi-GPU configurations.
