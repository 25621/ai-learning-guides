# Format Sweep

## Key Insight

Training deep learning models in different numerical precisions exposes the fundamental trade-off between speed and accuracy. By comparing [float32](/shared/glossary/#float32), [bfloat16](/shared/glossary/#bfloat16), [float16](/shared/glossary/#float16), and [FP8](/shared/glossary/#fp8) formats, developers learn that reducing precision significantly improves training speed and cuts memory usage. However, lower precision formats require careful handling—such as dynamic scaling or loss scaling—to avoid numerical issues like underflow and overflow that destabilize training.
