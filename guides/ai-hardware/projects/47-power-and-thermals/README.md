# Power and Thermals

## Key Insight

Running multiple high-performance [GPUs](/shared/glossary/#gpu) in a single workstation creates serious power and heat challenges that can silently degrade performance through [thermal throttling](/shared/glossary/#thermal-throttling) — the chip automatically slowing itself to avoid overheating. [Undervolting](/shared/glossary/#undervolting) lets developers lower each GPU's voltage-frequency curve to reduce power draw (often by 50–100 watts per card) while maintaining stable clock speeds and nearly identical [throughput](/shared/glossary/#throughput). This project teaches how to measure real-time power consumption with `nvidia-smi`, monitor junction temperatures, and configure system-level power limits so that a multi-GPU setup runs continuously under heavy training loads without exceeding the cooling system's capacity or the power supply's rated output.
