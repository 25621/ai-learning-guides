# 2-GPU Build Plan

## Key Insight

Designing a multi-[GPU](/shared/glossary/#gpu) workstation requires careful balancing of power, cooling, and [PCIe](/shared/glossary/#pcie) lane allocation to prevent hardware [bottlenecks](/shared/glossary/#bottleneck). A system that looks fast on paper can underperform if the motherboard cannot supply enough PCIe lanes for both GPUs to communicate at full bandwidth, or if the power supply unit cannot sustain the combined draw under heavy training loads. By planning a build around specific compute and [VRAM](/shared/glossary/#vram) requirements — and checking that every component (CPU, motherboard, PSU, cooling) supports the target workload — developers learn how each hardware choice directly affects [throughput](/shared/glossary/#throughput), memory capacity, and long-term stability.
