# Real-Time Loop Drill

## Key Insight

A robot's low-level controller requires deterministic, high-frequency execution (e.g., a 1 kHz control loop) to maintain stability and react to physical disturbances. Using Linux with the [PREEMPT_RT](/shared/glossary/#preempt_rt) patch turns the operating system into a soft or hard real-time system, ensuring that the control thread is scheduled with minimal and bounded [jitter](/shared/glossary/#jitter). Running this drill compares the latency and scheduling consistency of vanilla Linux against a real-time kernel, exposing how CPU load spikes can cause catastrophic timing failures on standard operating systems.
