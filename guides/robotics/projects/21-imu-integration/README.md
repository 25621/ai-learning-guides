# IMU Integration

## Key Insight

An [IMU](/shared/glossary/#imu) reports motion, not position: its [accelerometer](/shared/glossary/#accelerometer) measures acceleration and its [gyroscope](/shared/glossary/#gyroscope) measures rotation rate, both hundreds of times a second. To recover orientation and position you must sum those rates over time — a process called [dead reckoning](/shared/glossary/#dead-reckoning) — but every reading carries a tiny bias, and the summation piles those biases up relentlessly. Fusing these inputs via direct integration causes errors to compound quadratically over time, whereas an error-state formulation tracks the small deviations from a nominal trajectory, making the integration far more robust to noise and sensor bias. This project highlights how fast this [drift](/shared/glossary/#drift) grows and why error-state integration is the standard choice for high-accuracy state estimation.
