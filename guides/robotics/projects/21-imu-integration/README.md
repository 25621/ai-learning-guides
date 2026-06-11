# IMU Integration

## Key Insight

An [IMU](/shared/glossary/#imu) reports motion, not position: its [accelerometer](/shared/glossary/#accelerometer) measures acceleration and its [gyroscope](/shared/glossary/#gyroscope) measures rotation rate, both hundreds of times a second. To recover orientation and position you must sum those rates over time — a process called [dead reckoning](/shared/glossary/#dead-reckoning) — but every reading carries a tiny bias, and the summation piles those biases up relentlessly. The lesson this project drives home is how fast the resulting [drift](/shared/glossary/#drift) grows: orientation wanders off within seconds, and position, which requires summing *twice*, drifts even faster — which is exactly why an IMU is never trusted alone but fused with a camera or GPS that can periodically correct it.
