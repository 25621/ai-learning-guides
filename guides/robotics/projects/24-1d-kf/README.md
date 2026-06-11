# 1D Kalman Filter

## Key Insight

Fusing multiple noisy sensors mathematically yields a far more accurate estimate than any single sensor can provide. By using a [Kalman Filter (KF)](/shared/glossary/#kf) to combine two thermometer readings, the joint estimate's uncertainty shrinks below both individual sensor variances. This [covariance](/shared/glossary/#covariance) contraction demonstrates that the filter is not just averaging the data, but actively extracting certain information from uncertainty.
