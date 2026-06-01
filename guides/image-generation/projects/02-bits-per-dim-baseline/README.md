# Bits-per-Dim Baseline

## Key Insight

[Bits per dimension](/shared/glossary/#bpd-bits-per-dimension) measures how many bits a model needs, on average, to store each number in an image — lower is better, because a good model is "surprised" less often. This project computes the score for the dumbest possible model: one that thinks every pixel value from 0 to 255 is equally likely. That gives exactly 8 bits per dimension, the "no model at all" ceiling. Any real generative model must beat this number to prove it learned anything, so it is the first baseline you should always compute.
