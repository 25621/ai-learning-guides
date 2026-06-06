# Frame Extractor

## Key Insight

There are two different ways to pull N frames from a clip, and they are not the same: sampling evenly across the *frame indices* versus evenly across *time* (a fixed [frame rate](/shared/glossary/#frame-rate-fps)). For a clip with steady motion they look alike, but for one that mixes a long still shot and a burst of fast action, index-sampling oversamples the boring still part while fps-sampling keeps the motion evenly spaced in real seconds. Picking the wrong one feeds your model a distorted picture of how fast the world moves. This project samples both ways on fast and slow scenes so you can see the difference with your own eyes.
