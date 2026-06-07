# Scene-Cut Detector

## Key Insight

A training clip that accidentally spans a scene cut — the hard jump where a video switches shots — teaches the model a "motion" that is really just an editing splice, poisoning what it learns about how things actually move. This project performs [scene detection](/shared/glossary/#scene-detection) automatically by watching for sudden jumps in a frame's color histogram or deep-feature distance, then splits a long video into clean single-shot clips. Cutting on shot boundaries is one of the most important — and least glamorous — steps in building a usable video dataset, and skipping it quietly corrupts everything trained downstream.
