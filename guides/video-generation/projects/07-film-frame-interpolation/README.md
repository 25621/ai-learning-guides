# FILM Frame Interpolation

## Key Insight

[Frame interpolation](/shared/glossary/#frame-interpolation) — inventing the in-between frames that turn a choppy clip into a smooth or slow-motion one — is the gentlest version of video generation, because the model only has to fill in the short motion between two real frames it can already see rather than imagine a whole scene from nothing. This project runs a pretrained [FILM (Frame Interpolation for Large Motion)](/shared/glossary/#film) model to interpolate between two real frames and watches where it breaks. The interesting part is the artifacts: when an object moves a long way between the two frames, even a strong model smears, ghosts, or tears it, because it has to guess a motion path it was never shown. Spotting those failures builds intuition for why large, fast motion is the central difficulty that all of video generation keeps running into.
