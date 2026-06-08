# Keyframe Interpolation

## Key Insight

This project builds a long clip the way an animation studio does: first draw the [keyframes](/shared/glossary/#keyframe) — here, 4 frames spaced 5 seconds apart that pin down how the scene should look at those moments — then fill the gaps between them. The filling is done by an [image-to-video (I2V)](/shared/glossary/#i2v) model or a [frame-interpolation](/shared/glossary/#frame-interpolation) model, which only has to invent the short motion between two anchors it can already see rather than a whole scene from nothing. This is the simplest form of [hierarchical generation](/shared/glossary/#hierarchical-generation): decide the big structure first and the details second, which keeps a long video far more coherent than generating it straight through. The trade-off is that the keyframes must be chosen well — pick two that are too different and no interpolation can bridge them smoothly.
