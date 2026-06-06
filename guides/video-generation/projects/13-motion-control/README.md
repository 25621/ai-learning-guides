# Motion Control

## Key Insight

An [image-to-video](/shared/glossary/#i2v) model trained on raw clips gives you no say over *how much* things move — some outputs barely twitch, others thrash. A [motion score](/shared/glossary/#motion-score) (also called a *motion bucket*) fixes this by feeding the model a single number at training time that measures how much motion each training clip actually contains — typically derived from [optical-flow](/shared/glossary/#optical-flow) magnitude, i.e. how far pixels travel between frames — so that at inference you can dial that number up or down to request subtle or energetic motion. This project adds that input to the [Tiny I2V model](../12-tiny-i2v-model/README.md) and checks that the model truly learned the association: low scores should produce gentle, animated-still motion, high scores dramatic movement. It is the simplest *control surface* for video — one extra knob that separates *how much it moves* from *what is in it*.
