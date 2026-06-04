# Mel Spectrogram Pipeline

## Key Insight

Before a neural network can "hear," sound has to become a picture. A [mel spectrogram](/shared/glossary/#mel-spectrogram) is that picture — a 2D (time × frequency) image produced by a [Short-Time Fourier Transform](/shared/glossary/#stft) and then bent onto a perceptual pitch scale — and once audio is an image, the very same [CNN](/shared/glossary/#cnn) machinery built for vision can process it directly. Building the pipeline end to end on a 10-second clip shows why almost every audio model starts here rather than with the raw waveform: a minute of sound is millions of samples, but its mel spectrogram is a compact, perceptually meaningful grid a small network can chew through.
