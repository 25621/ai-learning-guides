# Mel Spectrogram From Scratch

## Key Insight

Libraries like `torchaudio` hand you a [mel spectrogram](/shared/glossary/#mel-spectrogram) in a single call, but rebuilding one by hand — windowing the raw waveform, running a [Short-Time Fourier Transform](/shared/glossary/#stft), then applying a *mel filterbank* — shows there is no magic inside. That [filterbank](/shared/glossary/#mel-spectrogram) is just a fixed matrix of triangular weights, so the famous "perceptual" step is one matrix multiply that folds the [STFT](/shared/glossary/#stft)'s many evenly-spaced frequency rows down into a handful of mel bands spaced the way human hearing is. Doing it from scratch on a 10-second clip turns the everyday habit of treating audio as an image into something you understand rather than trust.
