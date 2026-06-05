# EnCodec Tour

## Key Insight

[EnCodec](/shared/glossary/#neural-codec) is Meta's neural codec for audio: it squeezes a waveform into a short stream of discrete [audio tokens](/shared/glossary/#token-visualaudio) and decodes them back into sound. Running the same clip through it at several *bitrates* and listening to the reconstructions makes the core trade-off audible — more tokens per second rebuild richer, cleaner audio, while fewer tokens save space but blur detail and add artifacts. This matters beyond compression: once a second of sound is just a handful of tokens, a [transformer](/shared/glossary/#transformer) can generate or continue audio with the very same next-token machinery it uses for text.
