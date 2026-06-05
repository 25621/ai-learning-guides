# Whisper Encoder Reuse

## Key Insight

[Whisper](/shared/glossary/#whisper) was trained to turn speech into text, but its encoder — the front half that digests a [mel spectrogram](/shared/glossary/#mel-spectrogram) into a sequence of rich audio [embeddings](/shared/glossary/#embedding) — is useful far beyond transcription. Throw away the decoder, freeze the encoder, and you have a strong pretrained audio feature extractor for free: a tiny classifier trained on those frozen embeddings can recognize languages, emotions, or sound events with a fraction of the data needed to learn audio from scratch. It is the same "reuse a pretrained backbone, train a small head" move as a vision [linear probe](/shared/glossary/#linear-probe) — representations learned for one task are quietly general.
