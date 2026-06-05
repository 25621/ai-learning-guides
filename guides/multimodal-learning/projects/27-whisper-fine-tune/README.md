# Whisper Fine-Tune

## Key Insight

[Whisper](/shared/glossary/#whisper) is a strong general-purpose [ASR (Automatic Speech Recognition)](/shared/glossary/#asr-automatic-speech-recognition) model, but it was trained mostly on common languages and clean web audio, so it stumbles on a rare dialect, a heavy accent, or domain jargon like medical terms. [Fine-tuning](/shared/glossary/#fine-tuning) the small Whisper checkpoint on a few hours of in-domain (audio, transcript) pairs nudges its [mel-spectrogram](/shared/glossary/#mel-spectrogram)-to-text mapping toward that target without paying for the original 680,000-hour training run. The payoff is largest exactly where the base model is weakest — low-resource languages — because there the pretrained model has the least to go on and the most to gain from a little focused data.
