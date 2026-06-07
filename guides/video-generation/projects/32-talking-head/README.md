# Talking Head

## Key Insight

A [talking-head](/shared/glossary/#talking-head) model takes a single portrait photo and an audio clip and produces a video of that person speaking the audio, with lips, jaw, and head moving in sync — the audio drives the motion while the photo fixes the identity. This project runs a pretrained model such as [EMO](/shared/glossary/#talking-head) or [Hallo](/shared/glossary/#talking-head) on a portrait-plus-audio pair, then [fine-tunes](/shared/glossary/#fine-tuning) it for one specific speaker so the mouth shapes and mannerisms match that person more faithfully. The core challenge is [lip sync](/shared/glossary/#talking-head): the mouth must form the right shape for each sound at the right instant, which is why these models extract audio features (often with a speech encoder like wav2vec) and align them to facial motion frame by frame.
