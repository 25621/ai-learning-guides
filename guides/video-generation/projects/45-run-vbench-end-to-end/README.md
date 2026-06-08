# Run VBench End to End

## Key Insight

Evaluating video generation is even harder than evaluating images, and a single number like [FVD (Fréchet Video Distance)](/shared/glossary/#fvd) correlates poorly with what people actually like. [VBench](/shared/glossary/#vbench) breaks the vague question "is this video good?" into many separate axes — subject consistency, motion smoothness, [aesthetic quality](/shared/glossary/#aesthetic-score), text alignment, and more — and scores each one, so you learn *which* aspect a model is weak at instead of getting one blurry verdict. This project runs an [open](/shared/glossary/#open-model) [text-to-video (T2V)](/shared/glossary/#t2v) model through the full VBench suite and reproduces a published [leaderboard](/shared/glossary/#leaderboard) number — which is itself a lesson in how fragile "just reproduce the benchmark" turns out to be once prompt wording, sampling settings, and clip counts all start to matter.
