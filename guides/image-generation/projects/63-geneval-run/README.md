# GenEval Run

## Key Insight

Pretty pictures aren't the same as *correct* pictures, and [GenEval](/shared/glossary/#geneval) measures the difference by checking, with an object detector, whether a generated image actually contains the right number, color, and arrangement of objects the prompt asked for. Running an open [text-to-image](/shared/glossary/#stable-diffusion) model through this [benchmark](/shared/glossary/#benchmark) surfaces its real weaknesses — miscounting, swapped attributes, ignored spatial relations — that beauty metrics like [FID](/shared/glossary/#fid) never reveal. The takeaway is that compositional adherence is a separate axis from raw image quality, and you must measure it on purpose.
