# Action-Conditioned Video

## Key Insight

A plain video model predicts what happens next; the moment you also feed it an action and ask it to predict what happens next *because of that action*, it becomes a [world model](/shared/glossary/#world-model). This project adds [action conditioning](/shared/glossary/#action-conditioning) to a small [video diffusion model](/shared/glossary/#diffusion-model) — a discrete input of, say, four game buttons injected alongside the noisy frames — and trains it on a simple game's recorded play, where every frame is paired with the button that was pressed. Once trained, pressing *up* versus *left* sends the predicted future down different branches, which is the whole point: the action is the steering wheel. It is the smallest possible step from "generate a clip" to "simulate an interactive world."
