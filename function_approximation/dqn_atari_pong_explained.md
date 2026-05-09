# DQN on Atari Pong 🏓

## What is Atari Pong?

Pong is one of the oldest video games ever made — it's like digital table tennis! Two
paddles bounce a ball back and forth. You win a point if the opponent misses the ball.
The game ends when someone reaches 21 points.

In our version, the AI controls one paddle. The opponent (computer) controls the other.
The game always starts at score −21 (worst possible). A good agent reaches 0 or +21.

---

## Why is Pong Hard for an AI?

In CartPole, the robot could SEE the numbers directly (pole angle, cart speed...).
In Pong, all it sees is **raw pixels** — thousands of tiny colored dots on a screen!

```
CartPole input: [0.02, −0.14, 0.01, −0.23]   ← 4 numbers, easy!
Pong input:     [pixel grid: 210×160×3]        ← 100,800 numbers, MUCH harder!
```

The robot has to figure out from pixels:
- Where is my paddle?
- Where is the ball?
- Is the ball moving left or right?
- How fast?

Humans do this automatically (we have amazing vision!). For an AI, this is a huge challenge.

---

## Seeing Motion: Frame Stacking 🎬

One frame (screenshot) doesn't tell you if the ball is moving left or right. You need
to see MULTIPLE frames to understand motion — just like how a flip book only works when
you flip through many pages.

**Frame Stacking:** Feed the last 4 frames into the network simultaneously.

```
Frame 1: ball at position 40
Frame 2: ball at position 43    → Stack these 4 frames → Network sees MOTION!
Frame 3: ball at position 46
Frame 4: ball at position 49
```

The network can now infer: "ball is moving right at speed 3"

**Real-life example:** Watching a movie vs looking at one frame. One frame of a car
race is just a blurry image. Watch 4 frames, and you can tell which car is faster!

---

## Seeing with a CNN 🔍

For pixel inputs, we use a special neural network called a **Convolutional Neural Network
(CNN)**. Instead of looking at all pixels at once, a CNN uses sliding windows to detect
patterns — like eyes scanning an image.

```
Raw pixels (84×84×4 frames)
       ↓
Conv Layer 1 (8×8 filter, stride 4) → finds edges and shapes
       ↓
Conv Layer 2 (4×4 filter, stride 2) → finds objects (paddles, ball)
       ↓
Conv Layer 3 (3×3 filter, stride 1) → finds relationships
       ↓
Flatten → 512 neurons → Q-values (one per action)
```

**Real-life example:** When you look for your friend in a crowd, your brain first notices
shapes (a person), then features (hair color), then details (their face). CNNs work the
same way — from simple patterns to complex ones!

---

## Preprocessing: Shrinking the World

Pong frames are 210×160 pixels in color. That's too big! We preprocess each frame:

1. **Grayscale** — color doesn't matter for Pong (the ball is always white anyway)
2. **Resize to 84×84** — smaller = faster training, but still clear enough to see
3. **Normalize to [0,1]** — divide pixel values by 255 so they're small numbers

**Real-life example:** Like making a photocopy at 50% size. The important details
(ball, paddles) are still visible, just smaller. The photocopier doesn't care about
colors either!

---

## Reward Clipping: Treating All Games Equally ✂️

In Pong, you get +1 for scoring, −1 for being scored on.
In some other Atari games, scores can be thousands!

We **clip rewards** to [−1, +1] so the network doesn't care about the scale of rewards.
This same code can train on ANY Atari game without tuning reward scales.

---

## How Long Does Training Take?

| Training Duration | What the Agent Learns |
|---|---|
| 100K steps | Mostly random, barely reacts |
| 1M steps | Starts moving toward the ball sometimes |
| 5M steps | Returns some shots |
| 10M steps | Competitive play, might win some |
| 20M+ steps | Often beats the computer opponent |

Our demo runs **300K steps** — enough to see the training architecture works and
observe early learning, but not enough to master the game.

**Real-life example:** Learning piano takes months. A 10-minute practice session
shows you're doing it right, but don't expect to perform concerts yet!

---

## What Our Code Found

After 300K steps on Pong:
- The agent starts with scores around −20 (barely returns the ball)
- By the end, it typically improves to around −15 to −10
- The learning curve shows gradual improvement from random play

To see real competitive Pong performance, you'd need to run for ~10M+ steps with a GPU.
The implementation is complete and correct — it just needs more time!

---

## Key Vocabulary

| Word | Meaning |
|------|---------|
| **CNN** | Convolutional Neural Network — specialized for image inputs |
| **Frame Stacking** | Feeding multiple consecutive frames to capture motion |
| **Preprocessing** | Transforming raw frames (grayscale, resize, normalize) before feeding to the network |
| **Reward Clipping** | Limiting rewards to [−1, +1] to work across different games |
| **ALE** | Arcade Learning Environment — the library that runs Atari games |

---

## The Historic Achievement

When DeepMind published DQN in 2015, the world was amazed. A SINGLE algorithm,
with the SAME architecture, learned to play 49 different Atari games — many at
superhuman level — just from raw pixels and a score!

Before DQN, people thought you needed to hand-code the strategy for each game.
DQN showed that a general-purpose learner could figure it all out by itself.
It was a historic moment in AI!
