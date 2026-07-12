"""A Pong you can afford, plus the real Atari preprocessing pipeline.

Two things live here, and it is worth being blunt about why.

REAL Atari Pong needs roughly a million frames before DQN scores its first point,
which is hours of CPU. That does not fit in ten minutes, and pretending otherwise
would be a lie told with a learning curve. So this module has:

  * `AtariPipeline` -- the genuine preprocessing chain used on ALE Pong
    (grayscale, crop, downsample, frame-skip with max-pooling, frame stacking,
    reward clipping). It runs against real `ALE/Pong-v5` when `ale_py` is
    installed, and project 14 uses it to *show* what the network actually sees.

  * `MiniPong` -- Pong against a wall, rendered as real pixels on a 20x20 grid.
    Same shape of problem (learn control from raw pixels, infer ball velocity
    from consecutive frames, sparse-ish +1/-1 reward), small enough that the
    identical DQN solves it in about a minute.

Everything downstream (projects 15 and 17) trains on MiniPong; the pipeline that
touches it is the same pipeline you would point at the ALE.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "13-add-a-replay-buffer"))

GRID = 20          # MiniPong playfield: GRID x GRID pixels
STACK = 4          # how many consecutive frames the agent sees at once
PADDLE_H = 4       # paddle height in pixels
MAX_RALLIES = 10   # cap an episode so a perfect agent still terminates (return tops out at +10)


# --------------------------------------------------------------------------
# MiniPong: Pong against a wall
# --------------------------------------------------------------------------

class MiniPong:
    """Pong with the opponent replaced by a wall. Gymnasium-style API.

    The ball bounces off the top, bottom, and left walls; the agent's paddle is
    the right-hand column. Return the ball and score +1; miss it and score -1 and
    the episode ends. An episode also ends after MAX_RALLIES successful returns,
    so a good agent's episode return tops out at +20.

    Why a wall instead of an opponent: with a scripted opponent, roughly half of
    every episode is spent watching the ball travel away from you, and the reward
    for a good move arrives dozens of steps later. Against a wall, every rally is
    an honest test of the same skill, and credit lands ~20 steps after the action.
    Same lesson, a fifth of the compute.

    The observation is a single GRID x GRID float32 frame with the ball at 1.0 and
    the paddle at 0.6. Crucially, one frame does NOT reveal which way the ball is
    travelling -- that information exists only *between* frames, which is exactly
    why Atari DQN stacks four of them. Project 14 tests that claim directly.
    """

    n_actions = 3            # 0 = stay, 1 = up, 2 = down
    obs_shape = (GRID, GRID)

    def __init__(self, seed=0, reward_scale=1.0, rally_bonus=0.0):
        self.rng = np.random.default_rng(seed)
        self.reward_scale = reward_scale   # >1 makes the reward magnitudes Atari-like
        self.rally_bonus = rally_bonus     # a rare, large reward, to motivate clipping
        self._frame = np.zeros((GRID, GRID), dtype=np.float32)

    # -- dynamics ----------------------------------------------------------
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.ball = np.array([self.rng.integers(2, GRID - 2),
                              self.rng.integers(2, GRID // 2)], dtype=np.int64)  # (row, col)
        self.vel = np.array([self.rng.choice([-1, 1]), 1], dtype=np.int64)       # always toward the paddle
        self.paddle = GRID // 2 - PADDLE_H // 2
        self.rallies = 0
        return self._observe(), {}

    def _observe(self):
        f = self._frame
        f.fill(0.0)
        f[self.paddle:self.paddle + PADDLE_H, GRID - 1] = 0.6
        f[self.ball[0], self.ball[1]] = 1.0
        return f.copy()

    def step(self, action):
        # paddle first
        if action == 1:
            self.paddle = max(0, self.paddle - 2)
        elif action == 2:
            self.paddle = min(GRID - PADDLE_H, self.paddle + 2)

        r, c = self.ball + self.vel
        # bounce off top / bottom
        if r < 0 or r > GRID - 1:
            self.vel[0] *= -1
            r = self.ball[0] + self.vel[0]
        # bounce off the left wall
        if c < 0:
            self.vel[1] *= -1
            c = self.ball[1] + self.vel[1]

        reward, terminated = 0.0, False
        if c > GRID - 1:                      # the ball reached the paddle column
            if self.paddle <= r < self.paddle + PADDLE_H:
                self.vel[1] = -1              # returned
                c = GRID - 1
                self.rallies += 1
                reward = 1.0
                if self.rally_bonus and self.rallies % 5 == 0:
                    reward += self.rally_bonus   # the rare jackpot
                if self.rallies >= MAX_RALLIES:
                    terminated = True
            else:
                reward, terminated = -1.0, True  # missed

        self.ball = np.array([np.clip(r, 0, GRID - 1), np.clip(c, 0, GRID - 1)])
        return self._observe(), reward * self.reward_scale, terminated, False, {}


class FrameStack:
    """Stack the last k frames into one (k, H, W) observation.

    This is the wrapper that hands the network a sense of motion. With k = 1 the
    agent sees a still photograph of a ball and cannot tell an incoming ball from
    a departing one; with k = 4 the ball's trail encodes both direction and speed.
    """

    def __init__(self, env, k=STACK):
        self.env = env
        self.k = k
        self.n_actions = env.n_actions
        self.obs_shape = (k, *env.obs_shape)
        self.frames = np.zeros(self.obs_shape, dtype=np.float32)

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        self.frames[:] = 0.0
        self.frames[-1] = obs
        return self.frames.copy(), info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        self.frames = np.roll(self.frames, -1, axis=0)
        self.frames[-1] = obs
        return self.frames.copy(), r, term, trunc, info


class ClipReward:
    """Atari's `sign(r)` trick: keep the sign, throw away the magnitude.

    One shared learning rate has to serve every game, and Q-values scale with the
    reward. Clipping to {-1, 0, +1} makes the gradients comparable across games —
    at the cost of making the agent indifferent between a small win and a jackpot.
    """

    def __init__(self, env):
        self.env = env
        self.n_actions = env.n_actions
        self.obs_shape = env.obs_shape

    def reset(self, seed=None):
        return self.env.reset(seed=seed)

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        info = dict(info, raw_reward=r)
        return obs, float(np.sign(r)), term, trunc, info


def make_minipong(seed=0, stack=STACK, clip=False, reward_scale=1.0, rally_bonus=0.0):
    env = MiniPong(seed=seed, reward_scale=reward_scale, rally_bonus=rally_bonus)
    if clip:
        env = ClipReward(env)
    return FrameStack(env, k=stack)


# --------------------------------------------------------------------------
# The network that reads pixels
# --------------------------------------------------------------------------

class ConvQNet(nn.Module):
    """A small DQN torso: two conv layers, then a linear head.

    Nature-DQN's torso (32-64-64 channels, 8x8/4x4/3x3 kernels) is sized for
    84x84 Atari frames. On a 20x20 MiniPong frame the same design would be mostly
    padding, so this is the same idea shrunk to fit: strided convs that turn
    pixels into a small feature vector, then Q-values per action. `dueling` splits
    the head into value and advantage streams (project 15).
    """

    def __init__(self, in_frames=STACK, n_actions=3, grid=GRID, dueling=False):
        super().__init__()
        self.dueling = dueling
        self.features = nn.Sequential(
            nn.Conv2d(in_frames, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),  # 20 -> 10
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),         # 10 -> 5
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.features(torch.zeros(1, in_frames, grid, grid)).shape[1]
        if dueling:
            self.value = nn.Sequential(nn.Linear(n_flat, 128), nn.ReLU(), nn.Linear(128, 1))
            self.adv = nn.Sequential(nn.Linear(n_flat, 128), nn.ReLU(), nn.Linear(128, n_actions))
        else:
            self.head = nn.Sequential(nn.Linear(n_flat, 128), nn.ReLU(),
                                      nn.Linear(128, n_actions))

    def forward(self, x):
        h = self.features(x)
        if not self.dueling:
            return self.head(h)
        v, a = self.value(h), self.adv(h)
        # subtracting the mean advantage fixes the identifiability of V vs A:
        # without it, adding c to V and subtracting c from every A changes nothing.
        return v + a - a.mean(dim=-1, keepdim=True)


# --------------------------------------------------------------------------
# The real thing: ALE Pong preprocessing
# --------------------------------------------------------------------------

class AtariPipeline:
    """The genuine Atari wrapper chain, implemented in numpy (no OpenCV needed).

    Order matters, and every step is a fix for a specific property of the console:

      max-pool over 2 frames   Atari sprites flicker on alternate frames; taking
                               the elementwise max of the last two raw frames makes
                               them all visible at once.
      frame skip (4)           the same action is repeated for 4 frames. 60 Hz
                               decisions are wasted compute; 15 Hz is plenty.
      grayscale                color carries almost no signal in Pong; dropping it
                               cuts the input by 3x.
      crop + downsample        strip the scoreboard, halve the resolution.
      stack 4                  see motion, not stills.
      clip reward to sign      one learning rate across all 57 games.

    Requires `ale_py` (pip install ale_py). Used by project 14 only to *visualize*
    what the agent would see; training on it is a multi-hour job, not a ten-minute one.
    """

    def __init__(self, game="ALE/Pong-v5", skip=4, stack=STACK, seed=0):
        import ale_py                      # noqa: F401  (registers the ALE envs)
        import gymnasium as gym

        gym.register_envs(ale_py)
        # frameskip=1: we do our own skipping, so we can max-pool across the skipped frames
        self.env = gym.make(game, frameskip=1)
        self.skip = skip
        self.stack = stack
        self.seed = seed
        self.n_actions = self.env.action_space.n
        self.obs_shape = (stack, 84, 84)
        self._raw = None

    @staticmethod
    def _preprocess(frame):
        """(210, 160, 3) uint8 RGB  ->  (84, 84) float32 grayscale in [0, 1]."""
        gray = frame @ np.array([0.299, 0.587, 0.114], dtype=np.float32)  # luminance
        gray = gray[34:194]                       # crop the scoreboard and the floor
        small = gray[::2, ::2]                    # 160x160 -> 80x80 (nearest neighbour)
        out = np.zeros((84, 84), dtype=np.float32)
        out[2:82, 2:82] = small                   # pad back to the canonical 84x84
        return out / 255.0

    def reset(self):
        raw, _ = self.env.reset(seed=self.seed)
        self._raw = raw
        self.frames = np.zeros(self.obs_shape, dtype=np.float32)
        self.frames[-1] = self._preprocess(raw)
        return self.frames.copy()

    def step(self, action):
        """One agent step = `skip` console frames, max-pooled, reward summed then clipped."""
        total, term, trunc = 0.0, False, False
        buf = []
        for _ in range(self.skip):
            raw, r, term, trunc, _ = self.env.step(action)
            buf.append(raw)
            total += r
            if term or trunc:
                break
        pooled = np.maximum(buf[-1], buf[-2]) if len(buf) >= 2 else buf[-1]
        self._raw = pooled
        self.frames = np.roll(self.frames, -1, axis=0)
        self.frames[-1] = self._preprocess(pooled)
        return self.frames.copy(), float(np.sign(total)), term, trunc, {"raw_reward": total}

    @property
    def raw_frame(self):
        return self._raw

    def close(self):
        self.env.close()
