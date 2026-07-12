"""The reusable DQN core for Phase 3.

Project 12 writes DQN the naive way (one gradient step on the newest transition,
bootstrapping off the network being trained) and watches it fall over. This
module is the fixed version, factored so the rest of the phase can import it:

    buffer  -- where transitions live      (ReplayBuffer / OnlineBuffer)
    net     -- what maps state to Q-values (MLPQNet, or 14's ConvQNet)
    agent   -- how the target is built     (DQNAgent, or 18's C51Agent)
    loop    -- how the three are driven    (train_dqn)

Each later project swaps exactly one of those four and leaves the rest alone:
15 swaps the net (dueling) and a flag (double), 16 swaps the buffer (PER),
17 swaps all of them at once, 18 swaps the agent.

Everything is CPU-only on purpose. The nets here are a few thousand parameters
and the batch is 64 rows wide, so a GPU's kernel-launch overhead would cost more
than the arithmetic it saves; `torch.cuda` is deliberately never touched.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cpu")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------
# Buffers
# --------------------------------------------------------------------------

class Batch:
    """A sampled minibatch, already on-device as tensors."""

    __slots__ = ("s", "a", "r", "s2", "d", "w", "idx")

    def __init__(self, s, a, r, s2, d, w, idx):
        self.s, self.a, self.r, self.s2, self.d = s, a, r, s2, d
        self.w = w      # importance-sampling weights (all ones unless PER)
        self.idx = idx  # buffer slots, so PER can write back new priorities


class ReplayBuffer:
    """A fixed-size ring buffer of transitions, sampled uniformly at random.

    Stored as flat numpy arrays rather than a deque of tuples: sampling a batch
    is then one fancy-index per field instead of 64 Python-level tuple unpacks,
    which is what keeps the training loop's cost in the network rather than in
    the bookkeeping.

    `obs_dtype=np.uint8` turns on the memory trick real Atari DQN depends on.
    A stacked pixel observation held as float32 costs 4 bytes a pixel, and the
    buffer holds two of them per transition; at Atari's 1M transitions that is
    ~50 GB of RAM. Stored as bytes and divided by 255 at sample time it is a
    quarter of that. The MiniPong frames here are 4x20x20, so the same choice
    turns a 640 MB buffer into a 64 MB one -- which matters because a dozen of
    these run in parallel processes.
    """

    def __init__(self, capacity, obs_shape, obs_dtype=np.float32):
        self.capacity = int(capacity)
        self.obs_dtype = obs_dtype
        self.byte_obs = obs_dtype == np.uint8
        self.s = np.zeros((self.capacity, *obs_shape), dtype=obs_dtype)
        self.s2 = np.zeros((self.capacity, *obs_shape), dtype=obs_dtype)
        self.a = np.zeros(self.capacity, dtype=np.int64)
        self.r = np.zeros(self.capacity, dtype=np.float32)
        self.d = np.zeros(self.capacity, dtype=np.float32)
        self.pos = 0
        self.full = False

    def _encode(self, obs):
        # observations arrive in [0, 1]; quantize to bytes only if asked to
        return np.asarray(obs * 255.0, dtype=np.uint8) if self.byte_obs else obs

    def add(self, s, a, r, s2, d):
        i = self.pos
        self.s[i] = self._encode(s)
        self.s2[i] = self._encode(s2)
        self.a[i], self.r[i], self.d[i] = a, r, d
        self.pos = (i + 1) % self.capacity
        self.full = self.full or self.pos == 0

    def __len__(self):
        return self.capacity if self.full else self.pos

    def _obs(self, arr, idx):
        t = torch.as_tensor(arr[idx])
        return t.float().div_(255.0) if self.byte_obs else t

    def _to_batch(self, idx, w=None):
        n = len(idx)
        w = np.ones(n, dtype=np.float32) if w is None else w
        return Batch(
            self._obs(self.s, idx),
            torch.as_tensor(self.a[idx]),
            torch.as_tensor(self.r[idx]),
            self._obs(self.s2, idx),
            torch.as_tensor(self.d[idx]),
            torch.as_tensor(w),
            idx,
        )

    def sample(self, batch_size, rng):
        idx = rng.integers(0, len(self), size=batch_size)
        return self._to_batch(idx)

    def update_priorities(self, idx, td_errors):
        """No-op. PER (project 16) overrides this."""


class OnlineBuffer:
    """The 'buffer' project 12 uses: remembers only the newest transition.

    Same interface as ReplayBuffer, so `train_dqn` cannot tell the difference —
    which is the point. Turning experience replay off becomes a one-word change
    at the call site instead of a different training loop, and any behavior gap
    between the two is therefore attributable to replay alone.
    """

    def __init__(self, obs_shape, obs_dtype=np.float32):
        self.last = None
        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype
        self.n = 0

    def add(self, s, a, r, s2, d):
        self.last = (
            np.asarray(s, dtype=self.obs_dtype),
            np.int64(a),
            np.float32(r),
            np.asarray(s2, dtype=self.obs_dtype),
            np.float32(d),
        )
        self.n += 1

    def __len__(self):
        return 0 if self.last is None else 1

    def sample(self, batch_size, rng):
        s, a, r, s2, d = self.last
        return Batch(
            torch.as_tensor(s[None]),
            torch.as_tensor(np.array([a], dtype=np.int64)),
            torch.as_tensor(np.array([r], dtype=np.float32)),
            torch.as_tensor(s2[None]),
            torch.as_tensor(np.array([d], dtype=np.float32)),
            torch.ones(1),
            None,
        )

    def update_priorities(self, idx, td_errors):
        """No-op."""


# --------------------------------------------------------------------------
# Networks
# --------------------------------------------------------------------------

class MLPQNet(nn.Module):
    """state -> one Q-value per action. Two hidden layers is plenty for CartPole."""

    def __init__(self, obs_dim, n_actions, hidden=128, dueling=False):
        super().__init__()
        self.dueling = dueling
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        if dueling:
            self.value = nn.Linear(hidden, 1)
            self.adv = nn.Linear(hidden, n_actions)
        else:
            self.head = nn.Linear(hidden, n_actions)

    def forward(self, x):
        h = self.body(x)
        if not self.dueling:
            return self.head(h)
        v, a = self.value(h), self.adv(h)
        return v + a - a.mean(dim=-1, keepdim=True)


# --------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------

@dataclass
class Config:
    total_steps: int = 40_000
    gamma: float = 0.99
    lr: float = 2.5e-4
    batch_size: int = 64
    buffer_size: int = 50_000
    learning_starts: int = 1_000
    train_freq: int = 1
    target_freq: int = 500      # hard copy every N gradient steps; 0 disables
    tau: float = 0.0            # >0 switches to Polyak (soft) target updates
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_frac: float = 0.5  # fraction of training spent annealing epsilon
    double: bool = False
    n_step: int = 1
    grad_clip: float = 10.0
    use_target: bool = True
    huber: bool = True
    eval_every: int = 2_000
    eval_episodes: int = 10
    eval_eps: float = 0.02
    seed: int = 0
    extra: dict = field(default_factory=dict)


class DQNAgent:
    """Standard DQN: epsilon-greedy actor, Huber loss on the bootstrapped target.

    `use_target=False` aims the bootstrap at the online network itself, which is
    exactly the pathology project 12 exhibits and project 13 measures.
    """

    def __init__(self, net_fn, n_actions, cfg):
        self.cfg = cfg
        self.n_actions = n_actions
        # Seed BEFORE the network exists. torch seeds its default generator from OS
        # entropy on first use, so building the net first would give it a random
        # initialization that no `seed` argument could reproduce -- and when runs are
        # spread across worker processes, the same config would then land on a
        # different curve every time the script is run.
        set_seed(cfg.seed)
        self.net = net_fn().to(DEVICE)
        if cfg.use_target:
            self.target = net_fn().to(DEVICE)
            self.target.load_state_dict(self.net.state_dict())
            for p in self.target.parameters():
                p.requires_grad_(False)
        else:
            self.target = self.net  # bootstrap off the network being trained
        self.optim = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.updates = 0

    @torch.no_grad()
    def q_values(self, obs_batch):
        return self.net(torch.as_tensor(obs_batch, dtype=torch.float32))

    def act(self, obs, eps, rng):
        if rng.random() < eps:
            return int(rng.integers(self.n_actions))
        q = self.q_values(obs[None])
        return int(q.argmax(dim=-1).item())

    def _target_q(self, batch):
        """r + gamma^n * Q(s', a*) * (1 - done), with a* chosen per the double flag."""
        cfg = self.cfg
        with torch.no_grad():
            if cfg.double:
                # online net picks the action, target net scores it
                a_star = self.net(batch.s2).argmax(dim=-1, keepdim=True)
                q_next = self.target(batch.s2).gather(-1, a_star).squeeze(-1)
            else:
                q_next = self.target(batch.s2).max(dim=-1).values
            return batch.r + (cfg.gamma ** cfg.n_step) * q_next * (1.0 - batch.d)

    def update(self, batch):
        cfg = self.cfg
        target = self._target_q(batch)
        pred = self.net(batch.s).gather(-1, batch.a[:, None]).squeeze(-1)
        td = pred - target
        if cfg.huber:
            per_sample = F.smooth_l1_loss(pred, target, reduction="none")
        else:
            per_sample = 0.5 * td.pow(2)
        loss = (batch.w * per_sample).mean()   # w == 1 unless PER is in play

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.net.parameters(), cfg.grad_clip)
        self.optim.step()
        self.updates += 1
        self.sync_target()
        return float(loss.item()), td.detach().abs().numpy(), float(pred.mean().item())

    def sync_target(self):
        cfg = self.cfg
        if not cfg.use_target:
            return
        if cfg.tau > 0.0:                                   # Polyak / soft update
            with torch.no_grad():
                for p, pt in zip(self.net.parameters(), self.target.parameters()):
                    pt.mul_(1 - cfg.tau).add_(cfg.tau * p)
        elif cfg.target_freq and self.updates % cfg.target_freq == 0:  # hard copy
            self.target.load_state_dict(self.net.state_dict())


# --------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------

def linear_schedule(start, end, duration, t):
    frac = min(1.0, t / max(1, duration))
    return start + frac * (end - start)


def evaluate(env, agent, episodes, eps, rng):
    """Greedy-ish rollouts with a tiny epsilon (CartPole and MiniPong are both
    deterministic enough that a pure argmax policy can get stuck in a loop)."""
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
        done, total = False, 0.0
        while not done:
            a = agent.act(np.asarray(obs, dtype=np.float32), eps, rng)
            obs, r, term, trunc, _ = env.step(a)
            total += r
            done = term or trunc
        returns.append(total)
    return float(np.mean(returns))


def train_dqn(env, agent, buffer, cfg, eval_env=None, progress=None, probe=None):
    """One DQN run. Returns a history dict; the caller does the plotting.

    Also records `q_pred` (mean predicted Q on training batches) alongside the
    return curve, because the interesting failures in this phase show up in the
    Q-values long before they show up in the score.

    `probe(agent, step) -> dict` is called at every evaluation checkpoint and its
    scalars are appended to the history under their own keys. Project 15 uses it
    to measure Q-value overestimation mid-run without forking the training loop.
    """
    rng = np.random.default_rng(cfg.seed)
    set_seed(cfg.seed)

    hist = {"step": [], "ep_return": [], "ep_step": [],
            "eval_step": [], "eval_return": [], "q_pred": [], "loss": []}

    obs, _ = env.reset(seed=cfg.seed)
    obs = np.asarray(obs, dtype=np.float32)
    ep_return, decay_steps = 0.0, int(cfg.eps_decay_frac * cfg.total_steps)

    for step in range(1, cfg.total_steps + 1):
        eps = linear_schedule(cfg.eps_start, cfg.eps_end, decay_steps, step)
        a = agent.act(obs, eps, rng)
        nxt, r, term, trunc, _ = env.step(a)
        nxt = np.asarray(nxt, dtype=np.float32)
        ep_return += r

        # `term` (the pole fell) bootstraps to 0; `trunc` (time limit) does not —
        # the episode's value did not actually end, we just stopped looking.
        buffer.add(obs, a, r, nxt, float(term))
        obs = nxt

        if term or trunc:
            hist["ep_return"].append(ep_return)
            hist["ep_step"].append(step)
            obs, _ = env.reset()
            obs = np.asarray(obs, dtype=np.float32)
            ep_return = 0.0

        if step >= cfg.learning_starts and step % cfg.train_freq == 0 and len(buffer) > 0:
            batch = buffer.sample(cfg.batch_size, rng)
            loss, td, q_mean = agent.update(batch)
            buffer.update_priorities(batch.idx, td)
            if step % 100 == 0:
                hist["step"].append(step)
                hist["q_pred"].append(q_mean)
                hist["loss"].append(loss)

        if cfg.eval_every and step % cfg.eval_every == 0:
            if eval_env is not None:
                score = evaluate(eval_env, agent, cfg.eval_episodes, cfg.eval_eps, rng)
                hist["eval_step"].append(step)
                hist["eval_return"].append(score)
                if progress:
                    print(f"  [{progress}] step {step:>6}  eval {score:7.2f}  eps {eps:.2f}")
            if probe is not None:
                hist.setdefault("probe_step", []).append(step)
                for key, value in probe(agent, step).items():
                    hist.setdefault(key, []).append(value)

    return hist
