"""
Explore the PettingZoo multi-agent API.

PettingZoo is to multi-agent RL what Gymnasium is to single-agent RL: a
common interface so every environment looks the same.  Two API styles:

  * AEC      (Agent-Environment-Cycle): one agent acts at a time, like a
             turn-based game.  Iterate with env.agent_iter().
  * Parallel: all agents act at the same time, like a real-time game.
             step(actions_dict) -> (obs_dict, reward_dict, ...)

This script demonstrates BOTH styles using a tiny, hand-coded environment
that follows the PettingZoo conventions exactly, so it works whether or not
the `pettingzoo` package is installed in your venv.  At the bottom we also
try to import the real PettingZoo and roll out one of its built-in
environments (rock_paper_scissors_v2) if available.

We then train independent tabular Q-learners on the iterated game and plot
the average team reward per episode.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# A tiny PettingZoo-style environment: Iterated Coordination Game.
#
# Two agents simultaneously pick one of two "channels" (0 or 1).
#   - If they pick the same channel:  both get +1.
#   - If they pick different channels: both get -1.
#
# The observation each agent receives is the *previous joint action* (or a
# special "start" symbol on the first step).  An episode lasts 25 steps.
#
# We implement the Parallel API (simpler) and provide an AEC adapter on top.
# =============================================================================
class IteratedCoordinationParallel:
    """PettingZoo-style ParallelEnv."""

    metadata = {"name": "iterated_coordination_v0", "is_parallelizable": True}

    def __init__(self, max_steps=25, seed=None):
        self.agents = ["agent_0", "agent_1"]
        self.possible_agents = list(self.agents)
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self._step = 0
        self._last_joint = (2, 2)  # 2 = "start" symbol -> 9 possible obs values

    @property
    def observation_spaces(self):
        return {a: {"n": 9} for a in self.possible_agents}  # 3 x 3 joint actions

    @property
    def action_spaces(self):
        return {a: {"n": 2} for a in self.possible_agents}

    def _obs(self):
        idx = self._last_joint[0] * 3 + self._last_joint[1]
        return {a: idx for a in self.agents}

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._step = 0
        self._last_joint = (2, 2)
        self.agents = list(self.possible_agents)
        obs = self._obs()
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions):
        a0 = actions["agent_0"]
        a1 = actions["agent_1"]
        if a0 == a1:
            r = 1.0
        else:
            r = -1.0
        rewards = {"agent_0": r, "agent_1": r}
        self._last_joint = (a0, a1)
        self._step += 1
        done = self._step >= self.max_steps
        terminations = {a: False for a in self.agents}
        truncations = {a: done for a in self.agents}
        infos = {a: {} for a in self.agents}
        obs = self._obs()
        if done:
            self.agents = []
        return obs, rewards, terminations, truncations, infos


# AEC-style adapter (one-agent-at-a-time loop, following PettingZoo conventions).
class IteratedCoordinationAEC:
    """Tiny AEC wrapper around the parallel env above.

    Mirrors the agent_iter / observe / step / last() pattern.
    """

    def __init__(self, max_steps=25, seed=None):
        self.parallel = IteratedCoordinationParallel(max_steps=max_steps, seed=seed)
        self.possible_agents = list(self.parallel.possible_agents)
        self.agents = list(self.possible_agents)
        self._pending_actions = {}
        self._obs = None
        self._rewards = {a: 0.0 for a in self.possible_agents}
        self._terms = {a: False for a in self.possible_agents}
        self._truncs = {a: False for a in self.possible_agents}

    def reset(self, seed=None):
        self._obs, _ = self.parallel.reset(seed=seed)
        self.agents = list(self.possible_agents)
        self._pending_actions = {}
        self._rewards = {a: 0.0 for a in self.possible_agents}
        self._terms = {a: False for a in self.possible_agents}
        self._truncs = {a: False for a in self.possible_agents}
        self.agent_selection = self.agents[0]

    def agent_iter(self):
        """Yield agents in order until all are done."""
        while self.agents:
            for a in list(self.possible_agents):
                if self._terms[a] or self._truncs[a]:
                    continue
                self.agent_selection = a
                yield a
                if not self.agents:
                    return

    def last(self):
        a = self.agent_selection
        return self._obs[a], self._rewards[a], self._terms[a], self._truncs[a], {}

    def step(self, action):
        a = self.agent_selection
        self._pending_actions[a] = action
        # Once both agents have submitted, advance the parallel env.
        if len(self._pending_actions) == len(self.possible_agents):
            obs, rewards, terms, truncs, _ = self.parallel.step(self._pending_actions)
            self._obs = obs
            self._rewards = rewards
            self._terms = terms
            self._truncs = truncs
            self._pending_actions = {}
            if all(truncs[x] or terms[x] for x in self.possible_agents):
                self.agents = []


# =============================================================================
# Tabular Q-learners that use observations as state.
# =============================================================================
class TabularQ:
    def __init__(self, n_obs, n_actions, alpha=0.1, gamma=0.95, epsilon=0.1, rng=None):
        self.Q = np.zeros((n_obs, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.rng = rng or np.random.default_rng()

    def act(self, obs):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        q = self.Q[obs]
        best = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best))

    def update(self, obs, action, reward, next_obs, done):
        target = reward if done else reward + self.gamma * self.Q[next_obs].max()
        self.Q[obs, action] += self.alpha * (target - self.Q[obs, action])


# =============================================================================
# Training: Parallel API
# =============================================================================
def train_parallel(n_episodes=2000, seed=0):
    rng = np.random.default_rng(seed)
    env = IteratedCoordinationParallel(seed=seed)
    learners = {
        a: TabularQ(n_obs=9, n_actions=2, rng=np.random.default_rng(seed + i))
        for i, a in enumerate(env.possible_agents)
    }

    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_return = 0.0
        done = False
        while not done:
            actions = {a: learners[a].act(obs[a]) for a in env.agents}
            next_obs, rewards, terms, truncs, _ = env.step(actions)
            ep_done = all(terms[a] or truncs[a] for a in env.possible_agents)
            for a in env.possible_agents:
                learners[a].update(
                    obs[a], actions[a], rewards[a], next_obs[a], ep_done
                )
            obs = next_obs
            ep_return += rewards["agent_0"]  # symmetric, either agent works
            done = ep_done
        returns.append(ep_return)

    return np.array(returns), learners


# =============================================================================
# Training: AEC API (same algorithm, different interface)
# =============================================================================
def rollout_aec(n_episodes=50, seed=0):
    """Show that the AEC interface works.  No learning here, just a sanity rollout."""
    rng = np.random.default_rng(seed)
    env = IteratedCoordinationAEC(seed=seed)
    total = 0.0
    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        ep_r = 0.0
        for agent in env.agent_iter():
            obs, reward, term, trunc, _ = env.last()
            ep_r += reward
            if term or trunc:
                env.step(None)
                continue
            action = int(rng.integers(2))
            env.step(action)
        total += ep_r
    return total / n_episodes


# =============================================================================
# Try to use the real PettingZoo if installed.
# =============================================================================
def try_real_pettingzoo():
    try:
        from pettingzoo.classic import rps_v2  # type: ignore
    except Exception as e:
        print("\n[real PettingZoo not available -- skipping]")
        print(f"  reason: {type(e).__name__}: {e}")
        print("  install with:  pip install 'pettingzoo[classic]'")
        return

    print("\n[real PettingZoo found -- rolling out rps_v2]")
    env = rps_v2.env(max_cycles=10)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    rewards = {a: 0.0 for a in env.possible_agents}
    for agent in env.agent_iter():
        obs, reward, term, trunc, _ = env.last()
        rewards[agent] += reward
        if term or trunc:
            env.step(None)
            continue
        action = env.action_space(agent).sample()
        env.step(action)
    print(f"  total rewards after random rollout: {rewards}")


def main():
    print("=== PettingZoo-style multi-agent environments ===")

    print("\n[1/3] Sanity rollout with the AEC API (random policies):")
    avg_r = rollout_aec(n_episodes=50)
    print(f"  average episode return (per agent) with random play: {avg_r:.2f}")
    print("  (expect ~0 -- coordination is unlikely by chance)")

    print("\n[2/3] Training two independent Q-learners (Parallel API):")
    returns, learners = train_parallel(n_episodes=2000, seed=0)
    print(f"  first 100 episodes avg return : {returns[:100].mean():+.2f}")
    print(f"  last  100 episodes avg return : {returns[-100:].mean():+.2f}")
    print(f"  max possible per episode       : {25.0:+.2f}  (25 steps x +1)")
    print(f"  min possible per episode       : {-25.0:+.2f}")

    # Plot smoothed learning curve
    window = 50
    smoothed = np.convolve(returns, np.ones(window) / window, mode="valid")
    plt.figure(figsize=(10, 5))
    plt.plot(returns, color="#bdc3c7", linewidth=0.5, label="Episode return")
    plt.plot(np.arange(window - 1, len(returns)), smoothed,
             color="#2980b9", linewidth=2.2, label=f"Rolling mean ({window})")
    plt.axhline(25, color="#27ae60", linestyle="--", linewidth=1, label="Perfect coordination")
    plt.axhline(0,  color="#7f8c8d", linestyle=":",  linewidth=1, label="Random play")
    plt.xlabel("Episode")
    plt.ylabel("Total per-agent return per episode")
    plt.title("Independent Q-learners coordinating in a PettingZoo-style env")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "pettingzoo_coordination.png")
    plt.savefig(out, dpi=120)
    print(f"  Plot saved to {out}")

    print("\n[3/3] Try the real PettingZoo library:")
    try_real_pettingzoo()


if __name__ == "__main__":
    main()
