# Reinforcement Learning: A Comprehensive Technical Guide

> From Markov Decision Processes to Decision Transformers and RLHF.
> A self-contained roadmap for readers who want to move from "I have heard of Q-learning" to "I can read, implement, and critique modern RL papers."

---

## Table of Contents

1. [Foundations](#1-foundations)
2. [Tabular Methods](#2-tabular-methods)
3. [Function Approximation](#3-function-approximation)
4. [Deep Reinforcement Learning — Value-Based](#4-deep-reinforcement-learning--value-based)
5. [Deep Reinforcement Learning — Policy-Based and Actor-Critic](#5-deep-reinforcement-learning--policy-based-and-actor-critic)
6. [Model-Based RL](#6-model-based-rl)
7. [Advanced Topics](#7-advanced-topics)
8. [Offline RL and RL as Sequence Modeling](#8-offline-rl-and-rl-as-sequence-modeling)
9. [RLHF and Modern Applications](#9-rlhf-and-modern-applications)
10. [Building RL Systems in Practice](#10-building-rl-systems-in-practice)
11. [Research Frontiers and Further Reading](#11-research-frontiers-and-further-reading)

---

## 1. Foundations

Reinforcement learning (RL) studies how an **agent** ought to take **actions** in an **environment** so as to maximize a cumulative scalar **reward** signal over time. Unlike supervised learning, the agent does not receive labelled examples; it receives only an evaluation signal and must discover good behavior by trial-and-error interaction.

### 1.1 The Agent–Environment Interaction Loop

At each discrete time step $t = 0, 1, 2, \dots$:

1. The agent observes state $S_t \in \mathcal{S}$.
2. It selects an action $A_t \in \mathcal{A}$ according to its **policy** $\pi(a \mid s) = \Pr[A_t = a \mid S_t = s]$.
3. The environment emits reward $R_{t+1} \in \mathbb{R}$ and next state $S_{t+1}$ sampled from the dynamics $p(s', r \mid s, a)$.

This produces a trajectory:

$$
\tau = (S_0, A_0, R_1, S_1, A_1, R_2, S_2, \dots).
$$

### 1.2 Markov Decision Processes

A finite Markov Decision Process (MDP) is the tuple $\langle \mathcal{S}, \mathcal{A}, p, r, \gamma \rangle$, where

- $\mathcal{S}$ is the state space,
- $\mathcal{A}$ is the action space,
- $p(s', r \mid s, a) = \Pr[S_{t+1} = s', R_{t+1} = r \mid S_t = s, A_t = a]$ is the transition kernel,
- $r(s, a) = \mathbb{E}[R_{t+1} \mid S_t = s, A_t = a]$ is the expected reward,
- $\gamma \in [0, 1]$ is the discount factor.

The **Markov property** says the future depends on the past only through the present state:

$$
\Pr[S_{t+1}, R_{t+1} \mid S_0, A_0, \dots, S_t, A_t] = \Pr[S_{t+1}, R_{t+1} \mid S_t, A_t].
$$

**Intuition.** A state is "Markov" if it summarizes everything from history needed to predict the future. Designing good states is half of an RL problem.

### 1.3 Return and Value Functions

The **return** from time $t$ is the discounted sum of future rewards:

$$
G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}.
$$

For $\gamma < 1$ and bounded rewards, $G_t$ is well-defined and finite. For episodic tasks ($\gamma$ may equal 1), the sum terminates at the end of the episode.

The **state-value function** under policy $\pi$:

$$
v_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s].
$$

The **action-value function**:

$$
q_\pi(s, a) = \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a].
$$

### 1.4 Bellman Equations

Decomposing the return recursively yields the **Bellman expectation equations**:

$$
v_\pi(s) = \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a)\bigl[r + \gamma\, v_\pi(s')\bigr],
$$

$$
q_\pi(s, a) = \sum_{s', r} p(s', r \mid s, a)\Bigl[r + \gamma \sum_{a'} \pi(a' \mid s')\, q_\pi(s', a')\Bigr].
$$

The **optimal value functions** are

$$
v_*(s) = \max_\pi v_\pi(s), \qquad q_*(s, a) = \max_\pi q_\pi(s, a),
$$

and they satisfy the **Bellman optimality equations**:

$$
v_*(s) = \max_a \sum_{s', r} p(s', r \mid s, a)\bigl[r + \gamma\, v_*(s')\bigr],
$$

$$
q_*(s, a) = \sum_{s', r} p(s', r \mid s, a)\Bigl[r + \gamma\, \max_{a'} q_*(s', a')\Bigr].
$$

Once $q_*$ is known, an optimal policy is greedy:

$$
\pi_*(s) \in \arg\max_a q_*(s, a).
$$

The Bellman optimality operator is a $\gamma$-contraction in the sup-norm, which is the engine driving convergence proofs of essentially every value-based method.

### 1.5 Exploration vs. Exploitation

The agent must **exploit** what it knows yields high reward, but also **explore** to discover potentially better actions. The simplest schemes:

- **$\epsilon$-greedy:** with probability $\epsilon$ act uniformly at random, otherwise act greedily.
- **Softmax / Boltzmann:** $\pi(a \mid s) \propto \exp(q(s, a) / \tau)$.
- **Upper Confidence Bound (UCB):** in bandits, $a_t = \arg\max_a \hat{q}(a) + c\sqrt{\ln t / N(a)}$.
- **Thompson sampling:** sample from a posterior over MDPs and act optimally.

Most modern RL agents use entropy regularization, parameter noise (NoisyNet), or intrinsic rewards to encourage exploration in high-dimensional spaces.

### 1.6 Key Takeaways

- An MDP is the standard mathematical scaffolding for RL; the Markov property is what makes value functions well-defined.
- Returns aggregate rewards; value functions are expected returns.
- The Bellman equations decompose value into "reward now + discounted value of next state"; the optimality form is greedy.
- Exploration is not optional — without it, the agent's value estimates are biased toward its current habits.

### Further Reading

- R. S. Sutton & A. G. Barto. *Reinforcement Learning: An Introduction*, 2nd ed. (2018), Chapters 1–3.
- M. L. Puterman. *Markov Decision Processes*, Wiley (1994).
- C. Szepesvári. *Algorithms for Reinforcement Learning* (2010).
- D. Bertsekas & J. Tsitsiklis. *Neuro-Dynamic Programming* (1996).

---

## 2. Tabular Methods

Tabular methods assume $|\mathcal{S}|$ and $|\mathcal{A}|$ are small enough that we can store one value per state (or state–action pair) in a table. They are the conceptual core of RL; every later method is a generalization.

### 2.1 Dynamic Programming

Dynamic programming (DP) assumes the model $p(s', r \mid s, a)$ is known.

**Policy Evaluation** iteratively applies the Bellman expectation operator:

$$
v^{(k+1)}(s) \leftarrow \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a)\bigl[r + \gamma\, v^{(k)}(s')\bigr].
$$

**Policy Improvement** makes the policy greedy w.r.t. the current value function:

$$
\pi'(s) \leftarrow \arg\max_a \sum_{s', r} p(s', r \mid s, a)\bigl[r + \gamma\, v(s')\bigr].
$$

**Policy Iteration** alternates evaluation and improvement and converges in a finite number of iterations for finite MDPs.

**Value Iteration** combines them in one step:

$$
v^{(k+1)}(s) \leftarrow \max_a \sum_{s', r} p(s', r \mid s, a)\bigl[r + \gamma\, v^{(k)}(s')\bigr].
$$

Both converge geometrically with rate $\gamma$.

### 2.2 Monte Carlo Methods

Monte Carlo (MC) methods estimate values from **sample returns** of complete episodes — no model required.

- **First-visit MC** averages returns following the first occurrence of $s$ in each episode.
- **Every-visit MC** averages returns following every occurrence.

For control, MC methods alternate between:

- **Policy evaluation:** estimate $q_\pi(s, a)$ from sampled episodes.
- **Policy improvement:** make $\pi$ greedy w.r.t. $q_\pi$ (usually $\epsilon$-greedy to keep exploring).

**On-policy** MC evaluates and improves the same policy. **Off-policy** MC evaluates a target policy $\pi$ using data from a behavior policy $b$, via importance sampling:

$$
\rho_{t:T-1} = \prod_{k=t}^{T-1} \frac{\pi(A_k \mid S_k)}{b(A_k \mid S_k)}, \qquad V(s) \approx \frac{1}{|\mathcal{T}(s)|} \sum_{t \in \mathcal{T}(s)} \rho_{t:T-1} G_t.
$$

MC is unbiased but high variance and only works for episodic tasks.

### 2.3 Temporal-Difference Learning

TD learning bootstraps: it updates value estimates from estimates rather than waiting for the full return.

**TD(0) prediction:**

$$
V(S_t) \leftarrow V(S_t) + \alpha\bigl[R_{t+1} + \gamma\, V(S_{t+1}) - V(S_t)\bigr].
$$

The bracketed quantity is the **TD error** $\delta_t$.

**SARSA (on-policy TD control):**

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha\bigl[R_{t+1} + \gamma\, Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)\bigr].
$$

**Q-learning (off-policy TD control):**

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha\bigl[R_{t+1} + \gamma\, \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t)\bigr].
$$

Pseudocode:

```text
Q-learning (tabular, ε-greedy)
─────────────────────────────────
Initialize Q(s, a) arbitrarily, Q(terminal, ·) = 0
for each episode:
    s ← env.reset()
    while not done:
        a ← ε-greedy(Q, s)
        s', r, done ← env.step(a)
        Q(s, a) ← Q(s, a) + α [ r + γ max_{a'} Q(s', a') − Q(s, a) ]
        s ← s'
```

**Expected SARSA** replaces the bootstrapped action sample with its expectation under the policy:

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha\Bigl[R_{t+1} + \gamma\, \mathbb{E}_\pi[Q(S_{t+1}, A_{t+1}) \mid S_{t+1}] - Q(S_t, A_t)\Bigr].
$$

This reduces variance vs. SARSA at no algorithmic cost.

### 2.4 n-step TD and TD(λ)

**n-step return:**

$$
G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \dots + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n}).
$$

The **λ-return** mixes n-step returns geometrically:

$$
G_t^\lambda = (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_{t:t+n}.
$$

TD(λ) implements this online via **eligibility traces**:

$$
e_t(s) = \gamma\lambda\, e_{t-1}(s) + \mathbb{1}[S_t = s], \qquad V(s) \leftarrow V(s) + \alpha \delta_t e_t(s).
$$

n-step methods (and TD(λ)) sit on a spectrum: $n=1$ is TD(0), $n=\infty$ (or $\lambda = 1$) is Monte Carlo.

### 2.5 Convergence Guarantees

| Method | Convergence (tabular, finite MDP) |
|---|---|
| Policy / value iteration | Yes — operator is a $\gamma$-contraction. |
| First-/every-visit MC | Yes (w.p. 1) under standard step-size conditions. |
| TD(0) | Yes for prediction with linear or tabular setup. |
| SARSA | Converges if every $(s, a)$ visited infinitely often and $\pi$ becomes greedy in the limit. |
| Q-learning | Converges w.p. 1 with infinite visitation and decreasing step sizes (Watkins & Dayan, 1992). |

### 2.6 When to Use Which

| Setting | Method |
|---|---|
| Known model, small state space | Dynamic programming |
| Unknown model, episodic | Monte Carlo |
| Unknown model, continuing / fast updates | TD(0), SARSA, Q-learning |
| Bias/variance control needed | n-step TD, TD(λ) |
| Off-policy learning required | Q-learning, importance-sampled MC |

### Key Takeaways

- DP uses the model; MC uses full episodes; TD bootstraps from current value estimates.
- Q-learning's `max` operator makes it off-policy and powerful — but also biased (overestimation).
- n-step and λ-returns are the bridge between MC and TD; they tune the bias/variance tradeoff.
- Tabular methods enjoy clean convergence proofs; later methods give them up for scalability.

### Further Reading

- Sutton & Barto (2018), Chapters 4–7, 12.
- C. Watkins & P. Dayan. "Q-learning." *Machine Learning*, 1992.
- J. Tsitsiklis. "Asynchronous stochastic approximation and Q-learning." *Machine Learning*, 1994.
- H. van Seijen et al. "A theoretical and empirical analysis of Expected Sarsa." *ADPRL*, 2009.

---

## 3. Function Approximation

Tabular methods do not scale: a $200 \times 200$ pixel grayscale image has $256^{40{,}000}$ possible "states." We must approximate the value or policy with a parametric function.

### 3.1 From Tables to Parameters

Replace $V(s)$ with $\hat{v}(s; \mathbf{w})$ and $Q(s, a)$ with $\hat{q}(s, a; \mathbf{w})$. The prediction objective becomes a regression:

$$
\overline{\text{VE}}(\mathbf{w}) = \sum_s \mu(s)\bigl[v_\pi(s) - \hat{v}(s; \mathbf{w})\bigr]^2,
$$

where $\mu(s)$ is a state-distribution weighting.

### 3.2 Linear Function Approximation

Use features $\mathbf{x}(s) \in \mathbb{R}^d$ and weights $\mathbf{w}$: $\hat{v}(s; \mathbf{w}) = \mathbf{w}^\top \mathbf{x}(s)$.

**Gradient Monte Carlo:**

$$
\mathbf{w} \leftarrow \mathbf{w} + \alpha\bigl[G_t - \hat{v}(S_t; \mathbf{w})\bigr]\, \mathbf{x}(S_t).
$$

**Semi-gradient TD(0):**

$$
\mathbf{w} \leftarrow \mathbf{w} + \alpha\bigl[R_{t+1} + \gamma\, \hat{v}(S_{t+1}; \mathbf{w}) - \hat{v}(S_t; \mathbf{w})\bigr]\, \mathbf{x}(S_t).
$$

It is called "semi-gradient" because we ignore the dependence of the target on $\mathbf{w}$.

For linear, on-policy TD, the algorithm converges to a fixed point $\mathbf{w}^*$ satisfying $\mathbf{A}\mathbf{w}^* = \mathbf{b}$ where $\mathbf{A} = \mathbb{E}[\mathbf{x}(\mathbf{x} - \gamma \mathbf{x}')^\top]$.

### 3.3 The Deadly Triad

Sutton & Barto identify the **deadly triad** — combining all three of these can cause divergence:

1. **Function approximation** (especially non-linear),
2. **Bootstrapping** (TD-style targets that use current estimates),
3. **Off-policy training** (training distribution $\neq$ target-policy distribution).

Pure MC + linear is safe (no bootstrapping). On-policy TD + linear is safe. Off-policy + bootstrapping + non-linear (e.g. naive deep Q-learning without tricks) can diverge. The DQN innovations of Section 4 are largely about taming this triad.

### 3.4 Toward Deep RL

Replace linear $\hat{v}$ with a neural network $\hat{v}_\theta$. Now $\theta$ is updated by stochastic gradient descent on a chosen loss (TD, policy gradient, etc.). This unlocks Atari, MuJoCo, Go, StarCraft, and ChatGPT — at the cost of substantially worse stability and reproducibility.

| Property | Tabular | Linear FA | Deep FA |
|---|---|---|---|
| Capacity | $O(|\mathcal{S}||\mathcal{A}|)$ | $O(d)$ | Effectively unlimited |
| Generalization | None | Limited | Strong |
| Convergence guarantees | Strong | Many cases | Few |
| Sample efficiency | Low (per state) | Medium | Often poor |

### Key Takeaways

- Approximation is mandatory beyond toy problems; choosing features (or architectures) is now part of the algorithm.
- Linear methods are well-understood; non-linear methods are powerful but unstable.
- The deadly triad is the central reason naive deep TD methods diverge.

### Further Reading

- Sutton & Barto (2018), Chapters 9–11.
- L. Baird. "Residual algorithms: reinforcement learning with function approximation." *ICML*, 1995. (Baird's counterexample.)
- H. van Hasselt et al. "Deep reinforcement learning and the deadly triad." *arXiv:1812.02648*, 2018.
- J. Tsitsiklis & B. Van Roy. "An analysis of temporal-difference learning with function approximation." *IEEE TAC*, 1997.

---

## 4. Deep Reinforcement Learning — Value-Based

### 4.1 DQN: The Deep Q-Network

Mnih et al. (2015) showed that a CNN trained with Q-learning could play 49 Atari games at superhuman level. The key innovations:

- **Experience replay buffer** $\mathcal{D}$ stores $(s, a, r, s')$ tuples; mini-batches are sampled uniformly, breaking temporal correlation.
- **Target network** $\hat{Q}_{\theta^-}$ provides stable bootstrapped targets and is periodically copied from the online net.
- **Reward clipping** to $[-1, 1]$ stabilizes gradients across games.

DQN minimizes

$$
L(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}\Bigl[\bigl(r + \gamma\, \max_{a'} \hat{Q}_{\theta^-}(s', a') - Q_\theta(s, a)\bigr)^2\Bigr].
$$

### 4.2 Double DQN

Vanilla Q-learning overestimates because the same network selects and evaluates actions. Double DQN (van Hasselt et al., 2016) decouples them:

$$
y = r + \gamma\, \hat{Q}_{\theta^-}\!\Bigl(s', \arg\max_{a'} Q_\theta(s', a')\Bigr).
$$

### 4.3 Dueling DQN

Wang et al. (2016) factor $Q$ into a state-value $V$ and an advantage $A$:

$$
Q(s, a) = V(s) + \Bigl(A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a')\Bigr).
$$

This decouples "how good is this state?" from "how good is this action vs. average?" — useful when many actions are similarly valued.

### 4.4 Prioritized Experience Replay (PER)

Schaul et al. (2016): replay transitions with priority proportional to TD error, $p_i \propto |\delta_i|^\alpha$. Correct the introduced bias with importance-sampling weights $w_i = (Np_i)^{-\beta}$.

### 4.5 Rainbow

Hessel et al. (2018) combined six DQN improvements:

1. Double DQN
2. Dueling networks
3. PER
4. Multi-step targets
5. Distributional Q-learning (C51)
6. NoisyNets (parametric exploration)

Rainbow remains a strong baseline on Atari.

### 4.6 Distributional RL: C51 and QR-DQN

Rather than predicting $\mathbb{E}[G_t]$, predict the entire distribution $Z(s, a)$ of returns.

**C51** (Bellemare et al., 2017) represents $Z$ over a fixed support $\{z_0, \dots, z_{N-1}\}$ with $N=51$ atoms and minimizes a projected categorical KL.

**QR-DQN** (Dabney et al., 2018) instead predicts $N$ **quantiles**; the loss is the quantile (Wasserstein-1-friendly) regression loss:

$$
\mathcal{L}_\tau(\delta) = \bigl(\tau - \mathbb{1}[\delta < 0]\bigr)\,\delta.
$$

Distributional methods often improve sample efficiency and stability, even when only the mean is used for action selection.

### 4.7 Comparison

| Algorithm | Year | Key idea | Main benefit |
|---|---|---|---|
| DQN | 2015 | Replay + target net | Stable deep Q-learning |
| Double DQN | 2016 | Decouple selection/eval | Less overestimation |
| Dueling DQN | 2016 | V + A factorization | Better with similar-value actions |
| PER | 2016 | Sample by TD error | Faster learning on rare events |
| C51 | 2017 | Categorical return distribution | Better signal for learning |
| Rainbow | 2018 | Combine all of the above | Strong default for Atari |
| QR-DQN | 2018 | Quantile regression | Wasserstein-friendly distributional RL |

### Key Takeaways

- Replay + target network are the two pillars that make deep Q-learning work.
- Most "improvements" to DQN are independent and can be combined (Rainbow).
- Distributional RL learns more than expected return and consistently helps.

### Further Reading

- V. Mnih et al. "Human-level control through deep reinforcement learning." *Nature*, 2015.
- H. van Hasselt et al. "Deep reinforcement learning with double Q-learning." *AAAI*, 2016.
- M. Hessel et al. "Rainbow: combining improvements in deep RL." *AAAI*, 2018.
- M. Bellemare et al. "A distributional perspective on reinforcement learning." *ICML*, 2017.
- W. Dabney et al. "Distributional reinforcement learning with quantile regression." *AAAI*, 2018.

---

## 5. Deep Reinforcement Learning — Policy-Based and Actor-Critic

Value-based methods learn $Q$ and act greedily; policy-gradient methods learn $\pi_\theta$ directly. The two are unified in actor-critic.

### 5.1 The Policy Gradient Theorem

For a parameterized policy $\pi_\theta(a \mid s)$, the objective is

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_t \gamma^t r_t\right].
$$

The **policy gradient theorem** (Sutton et al., 2000) states:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^{\pi_\theta},\, a \sim \pi_\theta}\bigl[\nabla_\theta \log \pi_\theta(a \mid s)\, q_{\pi_\theta}(s, a)\bigr].
$$

Crucially, the gradient does not depend on the gradient of $d^{\pi_\theta}$ — the state distribution. This is the **log-derivative trick** applied carefully.

### 5.2 REINFORCE

Replace $q_{\pi_\theta}$ with a Monte Carlo return:

$$
\nabla_\theta J \approx \frac{1}{N} \sum_i \sum_t \nabla_\theta \log \pi_\theta(a_t^i \mid s_t^i)\, G_t^i.
$$

Pseudocode:

```text
REINFORCE
─────────────────────────────────
Initialize θ
for each episode:
    Collect trajectory τ = (s_0, a_0, r_1, …, s_T) using π_θ
    Compute returns G_t = Σ_{k≥t} γ^(k−t) r_{k+1}
    For each t:
        θ ← θ + α γ^t G_t ∇_θ log π_θ(a_t | s_t)
```

### 5.3 Baselines and Advantage

Subtract a state-dependent baseline $b(s)$ (often $V(s)$) — it does not bias the gradient but reduces variance:

$$
\nabla_\theta J = \mathbb{E}\bigl[\nabla_\theta \log \pi_\theta(a \mid s)\, (G_t - b(s))\bigr].
$$

The advantage is $A(s, a) = Q(s, a) - V(s)$; estimating it well is the central problem in actor-critic.

### 5.4 A2C / A3C

A2C uses a critic $V_\phi$ trained by TD; the actor uses a one-step advantage $\hat{A}_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$. A3C runs many parallel workers asynchronously (Mnih et al., 2016).

### 5.5 Generalized Advantage Estimation (GAE)

Schulman et al. (2016) interpolate between one-step and Monte Carlo via TD($\lambda$)-like exponential weighting:

$$
\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l\, \delta_{t+l}, \qquad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t).
$$

$\lambda = 0$ recovers TD(0); $\lambda = 1$ recovers Monte Carlo. Empirically $\lambda \in [0.9, 0.99]$.

### 5.6 TRPO and PPO

Plain policy gradient is unstable when step sizes are too large. **TRPO** (Schulman et al., 2015) constrains the KL between successive policies:

$$
\max_\theta \mathbb{E}\!\left[\frac{\pi_\theta(a \mid s)}{\pi_{\theta_{\text{old}}}(a \mid s)}\, \hat{A}\right] \quad \text{s.t.} \quad \mathbb{E}[\text{KL}(\pi_{\theta_{\text{old}}} \,\|\, \pi_\theta)] \le \delta.
$$

**PPO** (Schulman et al., 2017) replaces the constraint with a *clipped* surrogate, removing second-order computation:

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}\Bigl[\min\!\bigl(\rho_t(\theta)\,\hat{A}_t,\ \text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon)\,\hat{A}_t\bigr)\Bigr],
$$

where $\rho_t(\theta) = \pi_\theta(a_t \mid s_t)/\pi_{\theta_{\text{old}}}(a_t \mid s_t)$. The full PPO loss adds a value-function term and entropy bonus:

$$
L^{\text{PPO}} = L^{\text{CLIP}} - c_1 (V_\phi(s) - V^{\text{target}})^2 + c_2 \mathcal{H}[\pi_\theta(\cdot \mid s)].
$$

Pseudocode:

```text
PPO (clipped, single-threaded)
─────────────────────────────────
for iteration = 1, 2, …:
    Run π_{θ_old} for T steps, store (s_t, a_t, r_t, V_φ(s_t), log π_{θ_old}(a_t|s_t))
    Compute advantages Â_t with GAE; targets V^target_t = Â_t + V_φ(s_t)
    Normalize Â
    for K epochs over the buffer in minibatches:
        ρ_t = exp(log π_θ(a_t|s_t) − log π_{θ_old}(a_t|s_t))
        L^CLIP = min(ρ_t Â_t, clip(ρ_t, 1−ε, 1+ε) Â_t)
        L = −E[L^CLIP] + c1 E[(V_φ − V^target)^2] − c2 E[H(π_θ)]
        SGD step on L
    θ_old ← θ
```

### 5.7 Deterministic Policy Gradients: DDPG, TD3

For continuous actions, **DDPG** (Lillicrap et al., 2016) trains a deterministic actor $\mu_\theta(s)$ and a critic $Q_\phi(s, a)$ with the **deterministic policy gradient** (Silver et al., 2014):

$$
\nabla_\theta J = \mathbb{E}\bigl[\nabla_\theta \mu_\theta(s)\, \nabla_a Q_\phi(s, a)\big|_{a = \mu_\theta(s)}\bigr].
$$

**TD3** (Fujimoto et al., 2018) addresses DDPG's overestimation and brittleness:

1. **Twin critics:** take the min of two $Q$ networks for targets.
2. **Delayed policy updates:** update the actor every $d$ critic updates.
3. **Target policy smoothing:** add clipped Gaussian noise to target actions.

### 5.8 Soft Actor-Critic (SAC) and Maximum-Entropy RL

Haarnoja et al. (2018) optimize a maximum-entropy objective:

$$
J(\pi) = \mathbb{E}_\pi\!\left[\sum_t r_t + \alpha\, \mathcal{H}[\pi(\cdot \mid s_t)]\right].
$$

This pushes the agent to be as random as possible while maximizing reward — improving exploration and robustness. The soft Bellman backup is

$$
Q^\pi(s, a) = r + \gamma\, \mathbb{E}_{s'}\bigl[\mathbb{E}_{a' \sim \pi}[Q^\pi(s', a') - \alpha \log \pi(a' \mid s')]\bigr].
$$

The policy is updated by minimizing the expected KL to the soft-Q exponential:

$$
\pi_{\text{new}} = \arg\min_\pi \text{KL}\!\left(\pi(\cdot \mid s) \,\bigg\|\, \frac{\exp(Q^{\pi_{\text{old}}}(s, \cdot)/\alpha)}{Z(s)}\right).
$$

Reparameterize $a = f_\theta(s, \epsilon)$ with $\epsilon \sim \mathcal{N}(0, I)$ and learn $\alpha$ via a target-entropy constraint. Pseudocode:

```text
SAC (continuous actions)
─────────────────────────────────
Initialize policy π_θ, twin critics Q_{φ1}, Q_{φ2}, targets Q̄_{φ1}, Q̄_{φ2}, replay D
for each step:
    a ~ π_θ(·|s);  s', r ← env.step(a);  D ← D ∪ {(s,a,r,s')}
    Sample minibatch B from D
    a' ~ π_θ(·|s')
    y = r + γ ( min_i Q̄_{φi}(s', a') − α log π_θ(a'|s') )
    Update φ_i with ∇(Q_{φi}(s,a) − y)^2 for i = 1, 2
    Update θ with ∇_θ ( α log π_θ(a|s) − min_i Q_{φi}(s, a) ),   a = f_θ(s, ε)
    Update α with ∇_α [ −α (log π_θ(a|s) + H̄) ]
    Polyak-average Q̄_{φi} ← τ Q_{φi} + (1−τ) Q̄_{φi}
```

### 5.9 On-Policy vs. Off-Policy: When to Use What

| Algorithm | On/Off-policy | Action space | Strengths |
|---|---|---|---|
| REINFORCE | On | Discrete/cont. | Simple, unbiased |
| A2C | On | Discrete/cont. | Cheap, scalable |
| PPO | On | Discrete/cont. | Robust, default |
| TRPO | On | Discrete/cont. | Strong theory, slow |
| DDPG | Off | Continuous | Sample-efficient |
| TD3 | Off | Continuous | More stable DDPG |
| SAC | Off | Continuous (mostly) | Exploration via entropy |
| DQN family | Off | Discrete | Sample-efficient on Atari |

### Key Takeaways

- The policy gradient theorem turns "improve $\pi$" into a supervised-learning-style gradient.
- Variance reduction (baselines, advantages, GAE) is what makes policy gradients usable in practice.
- PPO is the empirical workhorse; SAC is the off-policy continuous-control workhorse; TD3 fixes DDPG.
- Maximum-entropy RL gives a principled exploration-regularization story.

### Further Reading

- R. Sutton et al. "Policy gradient methods for RL with function approximation." *NeurIPS*, 2000.
- J. Schulman et al. "High-dimensional continuous control using generalized advantage estimation." *ICLR*, 2016.
- J. Schulman et al. "Proximal policy optimization algorithms." *arXiv:1707.06347*, 2017.
- T. Haarnoja et al. "Soft Actor-Critic: Off-policy maximum entropy deep RL." *ICML*, 2018.
- S. Fujimoto et al. "Addressing function approximation error in actor-critic." *ICML*, 2018.

---

## 6. Model-Based RL

Model-based RL (MBRL) learns (or is given) a transition model $\hat{p}(s' \mid s, a)$ and reward model $\hat{r}(s, a)$, then plans or generates synthetic experience.

### 6.1 Dyna and World Models

**Dyna** (Sutton, 1990) interleaves model learning, planning (Q-updates on imagined transitions), and acting:

```text
Dyna-Q
─────────────────────────────────
Initialize Q, model M
for each step:
    a ← ε-greedy(Q, s); take a, observe r, s'
    Q(s,a) ← Q(s,a) + α[r + γ max_{a'} Q(s', a') − Q(s,a)]
    M(s,a) ← (r, s')                      # store
    repeat n times:                        # planning
        sample (s,a) seen before; (r, s') ← M(s,a)
        Q(s,a) ← Q(s,a) + α[r + γ max_{a'} Q(s', a') − Q(s,a)]
```

**World Models** (Ha & Schmidhuber, 2018) compress observations with a VAE, model dynamics with an RNN, and train a controller in imagination.

### 6.2 MuZero

Schrittwieser et al. (2020) showed that a *latent* model — one trained end-to-end to predict reward, value, and policy from MCTS rollouts, *without* reconstructing observations — matches AlphaZero on board games and beats model-free agents on Atari.

The MuZero learned model has three networks (representation, dynamics, prediction) and is trained jointly to make value/policy/reward predictions consistent with MCTS targets over multi-step unrolls.

### 6.3 Dreamer (v1, v2, v3)

Hafner et al. learn a **recurrent state-space model (RSSM)** with stochastic and deterministic latent state, then train an actor-critic *entirely in imagination*:

- **DreamerV1** (2019): pixel-based latent dynamics, MuJoCo-style continuous control.
- **DreamerV2** (2021): discrete latent variables, Atari-100k state of the art.
- **DreamerV3** (2023): single set of hyperparameters across 150+ tasks; combines symlog reward transforms, KL balancing, and unimix categoricals.

### 6.4 Planning vs. Learning

| | Model-free | Model-based |
|---|---|---|
| Sample efficiency | Lower | Higher |
| Wall-clock to train | Lower | Higher per step (planning) |
| Asymptotic perf. | Often high | Bounded by model bias |
| Robust to model error | Yes (no model) | No |
| Best for | Cheap simulators | Costly real-world data, planning required |

Hybrid approaches (Dyna, MBPO, Dreamer) try to capture the best of both: real-world data trains the model; imagined data trains the policy.

### Key Takeaways

- Models can be planning targets (AlphaZero/MuZero) or imagination engines (Dreamer).
- Latent models beat pixel-reconstruction models in high-dim observation spaces.
- MBRL excels when real data is expensive; suffers when the model has exploitable bugs.

### Further Reading

- R. Sutton. "Integrated architectures for learning, planning, and reacting." *ML Workshop*, 1990.
- D. Ha & J. Schmidhuber. "World models." *NeurIPS*, 2018.
- J. Schrittwieser et al. "Mastering Atari, Go, chess and shogi by planning with a learned model." *Nature*, 2020. (MuZero)
- D. Hafner et al. "Dream to control / Mastering Atari with discrete world models / Mastering diverse domains through world models." (DreamerV1/2/3), 2019–2023.
- M. Janner et al. "When to trust your model: Model-based policy optimization." *NeurIPS*, 2019. (MBPO)

---

## 7. Advanced Topics

### 7.1 Exploration

Beyond $\epsilon$-greedy, modern exploration techniques add **intrinsic rewards** $r_t^{\text{int}}$:

- **Count-based:** $r^{\text{int}} \propto 1/\sqrt{N(s)}$. Pseudo-counts (Bellemare et al., 2016) generalize counts via density models.
- **Curiosity / ICM** (Pathak et al., 2017): intrinsic reward = error of a learned forward model.
- **Random Network Distillation (RND)** (Burda et al., 2019): predict a random target network's output; novelty = prediction error.
- **NoisyNets** (Fortunato et al., 2018): inject learned parameter noise into the policy.
- **Go-Explore** (Ecoffet et al., 2021): archive promising states, return to them, then explore.

These were critical for hard-exploration Atari games like Montezuma's Revenge.

### 7.2 Hierarchical RL (HRL)

HRL operates at multiple time scales.

- **Options framework** (Sutton, Precup, Singh, 1999): an option is $\langle I, \pi, \beta \rangle$ — initiation set, intra-option policy, termination function. Acting on options yields a *semi-MDP*.
- **Option-Critic** (Bacon et al., 2017): learn options end-to-end with policy gradients.
- **FeUdal Networks** (Vezhnevets et al., 2017): a high-level manager outputs latent goals; a worker policy is rewarded for moving the latent state in the goal direction.
- **HIRO** (Nachum et al., 2018): off-policy HRL with goal relabeling.

### 7.3 Multi-Agent RL (MARL)

- **Independent learners**: each agent treats others as part of the environment (non-stationary).
- **Centralized training, decentralized execution (CTDE)**: e.g. **MADDPG** (Lowe et al., 2017) — each agent's critic sees all observations and actions during training, the actor only sees its own.
- **Self-play**: AlphaGo, AlphaZero, OpenAI Five, AlphaStar. Critical question: how to prevent cycling / non-transitive strategies — solutions include fictitious self-play, league training, PSRO.
- **Cooperative MARL**: QMIX, VDN factor a joint $Q$ as a monotonic mixture of per-agent $Q$s.

### 7.4 Imitation and Inverse RL

When rewards are sparse or unspecified, imitate experts.

- **Behavioral cloning (BC)**: supervised learning from $(s, a)$ pairs. Suffers from *covariate shift*: small errors compound.
- **DAgger** (Ross et al., 2011): iteratively query the expert on states *visited by the current learner*. Reduces compounding error.
- **Inverse RL (IRL)** (Ng & Russell, 2000): recover a reward function under which the expert is optimal.
- **GAIL** (Ho & Ermon, 2016): cast IRL as a GAN — a discriminator distinguishes expert from learner, the policy is trained to fool it.
- **Apprenticeship learning, MaxEnt IRL, AIRL**: variants with different reward/discriminator parameterizations.

### 7.5 Meta-RL

The agent learns to learn: across a distribution of tasks, it acquires inductive biases that make adaptation to new tasks fast.

- **Gradient-based**: **MAML** (Finn et al., 2017) — initialize parameters such that one (or few) SGD steps on a new task yield good performance.
- **Recurrent**: **RL$^2$** (Duan et al., 2016) — a recurrent agent unrolled across a task is effectively a learning algorithm encoded in its hidden state.
- **Context-based**: **PEARL** (Rakelly et al., 2019) — encode a task into a latent variable $z$ and condition the policy on it.

### Key Takeaways

- Exploration is task-specific; sparse-reward games are an unsolved frontier.
- Hierarchy reduces effective horizon and improves transfer.
- Multi-agent and self-play raise stability problems that single-agent RL never sees.
- Imitation and IRL are how you get RL off the ground when designing a reward is harder than acting.

### Further Reading

- D. Pathak et al. "Curiosity-driven exploration by self-supervised prediction." *ICML*, 2017.
- Y. Burda et al. "Exploration by random network distillation." *ICLR*, 2019.
- P.-L. Bacon, J. Harb, D. Precup. "The Option-Critic Architecture." *AAAI*, 2017.
- R. Lowe et al. "Multi-agent actor-critic for mixed cooperative-competitive environments." *NeurIPS*, 2017.
- J. Ho & S. Ermon. "Generative adversarial imitation learning." *NeurIPS*, 2016.
- C. Finn et al. "Model-agnostic meta-learning." *ICML*, 2017.

---

## 8. Offline RL and RL as Sequence Modeling

**Offline RL** (a.k.a. batch RL) learns a policy from a fixed dataset $\mathcal{D} = \{(s, a, r, s')\}$ with no further interaction. The killer problem is **distribution shift**: the learned policy proposes out-of-distribution (OOD) actions whose $Q$-values are extrapolated and typically overestimated.

### 8.1 Conservative / Constrained Methods

**BCQ** (Fujimoto et al., 2019): learn a generative model of the behavior policy and restrict actions to those it would have taken.

**CQL** (Kumar et al., 2020): augment Bellman backups with a conservative penalty that *lowers* $Q$ on OOD actions:

$$
\min_Q\ \alpha\Bigl(\mathbb{E}_{s \sim \mathcal{D}, a \sim \mu}[Q(s,a)] - \mathbb{E}_{(s,a) \sim \mathcal{D}}[Q(s,a)]\Bigr) + \mathcal{L}_{\text{TD}}(Q).
$$

The first term pushes $Q$ down on the proposal distribution $\mu$ (often the learned policy) and up on the data — a lower bound on the true $Q^\pi$.

**IQL** (Kostrikov et al., 2022): never evaluate $Q$ on OOD actions at all. Train $V$ to estimate an expectile of $Q$ on data:

$$
L_V = \mathbb{E}_{(s,a) \sim \mathcal{D}}\bigl[L_2^\tau(Q(s,a) - V(s))\bigr], \quad L_2^\tau(u) = |\tau - \mathbb{1}(u<0)| u^2.
$$

Then update $Q$ with $r + \gamma V(s')$ (no max), and extract a policy via advantage-weighted regression:

$$
\pi(a \mid s) \propto \beta_\pi(a \mid s)\, \exp\!\bigl(\eta\,(Q(s,a) - V(s))\bigr).
$$

| Method | Key trick | Avoid OOD actions by |
|---|---|---|
| BCQ | Generative model of $\pi_\beta$ | Restricting action proposals |
| CQL | Conservative penalty | Pushing OOD $Q$ values down |
| IQL | Expectile $V$ + advantage-weighted regression | Never querying $Q$ on OOD actions |

### 8.2 RL as Sequence Modeling

A radically different idea: forget Bellman backups; treat RL as **supervised sequence modeling**.

**Decision Transformer (DT)** (Chen et al., 2021): tokenize trajectories as

$$
\bigl(\hat{R}_1, s_1, a_1, \hat{R}_2, s_2, a_2, \dots\bigr),
$$

where $\hat{R}_t = \sum_{t' \ge t} r_{t'}$ is the **return-to-go**. Train a causal Transformer with cross-entropy / MSE on actions. At inference, prompt with a *target* return and let the model emit actions autoregressively.

**Trajectory Transformer (TT)** (Janner et al., 2021): discretize $(s, a, r)$ tokens and train a sequence model + planner (beam search over generated trajectories).

Why this matters:

- No bootstrapping → no deadly triad.
- No max over OOD actions → no extrapolation error.
- Conditioning lets you specify desired returns or goals.
- The same architecture scales with data and compute, like LMs.

Limits: DT cannot improve on the dataset's stitched-together trajectories — it lacks the dynamic-programming-style "best of suffixes" that Q-learning enjoys. Variants (e.g. Q-Transformer, ODT) add value-style targets.

### Key Takeaways

- Distribution shift is the central obstacle in offline RL.
- The spectrum runs from "constrain to data" (BCQ) through "penalize OOD" (CQL) to "never query OOD" (IQL).
- Decision Transformer recasts RL as supervised sequence modeling, sidestepping bootstrapping entirely.

### Further Reading

- S. Levine et al. "Offline reinforcement learning: tutorial, review, and perspectives on open problems." *arXiv:2005.01643*, 2020.
- A. Kumar et al. "Conservative Q-Learning for offline RL." *NeurIPS*, 2020.
- I. Kostrikov et al. "Offline reinforcement learning with implicit Q-learning." *ICLR*, 2022.
- L. Chen et al. "Decision Transformer: reinforcement learning via sequence modeling." *NeurIPS*, 2021.
- M. Janner et al. "Offline RL as one big sequence modeling problem." *NeurIPS*, 2021.

---

## 9. RLHF and Modern Applications

Reinforcement Learning from Human Feedback (RLHF) is how modern large language models (LLMs) are aligned with human intent — and is arguably the most economically important application of RL today.

### 9.1 The Three-Stage RLHF Recipe

Used in InstructGPT (Ouyang et al., 2022) and most production LLMs:

1. **Supervised fine-tuning (SFT)**: train base model $\pi^{\text{SFT}}$ on curated $(x, y)$ pairs.
2. **Reward modeling**: collect human preferences $y^+ \succ y^-$ on pairs of completions for prompt $x$; train $r_\phi$ with the **Bradley–Terry** loss

$$
\mathcal{L}_R = -\mathbb{E}_{(x, y^+, y^-) \sim \mathcal{D}}\bigl[\log \sigma(r_\phi(x, y^+) - r_\phi(x, y^-))\bigr].
$$

3. **RL fine-tuning**: optimize $\pi_\theta$ to maximize $r_\phi$, with a KL anchor to the SFT model:

$$
\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)}\bigl[r_\phi(x, y) - \beta \log\tfrac{\pi_\theta(y|x)}{\pi^{\text{SFT}}(y|x)}\bigr].
$$

PPO is the standard optimizer. The "four-model" inference graph during training includes:

| Model | Role | Updated? |
|---|---|---|
| Actor $\pi_\theta$ | Policy being trained | Yes |
| Critic $V_\phi$ | Value baseline (often LM head) | Yes |
| Reference $\pi^{\text{SFT}}$ | KL anchor | No (frozen) |
| Reward model $r_\phi$ | Scalar reward | No (frozen) |

Memory and engineering are dominated by these four forward passes per token.

### 9.2 DPO: Direct Preference Optimization

Rafailov et al. (2023) note that the KL-regularized RL objective has a closed-form optimal policy:

$$
\pi^*(y \mid x) = \frac{1}{Z(x)} \pi^{\text{SFT}}(y \mid x) \exp(r_\phi(x, y)/\beta).
$$

Solving for $r$ and plugging into the Bradley–Terry loss eliminates the reward model entirely. The DPO loss is:

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y^+, y^-)}\!\left[\log \sigma\!\left(\beta \log \tfrac{\pi_\theta(y^+ \mid x)}{\pi^{\text{SFT}}(y^+ \mid x)} - \beta \log \tfrac{\pi_\theta(y^- \mid x)}{\pi^{\text{SFT}}(y^- \mid x)}\right)\right].
$$

Two model passes (actor, reference) instead of four; no RL loop; no reward model. Variants:

- **IPO** (Azar et al., 2023): identity preference objective; robust to over-confident preference labels.
- **KTO** (Ethayarajh et al., 2024): only needs scalar "good/bad" labels, not pairs; built on prospect theory.
- **ORPO** (Hong et al., 2024): combines SFT and odds-ratio preference loss into a single stage.

### 9.3 RLAIF and Constitutional AI

**RLAIF** (Bai et al., 2022) replaces human preference labels with model-generated ones, guided by principles (a "constitution"). Drastically cheaper at scale, with quality often comparable to RLHF.

### 9.4 RL for Reasoning

For mathematical, coding, and chain-of-thought reasoning, **outcome-based** rewards (final answer correct/incorrect) work surprisingly well at scale. Active areas:

- **PPO-style** with rule-based rewards (test passes, math grader).
- **GRPO** (Group Relative Policy Optimization, used in DeepSeek-R1): sample $G$ completions per prompt, normalize advantages within the group, drop the value network entirely — fewer parameters and forward passes than PPO at LM scale.
- **Process Reward Models (PRMs)**: score intermediate reasoning steps rather than only the final answer. Solves credit assignment over long chains but is expensive to label.

### Comparison: PPO-RLHF vs. DPO Family

| | PPO-RLHF | DPO / IPO / KTO / ORPO |
|---|---|---|
| Reward model | Required | None |
| Inference passes / step | 4 | 2 |
| Online sampling | Yes | No (offline pairs) |
| Hyperparam sensitivity | High ($\beta$, clip, KL coeff, …) | Lower |
| Asymptotic quality | Often higher with enough effort | Usually competitive |
| Data requirements | Pairs + on-policy rollouts | Pairs only (KTO: scalars) |

### Key Takeaways

- RLHF is the alignment workhorse; understand the actor/critic/reference/reward layout to debug it.
- DPO and friends reframe RLHF as a contrastive supervised loss, avoiding the RL loop entirely.
- Reasoning RL (e.g. GRPO, PRMs) is a fast-moving frontier where rewards are verifiable and rule-based.

### Further Reading

- L. Ouyang et al. "Training language models to follow instructions with human feedback." *NeurIPS*, 2022. (InstructGPT)
- R. Rafailov et al. "Direct preference optimization: your language model is secretly a reward model." *NeurIPS*, 2023.
- M. G. Azar et al. "A general theoretical paradigm to understand learning from human preferences." *AISTATS*, 2024. (IPO)
- K. Ethayarajh et al. "KTO: Model alignment as prospect theoretic optimization." 2024.
- Y. Bai et al. "Constitutional AI: harmlessness from AI feedback." *arXiv:2212.08073*, 2022.
- Z. Shao et al. "DeepSeekMath: pushing the limits of mathematical reasoning in open language models." 2024. (GRPO)

---

## 10. Building RL Systems in Practice

The gap between papers and working code is wider in RL than in any other ML subfield. Implementation details often matter more than algorithm choice.

### 10.1 Implementation Pitfalls

- **Reward scaling.** Wildly different reward magnitudes blow up value functions. Normalize, clip to $[-1, 1]$, or use return normalization (PPO).
- **Observation normalization.** Maintain a running mean/var, normalize, clip to e.g. $[-10, 10]$. Especially important for MuJoCo.
- **Advantage normalization.** Per-minibatch standardization of advantages stabilizes policy gradients.
- **Action clipping vs. squashing.** Gaussian policies on bounded actions need either clipping (cheap, biased) or a `tanh` squashing with the appropriate log-prob correction (SAC).
- **Value clipping.** Some PPO implementations clip the value function update analogously to the policy.
- **Learning-rate annealing.** Linear decay to zero over training is a strong default for PPO.
- **GAE $\lambda$.** $\lambda \in [0.9, 0.97]$ usually beats $\lambda = 1$ or $\lambda = 0$.
- **Network init.** Orthogonal init with gain $\sqrt{2}$ in hidden layers and gain 0.01 in the policy output is a known PPO trick.
- **Mini-batching and epochs.** PPO with 32 minibatches × 10 epochs is a common default; too many epochs causes the importance ratio to blow up.

### 10.2 Reproducibility

RL results are notoriously seed-sensitive. Henderson et al. (2018) showed that 5-seed mean curves can flip ordering between methods. Best practices:

- Report at least **5–10 seeds**; show mean, std, and individual curves.
- Use **stratified evaluation** (e.g. 100 evaluation episodes per checkpoint), not training rewards.
- Fix seeds for env, NumPy, PyTorch/TF, and CUDA backends.
- Log everything to Weights & Biases / TensorBoard.

### 10.3 Hyperparameter Sensitivity

Algorithms differ enormously in robustness.

| Algorithm | Hyperparam sensitivity |
|---|---|
| PPO | Moderate, well-known defaults exist |
| SAC | Lower — auto-tuned $\alpha$, replay-based |
| DDPG | High — without TD3 tricks, brittle |
| DQN | Moderate — replay size and target update interval matter |
| RLHF-PPO | High — KL coefficient, reward-model calibration, length penalties |

### 10.4 Evaluation Protocols

- **Deterministic eval policy** (e.g. $\epsilon = 0.01$ or mean of Gaussian) separates exploration from performance.
- **Pre-defined evaluation horizons** match the training MDP.
- **Atari-100k** and **DeepMind Control 1M** are sample-efficient evaluation regimes increasingly preferred over "billions of frames."
- **CleanRL** and **Stable-Baselines3** publish reference learning curves with hyperparameters — use them as ground truth before claiming a regression.

### 10.5 Libraries

| Library | Strengths | Weaknesses |
|---|---|---|
| **CleanRL** | Single-file, faithful to papers, readable | Not a framework — copy and adapt |
| **Stable-Baselines3** | Production-quality, well-tested | Some custom envs require shims |
| **RLlib** (Ray) | Scales to clusters, many algorithms | Heavyweight, opinionated |
| **Tianshou** | Modular PyTorch, fast | Smaller community |
| **Acme / Rlax** | Google DeepMind, JAX-based | Steep learning curve |
| **TorchRL** | Modern, PyTorch-native | Maturing API |
| **TRL / TRLX / OpenRLHF** | RLHF + LM-specific | Tied to HF ecosystem |

### 10.6 Benchmark Environments

| Domain | Suite |
|---|---|
| Classical control | Gymnasium (CartPole, LunarLander, MountainCar) |
| Pixel-based discrete | Atari (Arcade Learning Environment) |
| Continuous control | MuJoCo, DeepMind Control Suite, Brax (JAX) |
| Hard exploration | Montezuma, Pitfall, NetHack |
| Generalization | ProcGen, Crafter, MiniGrid |
| 3D, partial obs. | Habitat, DM Lab, MineRL |
| Robotics | RoboSuite, Isaac Gym, Meta-World |
| Multi-agent | PettingZoo, SMAC, Melting Pot |
| Offline RL | D4RL (MuJoCo, Adroit, AntMaze, Kitchen), RL Unplugged |
| LM-RL | TRLX prompts, math/code datasets (GSM8K, MATH, HumanEval) |

### Key Takeaways

- "PPO" is really *PPO + 10 hidden tricks* — read the implementation, not just the paper.
- Always re-run baselines from CleanRL / SB3 before believing a new method beats them.
- Seeds, evaluation protocol, and reward/observation normalization decide most outcomes.

### Further Reading

- L. Engstrom et al. "Implementation matters in deep RL: A case study on PPO and TRPO." *ICLR*, 2020.
- P. Henderson et al. "Deep reinforcement learning that matters." *AAAI*, 2018.
- A. Raffin et al. "Stable-Baselines3: reliable reinforcement learning implementations." *JMLR*, 2021.
- S. Huang et al. "CleanRL: high-quality single-file implementations of deep RL." *JMLR*, 2022.
- M. Andrychowicz et al. "What matters in on-policy reinforcement learning? A large-scale empirical study." *ICLR*, 2021.

---

## 11. Research Frontiers and Further Reading

### 11.1 Open Problems

- **Sample efficiency.** Atari-100k, DMC-1M, and real-world robotics push toward learning in *hours of experience*, not years. World models (Dreamer) and unsupervised pretraining are leading candidates.
- **Generalization.** Agents overfit to training MDPs; ProcGen and Crafter expose this. Procedurally generated training, data augmentation, and large-scale multitask training help but do not solve it.
- **Long-horizon credit assignment.** Vanilla bootstrapping decays $\gamma$-fast; PRMs, hierarchical RL, return-conditioning, and successor features are partial answers.
- **Safe RL.** Constrained MDPs, Lagrangian methods, shielded RL, and reachability analysis matter for real deployment.
- **Reward hacking and specification.** Inverse RL, RLHF, and constitutional methods address the symptom; the disease — that scalar rewards are a leaky abstraction — is unsolved.
- **RL + LLMs.** Process rewards, agentic loops, tool use, and verifiable rewards (math, code) are the highest-leverage application domain right now.

### 11.2 Foundational Papers

A starter reading list — read in roughly this order:

1. R. Sutton. "Learning to predict by the methods of temporal differences." *Machine Learning*, 1988.
2. C. Watkins & P. Dayan. "Q-learning." *Machine Learning*, 1992.
3. R. Sutton, D. McAllester, S. Singh, Y. Mansour. "Policy gradient methods for RL with function approximation." *NeurIPS*, 2000.
4. V. Mnih et al. "Human-level control through deep reinforcement learning." *Nature*, 2015.
5. J. Schulman et al. "Trust region policy optimization." *ICML*, 2015.
6. T. Lillicrap et al. "Continuous control with deep reinforcement learning." *ICLR*, 2016. (DDPG)
7. J. Schulman et al. "High-dimensional continuous control using generalized advantage estimation." *ICLR*, 2016.
8. D. Silver et al. "Mastering the game of Go with deep neural networks and tree search." *Nature*, 2016. (AlphaGo)
9. M. Bellemare, W. Dabney, R. Munos. "A distributional perspective on RL." *ICML*, 2017.
10. J. Schulman et al. "Proximal policy optimization algorithms." *arXiv*, 2017.
11. T. Haarnoja et al. "Soft Actor-Critic." *ICML*, 2018.
12. D. Hafner et al. "Dream to control: learning behaviors by latent imagination." *ICLR*, 2020.
13. J. Schrittwieser et al. "Mastering Atari, Go, chess and shogi by planning with a learned model." *Nature*, 2020. (MuZero)
14. L. Chen et al. "Decision Transformer: RL via sequence modeling." *NeurIPS*, 2021.
15. L. Ouyang et al. "Training language models to follow instructions with human feedback." *NeurIPS*, 2022.
16. R. Rafailov et al. "Direct preference optimization." *NeurIPS*, 2023.
17. D. Hafner et al. "Mastering diverse domains through world models." *Nature*, 2025. (DreamerV3)

### 11.3 Textbooks

- R. Sutton & A. Barto. *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018. — the canonical text. Read it cover-to-cover at least once.
- C. Szepesvári. *Algorithms for Reinforcement Learning*, Morgan & Claypool, 2010. — concise, theoretical.
- D. Bertsekas. *Reinforcement Learning and Optimal Control*, Athena Scientific, 2019. — control-theoretic perspective.
- M. Puterman. *Markov Decision Processes*. — the MDP bible.
- A. Geramifard et al. *A Tutorial on Linear Function Approximators for Dynamic Programming and RL*. *Foundations and Trends in ML*, 2013.

### 11.4 Courses

- **David Silver's UCL course** (2015). The classic lecture series — companion to Sutton & Barto.
- **Berkeley CS285** (Sergey Levine). Deep RL, updated each year. Lectures and homework public.
- **Stanford CS234** (Emma Brunskill). Strong on theory and exploration.
- **DeepMind × UCL Deep RL lectures** (Hadsell, van Hasselt, et al.).
- **OpenAI Spinning Up**. Practical introduction with reference implementations.
- **Hugging Face Deep RL course**. Hands-on, modern (PPO, SAC), uses Gymnasium and SB3.

### 11.5 Where to Track New Work

- Conferences: **NeurIPS, ICML, ICLR, AAAI, AISTATS, UAI, CoRL, RSS, IROS, ICRA**.
- Workshops: NeurIPS Deep RL Workshop, ICLR Reincarnating RL, NeurIPS Goal-conditioned RL.
- arXiv categories: `cs.LG`, `cs.AI`, `stat.ML`.
- Blogs / aggregators: Lil'Log (Lilian Weng), The Gradient, BAIR Blog, DeepMind Blog, OpenAI Blog, Yannic Kilcher (video).
- Code: Papers With Code, CleanRL, Hugging Face TRL, EleutherAI / OpenRLHF.

### Key Takeaways

- Sample efficiency, generalization, credit assignment, and reward specification are the four unsolved structural problems of the field.
- The Sutton & Barto book + David Silver lectures + CS285 + Spinning Up will get a reader to research literacy.
- The most consequential applied RL frontier today is RL on LLMs (RLHF, RLAIF, reasoning-RL).

### Further Reading

- K. Arulkumaran et al. "A brief survey of deep reinforcement learning." *IEEE Signal Processing Magazine*, 2017.
- Y. Li. "Deep reinforcement learning: an overview." *arXiv:1701.07274*.
- S. Levine. "Reinforcement learning and control as probabilistic inference." *arXiv:1805.00909*, 2018.
- T. Yu et al. "Reinforcement learning with prior data: a survey." 2023.
- A. Plaat. *Deep Reinforcement Learning*, Springer, 2022.

---

*End of guide.*
