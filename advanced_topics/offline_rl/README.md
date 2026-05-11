# Phase 5 · Offline / Batch RL

Three practical works that build offline RL from the ground up, on a
familiar environment (CartPole-v1) and the standard four-quality-level
D4RL recipe:

| # | Implementation | Concept Explainer |
|---|----------------|-------------------|
| 1 | [`d4rl_dataset.py`](d4rl_dataset.py) | [D4RL Benchmark Datasets](d4rl_dataset_explained.md) |
| 2 | [`cql.py`](cql.py) | [Conservative Q-Learning](cql_explained.md) |
| 3 | [`behavioral_cloning.py`](behavioral_cloning.py) | [Behavioral Cloning](behavioral_cloning_explained.md) |

## Why CartPole instead of HalfCheetah?

The real D4RL benchmark targets MuJoCo locomotion tasks
(HalfCheetah, Hopper, Walker2d, Ant). MuJoCo is heavy to install and
isn't in this repo's `requirements.txt`. The *structure* of D4RL — four
quality levels per task: `random`, `medium`, `expert`, `medium-replay` —
transfers cleanly to a smaller environment. Every lesson here (distribution
shift, why naive offline DQN collapses, why BC plateaus at the data
ceiling) is the same one you'd learn on HalfCheetah, with 100× less
training time.

If you have MuJoCo installed and want the real thing later:

```bash
pip install "d4rl"
# in Python:
import d4rl, gymnasium as gym
env = gym.make("halfcheetah-medium-replay-v2")
ds = env.get_dataset()
```

The keys in `ds` are identical to the keys we use in the `.npz` files
(`observations`, `actions`, `rewards`, `next_observations`, `terminals`),
so swapping in real D4RL is a one-line change.

## Quick Start

```bash
# from the repo root
source venv/bin/activate
cd advanced_topics/offline_rl

# 1) Build the four datasets (random / medium / expert / medium-replay)
python d4rl_dataset.py
#   -> writes outputs/cartpole_{random,medium,expert,medium-replay}.npz
#      and a return-distribution histogram.

# 2) Train CQL (and a naive offline-DQN baseline) on the medium-replay set
python cql.py
#   -> writes outputs/cql.png

# 3) Train Behavioral Cloning on all four datasets and compare to CQL
python behavioral_cloning.py
#   -> writes outputs/bc.png
```

All plots and `.npz` datasets land in `outputs/`.

## What You Should Observe

| Script | Expected outcome |
|--------|------------------|
| `d4rl_dataset.py`        | Four `.npz` files of 10k transitions each. Random ~22 reward/episode, medium ~180, expert ~490, medium-replay ~60 (long left tail — it remembers early failures). |
| `cql.py`                 | Naive offline DQN (alpha=0) is unstable — climbs early, often collapses to ~30–150 reward as distribution-shift hallucinations infect the Bellman target. CQL with alpha=1.0 reaches **300–450**. CQL with alpha=5.0 reaches **450–500** consistently. |
| `behavioral_cloning.py`  | BC's evaluation return **tracks the data's average return**: ~20 on random, ~150 on medium, ~480 on expert, ~60 on medium-replay. The clean "BC = imitate the data ceiling" story. |

## The Key Comparison

The whole point of this section is to compare **BC vs CQL on the same
mixed-quality dataset (`medium-replay`)**:

```
BC  on medium-replay   ->  ~60   (matches the average data quality)
CQL on medium-replay   ->  ~400+ (extracts a policy BETTER than the data)
```

CQL **beats the data** because it uses the reward signal and a
conservative Q-function to prefer the good transitions over the bad ones.
BC **matches the data** because it has no concept of "good" or "bad" —
it just copies every action with equal devotion.

This is why offline RL exists as a research field: when logs are mixed
quality (which real logs always are), reward-aware methods recover more
than simple imitation can.

## How They Fit Together

```
       ┌──────────────────────────────┐
       │ d4rl_dataset.py:             │
       │   train DQN, save 4 datasets │  <- builds the offline benchmark
       └──────────────┬───────────────┘
                      │  cartpole_{random,medium,expert,medium-replay}.npz
                      ▼
       ┌─────────────────────────────────────┐
       │ cql.py:                             │
       │   offline Q-learning + conservatism │  <- fix distribution shift
       └──────────────┬──────────────────────┘
                      │
                      ▼
       ┌──────────────────────────────┐
       │ behavioral_cloning.py:       │
       │   supervised cross-entropy   │  <- the trivial baseline that
       └──────────────────────────────┘     reveals when CQL was worth it
```

The shared idea: **the data is fixed; the algorithm is the only thing
that varies.** Once you can navigate distribution shift, you can scale
the same recipe up to IQL, Decision Transformer, AWAC, TD3+BC, and the
rest of modern offline RL.

## Dependencies

Everything works with what is already in the repo's `requirements.txt` —
plus `torch`, which the model-based-RL section already requires:

- `numpy` — arrays and indexing
- `matplotlib` — plots
- `gymnasium` — CartPole-v1
- `torch` — Q-networks and BC classifier

The full pipeline runs comfortably on CPU in a few minutes per script.

## Where to Go Next

- 📄 **CQL paper** (Kumar et al., 2020): [arxiv.org/abs/2006.04779](https://arxiv.org/abs/2006.04779)
- 📄 **Offline RL tutorial** (Levine et al., 2020): [arxiv.org/abs/2005.01643](https://arxiv.org/abs/2005.01643)
- 📄 **IQL** (Kostrikov et al., 2021): never queries Q on un-seen actions
- 📄 **Decision Transformer** (Chen et al., 2021): treats offline RL as
  sequence modelling — same datasets, very different machinery
- 💻 **Real D4RL**: [github.com/Farama-Foundation/D4RL](https://github.com/Farama-Foundation/D4RL)
