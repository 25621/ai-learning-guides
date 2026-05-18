# Apprendimento per Rinforzo: Da Principiante ad Avanzato

Una guida completa per comprendere e costruire sistemi di apprendimento per rinforzo, dai concetti fondamentali alla comprensione a livello di ricerca.

---

## Panoramica

| Fase | Focus | Durata |
|-------|-------|----------|
| 1 | [Fondamenti](#phase-1-foundations-2-4-weeks) | 2-4 settimane |
| 2 | [Metodi Tabulari](#phase-2-tabular-methods-3-4-weeks) | 3-4 settimane |
| 3 | [Approssimazione delle Funzioni](#phase-3-function-approximation-3-4-weeks) | 3-4 settimane |
| 4 | [Gradienti della Politica](#phase-4-policy-gradient-methods-4-5-weeks) | 4-5 settimane |
| 5 | [Argomenti Avanzati](#phase-5-advanced-topics-6-8-weeks) | 6-8 settimane |
| 6 | [Livello di Ricerca](#phase-6-research-level-ongoing) | In corso |

**Tempo totale per la competenza:** ~6 mesi

---

## Fase 1: Fondamenti (2-4 settimane)

### Obiettivo
Comprendere i concetti fondamentali senza matematica approfondita.

### Concetti da Imparare
- Agente, ambiente, stato, azione, ricompensa
- Processi Decisionali di Markov (MDP)
- Ritorno e fattore di sconto (γ)
- Politica vs funzione valore
- Compromesso tra esplorazione e sfruttamento (exploration vs exploitation)

### Lavoro Pratico
- [ ] [Implementare un semplice problema di multi-armed bandit da zero](foundations/multi_armed_bandit_explained.md)
- [ ] [Risolvere Frozen Lake con una politica casuale, osservare i risultati](foundations/frozen_lake_explained.md)
- [ ] [Visualizzare le funzioni stato-valore su una griglia semplice](foundations/state_value_visualization_explained.md)

### Strumenti
- Python
- NumPy
- Matplotlib (per la visualizzazione)

### Risorse
- 📖 [Sutton & Barto](http://incompleteideas.net/book/the-book.html) - Capitoli 1-3
- 🎥 [Lezioni di David Silver](https://www.youtube.com/watch?v=2pWv7GOvuf0) - Lezioni 1-2
- 📝 [OpenAI Spinning Up - Introduzione](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)

### Traguardo
Dovresti essere in grado di spiegare la formulazione del problema RL e perché gli MDP sono il framework standard.

---

## Fase 2: Metodi Tabulari (3-4 settimane)

### Obiettivo
Padroneggiare gli algoritmi classici di RL dove gli spazi di stato/azione sono abbastanza piccoli da essere memorizzati in tabelle.

### Concetti da Imparare
- Programmazione Dinamica
  - Valutazione della Politica (Policy Evaluation)
  - Iterazione della Politica (Policy Iteration)
  - Iterazione del Valore (Value Iteration)
- Metodi Monte Carlo
  - MC first-visit vs every-visit
  - Controllo MC con partenze esplorative (exploring starts)
- Apprendimento a Differenza Temporale (Temporal Difference Learning)
  - Predizione TD(0)
  - SARSA (controllo TD on-policy)
  - Q-learning (controllo TD off-policy)
- Tracce di Idoneità (Eligibility Traces) e TD(λ)

### Lavoro Pratico
- [ ] [Implementare l'iterazione della politica per GridWorld](tabular_methods/policy_iteration_gridworld_explained.md)
- [ ] [Costruire un agente Q-learning per Frozen Lake](tabular_methods/q_learning_frozen_lake_explained.md)
- [ ] [Implementare SARSA per Cliff Walking](tabular_methods/sarsa_cliff_walking_explained.md)
- [ ] [Confrontare il comportamento di SARSA vs Q-learning (percorsi sicuri vs ottimali)](tabular_methods/sarsa_vs_qlearning_explained.md)
- [ ] [Implementare il controllo Monte Carlo per Blackjack](tabular_methods/monte_carlo_blackjack_explained.md)

### Intuizione Chiave
Comprendere i diagrammi di backup: chiariscono come ogni algoritmo aggiorna i valori.

### Strumenti
- NumPy
- Gymnasium (per gli ambienti)

### Risorse
- 📖 Sutton & Barto - Capitoli 4-7
- 🎥 David Silver - Lezioni 3-5
- 💻 [Repository RL di Denny Britz](https://github.com/dennybritz/reinforcement-learning)

### Traguardo
Implementare Q-learning da zero e risolvere Frozen Lake con un tasso di successo >70%.

---

## Fase 3: Approssimazione delle Funzioni (3-4 settimane)

### Obiettivo
Scalare RL oltre i piccoli spazi di stato utilizzando approssimatori di funzione.

### Concetti da Imparare
- Perché i metodi tabulari falliscono su larga scala
- Approssimazione lineare della funzione
- Reti neurali come approssimatori di funzione
- Deep Q-Networks (DQN)
  - Experience replay
  - Target networks
  - Reward clipping
- Miglioramenti di DQN
  - Double DQN
  - Dueling DQN
  - Prioritized Experience Replay
  - Rainbow DQN

### Lavoro Pratico
- [ ] [Risolvere CartPole con Q-learning lineare](function_approximation/linear_q_cartpole_explained.md)
- [ ] [Implementare DQN da zero per CartPole](function_approximation/dqn_cartpole_explained.md)
- [ ] [Aggiungere experience replay, osservare il miglioramento della stabilità](function_approximation/dqn_experience_replay_explained.md)
- [ ] [Aggiungere target network, confrontare le curve di apprendimento](function_approximation/dqn_target_network_explained.md)
- [ ] [Addestrare DQN su Atari Pong (usare ALE-Py)](function_approximation/dqn_atari_pong_explained.md)
- [ ] [Implementare Double DQN, confrontare con DQN vanilla](function_approximation/double_dqn_cartpole_explained.md)

### Intuizione Chiave
La "triade micidiale" (approssimazione della funzione + bootstrapping + off-policy) causa instabilità. Le innovazioni di DQN affrontano questo problema.

### Strumenti
- PyTorch o TensorFlow
- Gymnasium
- ALE-Py (per Atari)
- Weights & Biases (per il tracciamento degli esperimenti)

### Risorse
- 📖 Sutton & Barto - Capitoli 9-11
- 📄 [Articolo DQN (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)
- 📄 [Articolo Rainbow](https://arxiv.org/abs/1710.02298)
- 💻 [Implementazioni CleanRL](https://github.com/vwxyzjn/cleanrl)

### Traguardo
Addestrare un agente DQN che ottenga una ricompensa positiva su Atari Pong.

---

## Fase 4: Gradienti della Politica (4-5 settimane)

### Obiettivo
Imparare a ottimizzare direttamente le politiche senza calcolare le funzioni valore.

### Concetti da Imparare
- Teorema del Gradiente della Politica
- Algoritmo REINFORCE
- Tecniche di riduzione della varianza
  - Baseline
  - Reward-to-go
- Metodi Actor-Critic
  - A2C (Advantage Actor-Critic)
  - A3C (variante asincrona)
- Generalized Advantage Estimation (GAE)
- Metodi Trust Region
  - TRPO (Trust Region Policy Optimization)
  - PPO (Proximal Policy Optimization)

### Lavoro Pratico
- [ ] [Implementare REINFORCE per CartPole](policy_gradients/reinforce_cartpole_explained.md)
- [ ] [Aggiungere baseline, misurare la riduzione della varianza](policy_gradients/reinforce_baseline_explained.md)
- [ ] [Costruire A2C per LunarLander](policy_gradients/a2c_lunarlander_explained.md)
- [ ] [Implementare PPO da zero](policy_gradients/ppo_scratch_explained.md)
- [ ] [Addestrare PPO su controllo continuo (BipedalWalker o MuJoCo)](policy_gradients/ppo_continuous_explained.md)
- [ ] [Confrontare la sensibilità degli iperparametri di PPO](policy_gradients/ppo_hyperparams_explained.md)

### Intuizione Chiave
PPO è il cavallo di battaglia della RL moderna: comprendi profondamente il suo meccanismo di clipping.

### Strumenti
- PyTorch
- Gymnasium
- Stable-Baselines3 (per riferimento)
- MuJoCo o Box2D (per il controllo continuo)

### Risorse
- 📖 Sutton & Barto - Capitolo 13
- 📝 [OpenAI Spinning Up - Policy Gradients](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
- 📄 [Articolo PPO (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- 📄 [Articolo GAE](https://arxiv.org/abs/1506.02438)
- 🎥 [Deep RL Bootcamp di Pieter Abbeel](https://sites.google.com/view/deep-rl-bootcamp/lectures)

### Traguardo
Implementare PPO e risolvere BipedalWalker-v3 (ricompensa > 300).

---

## Fase 5: Argomenti Avanzati (6-8 settimane)

Scegli 2-3 aree in base ai tuoi interessi.
- [RL Basato su Modelli](#model-based-rl)
- [RL Multi-Agente](#multi-agent-rl)
- [RL Offline](#offlinebatch-rl)
- [Esplorazione](#exploration)
- [RL Gerarchico](#hierarchical-rl)
- [RLHF (RL da Feedback Umano)](#rlhf-rl-from-human-feedback)

### RL Basato su Modelli
Imparare le dinamiche dell'ambiente per pianificare o generare esperienza sintetica.

**Concetti:**
- Architettura Dyna
- Modelli del mondo (World models)
- Model Predictive Control (MPC)
- MuZero, Dreamer

**Lavoro Pratico:**
- [ ] [Implementare Dyna-Q](advanced_topics/model_based_rl/dyna_q_explained.md)
- [ ] [Addestrare un modello del mondo su un ambiente semplice](advanced_topics/model_based_rl/world_model_explained.md)
- [ ] [Usare il modello appreso per la pianificazione](advanced_topics/model_based_rl/model_based_planning_explained.md)

**Risorse:**
- 📖 Sutton & Barto - Capitolo 8
- 📄 [Articolo World Models](https://arxiv.org/abs/1803.10122)
- 📄 [Articolo MuZero](https://arxiv.org/abs/1911.08265)
- 📄 [Articolo DreamerV3](https://arxiv.org/abs/2301.04104)

---

### RL Multi-Agente
Più agenti che apprendono simultaneamente in ambienti condivisi.

**Concetti:**
- Apprendimento indipendente (Independent learners)
- Addestramento centralizzato, esecuzione decentralizzata (CTDE)
- Self-play
- Comunicazione emergente

**Lavoro Pratico:**
- [ ] [Addestrare agenti in semplici giochi a matrice](advanced_topics/multi_agent_rl/matrix_games_explained.md)
- [ ] [Implementare self-play per un gioco da tavolo](advanced_topics/multi_agent_rl/self_play_tic_tac_toe_explained.md)
- [ ] [Esplorare gli ambienti PettingZoo](advanced_topics/multi_agent_rl/pettingzoo_explore_explained.md)

**Risorse:**
- 📄 [Sondaggio sulla RL Multi-Agente](https://arxiv.org/abs/1911.10635)
- 📄 [OpenAI Five](https://arxiv.org/abs/1912.06680)
- 💻 [Libreria PettingZoo](https://pettingzoo.farama.org/)

---

### RL Offline
Apprendere da set di dati fissi senza interazione con l'ambiente.

**Concetti:**
- Problema dello spostamento della distribuzione (Distribution shift)
- Conservative Q-Learning (CQL)
- Implicit Q-Learning (IQL)
- Decision Transformer

**Lavoro Pratico:**
- [ ] [Addestrare su dataset benchmark D4RL](advanced_topics/offline_rl/d4rl_dataset_explained.md)
- [ ] [Implementare CQL](advanced_topics/offline_rl/cql_explained.md)
- [ ] [Confrontare con il behavioral cloning](advanced_topics/offline_rl/behavioral_cloning_explained.md)

**Risorse:**
- 📄 [Tutorial sulla RL Offline](https://arxiv.org/abs/2005.01643)
- 📄 [Articolo CQL](https://arxiv.org/abs/2006.04779)
- 📄 [Decision Transformer](https://arxiv.org/abs/2106.01345)
- 💻 [Benchmark D4RL](https://github.com/Farama-Foundation/D4RL)

---

### Esplorazione
Affrontare ricompense sparse e problemi di esplorazione difficili.

**Concetti:**
- Motivazione intrinseca
- Esplorazione guidata dalla curiosità (ICM)
- Random Network Distillation (RND)
- Esplorazione basata sul conteggio (Count-based exploration)
- Go-Explore

**Lavoro Pratico:**
- [ ] [Implementare il bonus di curiosità](advanced_topics/exploration/curiosity_bonus_explained.md)
- [ ] [Addestrare su Montezuma's Revenge](advanced_topics/exploration/montezuma_revenge_explained.md)
- [ ] [Confrontare le strategie di esplorazione](advanced_topics/exploration/compare_exploration_explained.md)

**Risorse:**
- 📄 [Articolo ICM](https://arxiv.org/abs/1705.05363)
- 📄 [Articolo RND](https://arxiv.org/abs/1810.12894)
- 📄 [Go-Explore](https://arxiv.org/abs/1901.10995)
- 📄 [Deep Exploration via Bootstrapped DQN](https://arxiv.org/abs/1602.04621) — fonte del task DeepSea

---

### RL Gerarchico
Apprendere a più livelli di astrazione temporale.

**Concetti:**
- Framework delle opzioni (Options framework)
- Politiche condizionate dall'obiettivo (Goal-conditioned policies)
- Feudal networks
- HIRO, HAM

**Lavoro Pratico:**
- [ ] [Implementare l'architettura option-critic](advanced_topics/hierarchical_rl/option_critic_explained.md)
- [ ] [Addestrare una politica condizionata dall'obiettivo](advanced_topics/hierarchical_rl/goal_conditioned_policy_explained.md)
- [ ] [Testare su task a lungo orizzonte](advanced_topics/hierarchical_rl/long_horizon_tasks_explained.md)

**Risorse:**
- 📖 Sutton & Barto - Capitolo 12 (Options)
- 📄 [Articolo Option-Critic](https://arxiv.org/abs/1609.05140)
- 📄 [Articolo HIRO](https://arxiv.org/abs/1805.08296)

---

### RLHF (RL da Feedback Umano)
Allineare i modelli con le preferenze umane.

**Concetti:**
- Modellazione della ricompensa dai confronti
- Ottimizzazione della politica vincolata da KL
- IA Costituzionale
- DPO (Direct Preference Optimization)

**Lavoro Pratico:**
- [ ] [Addestrare un modello di ricompensa da dati di preferenza](advanced_topics/rlhf/reward_modeling_explained.md)
- [ ] [Fine-tuning di un piccolo LM con PPO](advanced_topics/rlhf/ppo_fine_tuning_explained.md)
- [ ] [Implementare DPO](advanced_topics/rlhf/dpo_implementation_explained.md)

**Risorse:**
- 📄 [Articolo InstructGPT](https://arxiv.org/abs/2203.02155)
- 📄 [IA Costituzionale](https://arxiv.org/abs/2212.08073)
- 📄 [Articolo DPO](https://arxiv.org/abs/2305.18290)
- 💻 [Libreria TRL](https://github.com/huggingface/trl)

---

## Fase 6: Livello di Ricerca (In corso)

### Obiettivo
Contribuire con lavori originali al campo.

### Attività
- Leggere regolarmente articoli da NeurIPS, ICML, ICLR
- Riprodurre i risultati chiave degli articoli recenti
- Identificare problemi aperti e limitazioni
- Eseguire esperimenti rigorosi con una valutazione corretta
- Considerare l'efficienza dei campioni, la generalizzazione, la sicurezza

### Articoli Fondamentali da Studiare
- [ ] [AlphaGo (Silver et al., 2016)](https://www.nature.com/articles/nature16961)
- [ ] [AlphaZero (Silver et al., 2018)](https://www.science.org/doi/10.1126/science.aar6404)
- [ ] [MuZero (Schrittwieser et al., 2020)](https://arxiv.org/abs/1911.08265)
- [ ] [Decision Transformer (Chen et al., 2021)](https://arxiv.org/abs/2106.01345)
- [ ] [InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)
- [ ] [DreamerV3 (Hafner et al., 2023)](https://arxiv.org/abs/2301.04104)

### Dove Pubblicare
- NeurIPS, ICML, ICLR (le sedi principali)
- AAAI, IJCAI
- CoRL (focus sulla robotica)
- Workshop per lavori iniziali

---

## Riepilogo Strumenti

| Fase | Strumenti Raccomandati |
|-------|-------------------|
| 1-2 | NumPy, Matplotlib |
| 3 | PyTorch, Gymnasium, ALE-Py, W&B |
| 4 | Stable-Baselines3, MuJoCo/Box2D |
| 5-6 | Codice personalizzato, Ray/RLlib, JAX (per la velocità) |

---

## Suggerimenti per il Successo

1. **Implementare da zero all'inizio** — Usa le librerie solo dopo aver compreso l'algoritmo
2. **Debuggare con ambienti semplici** — CartPole prima di Atari, sempre
3. **Registrare tutto** — Ricompense, perdite, gradienti, lunghezze degli episodi
4. **Visualizzare l'apprendimento** — Traccia le curve di apprendimento, renderizza gli episodi
5. **Leggere il libro di Sutton & Barto** — È la bibbia della RL
6. **Comprendere la matematica** — Almeno il teorema del gradiente della politica e le equazioni di Bellman
7. **Essere pazienti** — La RL è notoriamente instabile; le esecuzioni fallite sono normali
8. **Usare i seed** — La riproducibilità è importante; fai la media su più seed
9. **Unirsi alle community** — r/reinforcementlearning, RL Discord, Twitter/X

---

## Errori Comuni da Evitare

- ❌ Saltare le fondamenta per passare direttamente alla deep RL
- ❌ Non normalizzare osservazioni/ricompense
- ❌ Usare tassi di apprendimento troppo grandi/piccoli
- ❌ Dimenticare di impostare la modalità di valutazione durante il test
- ❌ Non usare abbastanza seed per gli esperimenti
- ❌ Implementare da articoli senza controllare il codice di riferimento
- ❌ Arrendersi dopo un addestramento fallito

---

## Glossario

| Termine | Definizione |
|------|------------|
| **MDP** | Processo Decisionale di Markov - framework formale per la RL |
| **Politica (π)** | Mappatura dagli stati alle azioni |
| **Funzione Valore (V)** | Ritorno atteso da uno stato |
| **Funzione Q** | Ritorno atteso da una coppia stato-azione |
| **Errore TD** | Differenza tra il valore predetto e quello bootstrap |
| **GAE** | Generalized Advantage Estimation |
| **PPO** | Proximal Policy Optimization |
| **RLHF** | Apprendimento per Rinforzo da Feedback Umano |

---

## Licenza

Questa guida è fornita a scopo educativo. Sentiti libero di condividerla e adattarla.

---
