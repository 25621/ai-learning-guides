# Reinforcement Learning: Vom Anfänger zum Profi

Ein umfassender Leitfaden zum Verständnis und zum Aufbau von Reinforcement-Learning-Systemen, von den grundlegenden Konzepten bis hin zum Verständnis auf Forschungsniveau.

---

## Übersicht

| Phase | Fokus | Dauer |
|-------|-------|----------|
| 1 | [Grundlagen](#phase-1-grundlagen-2-4-wochen) | 2-4 Wochen |
| 2 | [Tabellarische Methoden](#phase-2-tabellarische-methoden-3-4-wochen) | 3-4 Wochen |
| 3 | [Funktionsapproximation](#phase-3-funktionsapproximation-3-4-wochen) | 3-4 Wochen |
| 4 | [Policy-Gradient-Methoden](#phase-4-policy-gradient-methoden-4-5-wochen) | 4-5 Wochen |
| 5 | [Fortgeschrittene Themen](#phase-5-fortgeschrittene-themen-6-8-wochen) | 6-8 Wochen |
| 6 | [Forschungsniveau](#phase-6-forschungsniveau-fortlaufend) | Laufend |

**Gesamtzeit bis zur Kompetenz:** ca. 6 Monate

---

## Phase 1: Grundlagen (2-4 Wochen)

### Ziel
Kernkonzepte ohne tiefe Mathematik verstehen.

### Zu lernende Konzepte
- Agent, Umgebung (Environment), Zustand (State), Aktion, Belohnung (Reward)
- Markov-Entscheidungsprozesse (MDP)
- Return und Diskontierungsfaktor (γ)
- Policy vs. Value-Funktion
- Abwägung zwischen Exploration und Exploitation

### Praktische Arbeit
- [ ] [Implementierung eines einfachen Multi-Armed Bandit Problems von Grund auf](foundations/multi_armed_bandit_explained.md)
- [ ] [Lösen von Frozen Lake mit einer Zufallsstrategie, Beobachtung der Ergebnisse](foundations/frozen_lake_explained.md)
- [ ] [Visualisierung von Zustands-Value-Funktionen auf einem einfachen Gitter](foundations/state_value_visualization_explained.md)

### Werkzeuge
- Python
- NumPy
- Matplotlib (zur Visualisierung)

### Ressourcen
- 📖 [Sutton & Barto](http://incompleteideas.net/book/the-book.html) - Kapitel 1-3
- 🎥 [David Silver Vorlesungen](https://www.youtube.com/watch?v=2pWv7GOvuf0) - Vorlesung 1-2
- 📝 [OpenAI Spinning Up - Einführung](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)

### Meilenstein
Sie sollten in der Lage sein, die RL-Problemstellung zu erklären und zu begründen, warum MDPs das Standard-Framework sind.

---

## Phase 2: Tabellarische Methoden (3-4 Wochen)

### Ziel
Beherrschung klassischer RL-Algorithmen, bei denen Zustands-/Aktionsräume klein genug sind, um in Tabellen gespeichert zu werden.

### Zu lernende Konzepte
- Dynamische Programmierung
  - Policy Evaluation
  - Policy Iteration
  - Value Iteration
- Monte-Carlo-Methoden
  - First-visit vs. Every-visit MC
  - MC Control mit explorierenden Starts
- Temporal Difference Learning
  - TD(0) Prädiktion
  - SARSA (On-policy TD Control)
  - Q-Learning (Off-policy TD Control)
- Eligibility Traces und TD(λ)

### Praktische Arbeit
- [ ] [Implementierung der Policy Iteration für GridWorld](tabular_methods/policy_iteration_gridworld_explained.md)
- [ ] [Aufbau eines Q-Learning Agenten für Frozen Lake](tabular_methods/q_learning_frozen_lake_explained.md)
- [ ] [Implementierung von SARSA für Cliff Walking](tabular_methods/sarsa_cliff_walking_explained.md)
- [ ] [Vergleich des Verhaltens von SARSA vs. Q-Learning (sichere vs. optimale Pfade)](tabular_methods/sarsa_vs_qlearning_explained.md)
- [ ] [Implementierung von Monte-Carlo-Control für Blackjack](tabular_methods/monte_carlo_blackjack_explained.md)

### Kern-Erkenntnis
Verstehen Sie die Backup-Diagramme – sie verdeutlichen, wie jeder Algorithmus Werte aktualisiert.

### Werkzeuge
- NumPy
- Gymnasium (für Umgebungen)

### Ressourcen
- 📖 Sutton & Barto - Kapitel 4-7
- 🎥 David Silver Vorlesungen 3-5
- 💻 [Denny Britz' RL Repository](https://github.com/dennybritz/reinforcement-learning)

### Meilenstein
Implementieren Sie Q-Learning von Grund auf und lösen Sie Frozen Lake mit einer Erfolgsquote von > 70 %.

---

## Phase 3: Funktionsapproximation (3-4 Wochen)

### Ziel
RL über kleine Zustandsräume hinaus skalieren durch den Einsatz von Funktionsapproximatoren.

### Zu lernende Konzepte
- Warum tabellarische Methoden bei großen Problemen scheitern
- Lineare Funktionsapproximation
- Neuronale Netzwerke als Funktionsapproximatoren
- Deep Q-Networks (DQN)
  - Experience Replay
  - Target-Netzwerke
  - Reward Clipping
- DQN-Verbesserungen
  - Double DQN
  - Dueling DQN
  - Prioritized Experience Replay
  - Rainbow DQN

### Praktische Arbeit
- [ ] [Lösen von CartPole mit linearem Q-Learning](function_approximation/linear_q_cartpole_explained.md)
- [ ] [Implementierung von DQN von Grund auf für CartPole](function_approximation/dqn_cartpole_explained.md)
- [ ] [Hinzufügen von Experience Replay, Beobachtung der Stabilitätsverbesserung](function_approximation/dqn_experience_replay_explained.md)
- [ ] [Hinzufügen eines Target-Netzwerks, Vergleich der Lernkurven](function_approximation/dqn_target_network_explained.md)
- [ ] [Training von DQN auf Atari Pong (ALE-Py verwenden)](function_approximation/dqn_atari_pong_explained.md)
- [ ] [Implementierung von Double DQN, Vergleich mit Standard-DQN](function_approximation/double_dqn_cartpole_explained.md)

### Kern-Erkenntnis
Die „tödliche Triade“ (Funktionsapproximation + Bootstrapping + Off-Policy) verursacht Instabilität. DQN-Innovationen adressieren dies.

### Werkzeuge
- PyTorch oder TensorFlow
- Gymnasium
- ALE-Py (für Atari)
- Weights & Biases (für Experiment-Tracking)

### Ressourcen
- 📖 Sutton & Barto - Kapitel 9-11
- 📄 [DQN-Paper (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)
- 📄 [Rainbow-Paper](https://arxiv.org/abs/1710.02298)
- 💻 [CleanRL Implementierungen](https://github.com/vwxyzjn/cleanrl)

### Meilenstein
Trainieren Sie einen DQN-Agenten, der eine positive Belohnung bei Atari Pong erzielt.

---

## Phase 4: Policy-Gradient-Methoden (4-5 Wochen)

### Ziel
Lernen, Strategien (Policies) direkt zu optimieren, ohne Value-Funktionen berechnen zu müssen.

### Zu lernende Konzepte
- Policy-Gradient-Theorem
- REINFORCE-Algorithmus
- Techniken zur Varianzreduktion
  - Baselines
  - Reward-to-go
- Actor-Critic-Methoden
  - A2C (Advantage Actor-Critic)
  - A3C (Asynchrone Variante)
- Generalized Advantage Estimation (GAE)
- Trust-Region-Methoden
  - TRPO (Trust Region Policy Optimization)
  - PPO (Proximal Policy Optimization)

### Praktische Arbeit
- [ ] [Implementierung von REINFORCE für CartPole](policy_gradients/reinforce_cartpole_explained.md)
- [ ] [Hinzufügen einer Baseline, Messung der Varianzreduktion](policy_gradients/reinforce_baseline_explained.md)
- [ ] [Aufbau von A2C für LunarLander](policy_gradients/a2c_lunarlander_explained.md)
- [ ] [Implementierung von PPO von Grund auf](policy_gradients/ppo_scratch_explained.md)
- [ ] [Training von PPO auf kontinuierlicher Steuerung (BipedalWalker oder MuJoCo)](policy_gradients/ppo_continuous_explained.md)
- [ ] [Vergleich der PPO-Hyperparameter-Sensitivität](policy_gradients/ppo_hyperparams_explained.md)

### Kern-Erkenntnis
PPO ist das Arbeitstier des modernen RL – verstehen Sie den Clipping-Mechanismus im Detail.

### Werkzeuge
- PyTorch
- Gymnasium
- Stable-Baselines3 (als Referenz)
- MuJoCo oder Box2D (für kontinuierliche Steuerung)

### Ressourcen
- 📖 Sutton & Barto - Kapitel 13
- 📝 [OpenAI Spinning Up - Policy Gradients](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
- 📄 [PPO-Paper (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- 📄 [GAE-Paper](https://arxiv.org/abs/1506.02438)
- 🎥 [Pieter Abbeel's Deep RL Bootcamp](https://sites.google.com/view/deep-rl-bootcamp/lectures)

### Meilenstein
Implementieren Sie PPO und lösen Sie BipedalWalker-v3 (Belohnung > 300).

---

## Phase 5: Fortgeschrittene Themen (6-8 Wochen)

Wählen Sie 2-3 Bereiche basierend auf Ihren Interessen aus.
- [Modellbasiertes RL](#modellbasiertes-rl)
- [Multiagenten-RL](#multiagenten-rl)
- [Offline/Batch RL](#offlinebatch-rl)
- [Exploration](#exploration)
- [Hierarchisches RL](#hierarchisches-rl)
- [RLHF (RL from Human Feedback)](#rlhf-rl-aus-menschlichem-feedback)

### Modellbasiertes RL
Lernen der Umgebungsdynamik, um zu planen oder synthetische Erfahrungen zu generieren.

**Konzepte:**
- Dyna-Architektur
- Weltmodelle
- Model Predictive Control (MPC)
- MuZero, Dreamer

**Praktische Arbeit:**
- [ ] [Implementierung von Dyna-Q](advanced_topics/model_based_rl/dyna_q_explained.md)
- [ ] [Training eines Weltmodells in einer einfachen Umgebung](advanced_topics/model_based_rl/world_model_explained.md)
- [ ] [Nutzung eines gelernten Modells für die Planung](advanced_topics/model_based_rl/model_based_planning_explained.md)

**Ressourcen:**
- 📖 Sutton & Barto - Kapitel 8
- 📄 [World Models Paper](https://arxiv.org/abs/1803.10122)
- 📄 [MuZero Paper](https://arxiv.org/abs/1911.08265)
- 📄 [DreamerV3 Paper](https://arxiv.org/abs/2301.04104)

---

### Multiagenten-RL
Mehrere Agenten lernen gleichzeitig in gemeinsamen Umgebungen.

**Konzepte:**
- Unabhängige Lerner
- Zentralisiertes Training, dezentrale Ausführung (CTDE)
- Self-play
- Emergente Kommunikation

**Praktische Arbeit:**
- [ ] [Agenten in einfachen Matrix-Spielen trainieren](advanced_topics/multi_agent_rl/matrix_games_explained.md)
- [ ] [Self-play für ein Brettspiel implementieren](advanced_topics/multi_agent_rl/self_play_tic_tac_toe_explained.md)
- [ ] [Erkundung von PettingZoo-Umgebungen](advanced_topics/multi_agent_rl/pettingzoo_explore_explained.md)

**Ressourcen:**
- 📄 [Multi-Agent RL Survey](https://arxiv.org/abs/1911.10635)
- 📄 [OpenAI Five](https://arxiv.org/abs/1912.06680)
- 💻 [PettingZoo Bibliothek](https://pettingzoo.farama.org/)

---

### Offline/Batch RL
Lernen aus festen Datensätzen ohne Interaktion mit der Umgebung.

**Konzepte:**
- Problem der Verteilungsverschiebung (Distribution Shift)
- Conservative Q-Learning (CQL)
- Implicit Q-Learning (IQL)
- Decision Transformer

**Praktische Arbeit:**
- [ ] [Training auf D4RL Benchmark-Datensätzen](advanced_topics/offline_rl/d4rl_dataset_explained.md)
- [ ] [Implementierung von CQL](advanced_topics/offline_rl/cql_explained.md)
- [ ] [Vergleich mit Behavioral Cloning](advanced_topics/offline_rl/behavioral_cloning_explained.md)

**Ressourcen:**
- 📄 [Offline RL Tutorial](https://arxiv.org/abs/2005.01643)
- 📄 [CQL Paper](https://arxiv.org/abs/2006.04779)
- 📄 [Decision Transformer](https://arxiv.org/abs/2106.01345)
- 💻 [D4RL Benchmark](https://github.com/Farama-Foundation/D4RL)

---

### Exploration
Adressierung von Problemen mit spärlichen Belohnungen und schwieriger Exploration.

**Konzepte:**
- Intrinsische Motivation
- Neugiergetriebene Exploration (ICM)
- Random Network Distillation (RND)
- Zählbasierte Exploration
- Go-Explore

**Praktische Arbeit:**
- [ ] [Implementierung eines Curiosity-Bonus](advanced_topics/exploration/curiosity_bonus_explained.md)
- [ ] [Training auf Montezuma's Revenge](advanced_topics/exploration/montezuma_revenge_explained.md)
- [ ] [Vergleich von Explorationsstrategien](advanced_topics/exploration/compare_exploration_explained.md)

**Ressourcen:**
- 📄 [ICM Paper](https://arxiv.org/abs/1705.05363)
- 📄 [RND Paper](https://arxiv.org/abs/1810.12894)
- 📄 [Go-Explore](https://arxiv.org/abs/1901.10995)
- 📄 [Deep Exploration via Bootstrapped DQN](https://arxiv.org/abs/1602.04621) — Quelle der DeepSea-Aufgabe

---

### Hierarchisches RL
Lernen auf mehreren Ebenen zeitlicher Abstraktion.

**Konzepte:**
- Options-Framework
- Zielkonditionierte Strategien (Goal-conditioned policies)
- Feudal Networks
- HIRO, HAM

**Praktische Arbeit:**
- [ ] [Implementierung der Option-Critic-Architektur](advanced_topics/hierarchical_rl/option_critic_explained.md)
- [ ] [Training einer zielkonditionierten Strategie](advanced_topics/hierarchical_rl/goal_conditioned_policy_explained.md)
- [ ] [Test auf Aufgaben mit langem Zeithorizont](advanced_topics/hierarchical_rl/long_horizon_tasks_explained.md)

**Ressourcen:**
- 📖 Sutton & Barto - Kapitel 12 (Options)
- 📄 [Option-Critic Paper](https://arxiv.org/abs/1609.05140)
- 📄 [HIRO Paper](https://arxiv.org/abs/1805.08296)

---

### RLHF (RL aus menschlichem Feedback)
Ausrichtung (Alignment) von Modellen an menschlichen Präferenzen.

**Konzepte:**
- Belohnungsmodellierung aus Vergleichen
- KL-beschränkte Policy-Optimierung
- Constitutional AI
- DPO (Direct Preference Optimization)

**Praktische Arbeit:**
- [ ] [Training eines Belohnungsmodells aus Präferenzdaten](advanced_topics/rlhf/reward_modeling_explained.md)
- [ ] [Feintuning eines kleinen LM mit PPO](advanced_topics/rlhf/ppo_fine_tuning_explained.md)
- [ ] [Implementierung von DPO](advanced_topics/rlhf/dpo_implementation_explained.md)

**Ressourcen:**
- 📄 [InstructGPT Paper](https://arxiv.org/abs/2203.02155)
- 📄 [Constitutional AI](https://arxiv.org/abs/2212.08073)
- 📄 [DPO Paper](https://arxiv.org/abs/2305.18290)
- 💻 [TRL Bibliothek](https://github.com/huggingface/trl)

---

## Phase 6: Forschungsniveau (Fortlaufend)

### Ziel
Eigene Originalarbeiten zum Fachgebiet beisteuern.

### Aktivitäten
- Regelmäßiges Lesen von Papern der NeurIPS, ICML, ICLR
- Reproduktion zentraler Ergebnisse aktueller Veröffentlichungen
- Identifikation offener Probleme und Einschränkungen
- Durchführung rigoroser Experimente mit ordnungsgemäßer Evaluierung
- Berücksichtigung von Dateneffizienz, Generalisierung und Sicherheit

### Wegweisende Paper zum Studium
- [ ] [AlphaGo (Silver et al., 2016)](https://www.nature.com/articles/nature16961)
- [ ] [AlphaZero (Silver et al., 2018)](https://www.science.org/doi/10.1126/science.aar6404)
- [ ] [MuZero (Schrittwieser et al., 2020)](https://arxiv.org/abs/1911.08265)
- [ ] [Decision Transformer (Chen et al., 2021)](https://arxiv.org/abs/2106.01345)
- [ ] [InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)
- [ ] [DreamerV3 (Hafner et al., 2023)](https://arxiv.org/abs/2301.04104)

### Wo publizieren?
- NeurIPS, ICML, ICLR (Top-Konferenzen)
- AAAI, IJCAI
- CoRL (Schwerpunkt Robotik)
- Workshop-Beiträge für frühe Arbeiten

---

## Zusammenfassung der Werkzeuge

| Phase | Empfohlene Werkzeuge |
|-------|-------------------|
| 1-2 | NumPy, Matplotlib |
| 3 | PyTorch, Gymnasium, ALE-Py, W&B |
| 4 | Stable-Baselines3, MuJoCo/Box2D |
| 5-6 | Eigener Code, Ray/RLlib, JAX (für Geschwindigkeit) |

---

## Tipps für den Erfolg

1. **Zuerst von Grund auf implementieren** — Bibliotheken erst nutzen, wenn der Algorithmus verstanden wurde.
2. **Debuggen mit einfachen Umgebungen** — Immer CartPole vor Atari.
3. **Alles protokollieren** — Belohnungen, Verluste, Gradienten, Episodenlängen.
4. **Lernen visualisieren** — Lernkurven plotten, Episoden rendern.
5. **Das Buch von Sutton & Barto lesen** — Es ist die Bibel des RL.
6. **Die Mathematik verstehen** — Zumindest das Policy-Gradient-Theorem und die Bellman-Gleichungen.
7. **Geduldig sein** — RL ist bekanntlich instabil; fehlgeschlagene Läufe sind normal.
8. **Seeds verwenden** — Reproduzierbarkeit zählt; über mehrere Seeds mitteln.
9. **Communities beitreten** — r/reinforcementlearning, RL Discord, Twitter/X.

---

## Häufige Fallstricke

- ❌ Grundlagen überspringen, um direkt ins Deep RL einzusteigen
- ❌ Beobachtungen/Belohnungen nicht normalisieren
- ❌ Zu große/kleine Lernraten verwenden
- ❌ Vergessen, beim Testen den Evaluierungsmodus zu setzen
- ❌ Nicht genügend Seeds für Experimente verwenden
- ❌ Implementierung aus Papern ohne Prüfung des Referenzcodes
- ❌ Nach einem einzigen fehlgeschlagenen Trainingslauf aufgeben

---

## Glossar

| Begriff | Definition |
|------|------------|
| **MDP** | Markov Decision Process - formaler Rahmen für RL |
| **Policy (π)** | Abbildung von Zuständen auf Aktionen |
| **Value-Funktion (V)** | Erwarteter Return aus einem Zustand |
| **Q-Funktion** | Erwarteter Return aus einem Zustand-Aktions-Paar |
| **TD-Fehler** | Differenz zwischen vorhergesagtem und bootstrapped Wert |
| **GAE** | Generalized Advantage Estimation |
| **PPO** | Proximal Policy Optimization |
| **RLHF** | Reinforcement Learning from Human Feedback |

---

## Lizenz

Dieser Leitfaden wird zu Bildungszwecken zur Verfügung gestellt. Er darf gerne geteilt und angepasst werden.

---
