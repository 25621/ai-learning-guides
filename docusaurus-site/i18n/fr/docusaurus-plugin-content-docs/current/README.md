# Apprentissage par Renforcement : de Débutant à Avancé

Un guide structuré pour maîtriser l'apprentissage par renforcement (RL), des concepts fondamentaux à la recherche de pointe.

---

## Aperçu

| Phase | Objectif | Durée |
|-------|-------|----------|
| 1 | [Fondations](#phase-1--fondations-2-à-4-semaines) | 2-4 semaines |
| 2 | [Méthodes Tabulaires](#phase-2--méthodes-tabulaires-3-à-4-semaines) | 3-4 semaines |
| 3 | [Approximation de Fonction](#phase-3--approximation-de-fonction-3-à-4-semaines) | 3-4 semaines |
| 4 | [Méthodes de Gradient de Politique](#phase-4--méthodes-de-gradient-de-politique-4-à-5-semaines) | 4-5 semaines |
| 5 | [Sujets Avancés](#phase-5--sujets-avancés-6-à-8-semaines) | 6-8 semaines |
| 6 | [Niveau Recherche](#phase-6--niveau-recherche-continu) | Continu |

**Temps total pour devenir compétent :** ~6 mois

---

## Phase 1 : Fondations (2 à 4 semaines)

### Objectif
Comprendre les concepts de base sans mathématiques approfondies.

### Concepts à Apprendre
- Agent, environnement, état, action, récompense
- Processus de Décision de Markov (MDP)
- Retour et facteur d'actualisation (γ)
- Politique vs fonction de valeur
- Compromis exploration/exploitation

### Travaux Pratiques
- [ ] [Implémenter un problème simple de bandit manchot à partir de zéro](foundations/multi_armed_bandit_explained.md)
- [ ] [Résoudre Frozen Lake avec une politique aléatoire, observer les résultats](foundations/frozen_lake_explained.md)
- [ ] [Visualiser les fonctions de valeur d'état sur une grille simple](foundations/state_value_visualization_explained.md)

### Outils
- Python
- NumPy
- Matplotlib (pour la visualisation)

### Ressources
- 📖 [Sutton & Barto](http://incompleteideas.net/book/the-book.html) - Chapitres 1-3
- 🎥 [Cours de David Silver](https://www.youtube.com/watch?v=2pWv7GOvuf0) - Cours 1-2
- 📝 [OpenAI Spinning Up - Introduction](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)

### Jalon
Vous devriez être capable d'expliquer la formulation du problème RL et pourquoi les MDP sont le cadre standard.

---

## Phase 2 : Méthodes Tabulaires (3 à 4 semaines)

### Objectif
Maîtriser les algorithmes RL classiques où les espaces d'états/actions sont suffisamment petits pour être stockés dans des tableaux.

### Concepts à Apprendre
- Programmation Dynamique
  - Évaluation de politique
  - Itération de politique
  - Itération de valeur
- Méthodes de Monte Carlo
  - MC première visite vs chaque visite
  - Contrôle MC avec départs exploratoires
- Apprentissage par Différence Temporelle (TD)
  - Prédiction TD(0)
  - SARSA (contrôle TD on-policy)
  - Q-learning (contrôle TD off-policy)
- Traces d'éligibilité et TD(λ)

### Travaux Pratiques
- [ ] [Implémenter l'itération de politique pour GridWorld](tabular_methods/policy_iteration_gridworld_explained.md)
- [ ] [Construire un agent Q-learning pour Frozen Lake](tabular_methods/q_learning_frozen_lake_explained.md)
- [ ] [Implémenter SARSA pour Cliff Walking](tabular_methods/sarsa_cliff_walking_explained.md)
- [ ] [Comparer le comportement de SARSA vs Q-learning (chemins sûrs vs optimaux)](tabular_methods/sarsa_vs_qlearning_explained.md)
- [ ] [Implémenter le contrôle Monte Carlo pour le Blackjack](tabular_methods/monte_carlo_blackjack_explained.md)

### Point Clé
Comprendre les diagrammes de "backup" — ils clarifient comment chaque algorithme met à jour les valeurs.

### Outils
- NumPy
- Gymnasium (pour les environnements)

### Ressources
- 📖 Sutton & Barto - Chapitres 4-7
- 🎥 Cours de David Silver 3-5
- 💻 [Dépôt RL de Denny Britz](https://github.com/dennybritz/reinforcement-learning)

### Jalon
Implémenter le Q-learning à partir de zéro et résoudre Frozen Lake avec un taux de réussite > 70 %.

---

## Phase 3 : Approximation de Fonction (3 à 4 semaines)

### Objectif
Passer à l'échelle du RL au-delà des petits espaces d'états en utilisant des approximateurs de fonction.

### Concepts à Apprendre
- Pourquoi les méthodes tabulaires échouent à grande échelle
- Approximation de fonction linéaire
- Les réseaux de neurones comme approximateurs de fonction
- Deep Q-Networks (DQN)
  - Rejeu d'expérience (Experience replay)
  - Réseaux cibles (Target networks)
  - Écrêtage des récompenses (Reward clipping)
- Améliorations de DQN
  - Double DQN
  - Dueling DQN
  - Rejeu d'expérience priorisé (Prioritized Experience Replay)
  - Rainbow DQN

### Travaux Pratiques
- [ ] [Résoudre CartPole avec le Q-learning linéaire](function_approximation/linear_q_cartpole_explained.md)
- [ ] [Implémenter DQN à partir de zéro pour CartPole](function_approximation/dqn_cartpole_explained.md)
- [ ] [Ajouter le rejeu d'expérience, observer l'amélioration de la stabilité](function_approximation/dqn_experience_replay_explained.md)
- [ ] [Ajouter un réseau cible, comparer les courbes d'apprentissage](function_approximation/dqn_target_network_explained.md)
- [ ] [Entraîner DQN sur Atari Pong (utiliser ALE-Py)](function_approximation/dqn_atari_pong_explained.md)
- [ ] [Implémenter Double DQN, comparer avec DQN standard](function_approximation/double_dqn_cartpole_explained.md)

### Point Clé
La "triade mortelle" (approximation de fonction + bootstrapping + off-policy) provoque de l'instabilité. Les innovations de DQN s'attaquent à cela.

### Outils
- PyTorch ou TensorFlow
- Gymnasium
- ALE-Py (pour Atari)
- Weights & Biases (pour le suivi d'expériences)

### Ressources
- 📖 Sutton & Barto - Chapitres 9-11
- 📄 [Article DQN (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)
- 📄 [Article Rainbow](https://arxiv.org/abs/1710.02298)
- 💻 [Implémentations CleanRL](https://github.com/vwxyzjn/cleanrl)

### Jalon
Entraîner un agent DQN qui obtient une récompense positive sur Atari Pong.

---

## Phase 4 : Méthodes de Gradient de Politique (4 à 5 semaines)

### Objectif
Apprendre à optimiser directement les politiques sans calculer de fonctions de valeur.

### Concepts à Apprendre
- Théorème du Gradient de Politique
- Algorithme REINFORCE
- Techniques de réduction de la variance
  - Lignes de base (Baselines)
  - Récompense à venir (Reward-to-go)
- Méthodes Actor-Critic
  - A2C (Advantage Actor-Critic)
  - A3C (variante asynchrone)
- Estimation de l'Avantage Généralisée (GAE)
- Méthodes de Région de Confiance
  - TRPO (Trust Region Policy Optimization)
  - PPO (Proximal Policy Optimization)

### Travaux Pratiques
- [ ] [Implémenter REINFORCE pour CartPole](policy_gradients/reinforce_cartpole_explained.md)
- [ ] [Ajouter une ligne de base, mesurer la réduction de variance](policy_gradients/reinforce_baseline_explained.md)
- [ ] [Construire A2C pour LunarLander](policy_gradients/a2c_lunarlander_explained.md)
- [ ] [Implémenter PPO à partir de zéro](policy_gradients/ppo_scratch_explained.md)
- [ ] [Entraîner PPO sur le contrôle continu (BipedalWalker ou MuJoCo)](policy_gradients/ppo_continuous_explained.md)
- [ ] [Comparer la sensibilité des hyperparamètres de PPO](policy_gradients/ppo_hyperparams_explained.md)

### Point Clé
PPO est le moteur du RL moderne — comprenez son mécanisme d'écrêtage (clipping) en profondeur.

### Outils
- PyTorch
- Gymnasium
- Stable-Baselines3 (pour référence)
- MuJoCo ou Box2D (pour le contrôle continu)

### Ressources
- 📖 Sutton & Barto - Chapitre 13
- 📝 [OpenAI Spinning Up - Policy Gradients](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
- 📄 [Article PPO (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- 📄 [Article GAE](https://arxiv.org/abs/1506.02438)
- 🎥 [Deep RL Bootcamp de Pieter Abbeel](https://sites.google.com/view/deep-rl-bootcamp/lectures)

### Jalon
Implémenter PPO et résoudre BipedalWalker-v3 (récompense > 300).

---

## Phase 5 : Sujets Avancés (6 à 8 semaines)

Choisissez 2-3 domaines en fonction de vos intérêts.
- [RL Basé sur un Modèle (Model-Based)](#rl-basé-sur-un-modèle-model-based)
- [RL Multi-Agents](#rl-multi-agents)
- [RL Hors-ligne / Batch](#rl-hors-ligne--batch)
- [Exploration](#exploration)
- [RL Hiérarchique](#rl-hiérarchique)
- [RLHF (RL à partir de retours humains)](#rlhf-rl-à-partir-de-retours-humains)

### RL Basé sur un Modèle (Model-Based)
Apprendre la dynamique de l'environnement pour planifier ou générer de l'expérience synthétique.

**Concepts :**
- Architecture Dyna
- Modèles du monde (World models)
- Contrôle Prédictif de Modèle (MPC)
- MuZero, Dreamer

**Travaux Pratiques :**
- [ ] [Implémenter Dyna-Q](advanced_topics/model_based_rl/dyna_q_explained.md)
- [ ] [Entraîner un modèle du monde sur un environnement simple](advanced_topics/model_based_rl/world_model_explained.md)
- [ ] [Utiliser le modèle appris pour la planification](advanced_topics/model_based_rl/model_based_planning_explained.md)

**Ressources :**
- 📖 Sutton & Barto - Chapitre 8
- 📄 [Article World Models](https://arxiv.org/abs/1803.10122)
- 📄 [Article MuZero](https://arxiv.org/abs/1911.08265)
- 📄 [Article DreamerV3](https://arxiv.org/abs/2301.04104)

---

### RL Multi-Agents
Plusieurs agents apprenant simultanément dans des environnements partagés.

**Concepts :**
- Apprenants indépendants
- Entraînement centralisé, exécution décentralisée (CTDE)
- Auto-jeu (Self-play)
- Communication émergente

**Travaux Pratiques :**
- [ ] [Entraîner des agents dans des jeux matriciels simples](advanced_topics/multi_agent_rl/matrix_games_explained.md)
- [ ] [Implémenter l'auto-jeu pour un jeu de société](advanced_topics/multi_agent_rl/self_play_tic_tac_toe_explained.md)
- [ ] [Explorer les environnements PettingZoo](advanced_topics/multi_agent_rl/pettingzoo_explore_explained.md)

**Ressources :**
- 📄 [Revue du RL Multi-Agents](https://arxiv.org/abs/1911.10635)
- 📄 [OpenAI Five](https://arxiv.org/abs/1912.06680)
- 💻 [Bibliothèque PettingZoo](https://pettingzoo.farama.org/)

---

### RL Hors-ligne / Batch
Apprendre à partir de jeux de données fixes sans interaction avec l'environnement.

**Concepts :**
- Problème du décalage de distribution (distribution shift)
- Q-Learning Conservateur (CQL)
- Q-Learning Implicite (IQL)
- Decision Transformer

**Travaux Pratiques :**
- [ ] [Entraîner sur les jeux de données de référence D4RL](advanced_topics/offline_rl/d4rl_dataset_explained.md)
- [ ] [Implémenter CQL](advanced_topics/offline_rl/cql_explained.md)
- [ ] [Comparer avec le clonage comportemental](advanced_topics/offline_rl/behavioral_cloning_explained.md)

**Ressources :**
- 📄 [Tutoriel RL Hors-ligne](https://arxiv.org/abs/2005.01643)
- 📄 [Article CQL](https://arxiv.org/abs/2006.04779)
- 📄 [Decision Transformer](https://arxiv.org/abs/2106.01345)
- 💻 [Benchmark D4RL](https://github.com/Farama-Foundation/D4RL)

---

### Exploration
S'attaquer aux problèmes de récompense éparse et d'exploration difficile.

**Concepts :**
- Motivation intrinsèque
- Exploration guidée par la curiosité (ICM)
- Distillation de Réseau Aléatoire (RND)
- Exploration basée sur le comptage
- Go-Explore

**Travaux Pratiques :**
- [ ] [Implémenter le bonus de curiosité](advanced_topics/exploration/curiosity_bonus_explained.md)
- [ ] [Entraîner sur Montezuma's Revenge](advanced_topics/exploration/montezuma_revenge_explained.md)
- [ ] [Comparer les stratégies d'exploration](advanced_topics/exploration/compare_exploration_explained.md)

**Ressources :**
- 📄 [Article ICM](https://arxiv.org/abs/1705.05363)
- 📄 [Article RND](https://arxiv.org/abs/1810.12894)
- 📄 [Go-Explore](https://arxiv.org/abs/1901.10995)
- 📄 [Deep Exploration via Bootstrapped DQN](https://arxiv.org/abs/1602.04621) — source de la tâche DeepSea

---

### RL Hiérarchique
Apprendre à plusieurs niveaux d'abstraction temporelle.

**Concepts :**
- Cadre des Options (Options framework)
- Politiques conditionnées par des objectifs (Goal-conditioned policies)
- Réseaux féodaux (Feudal networks)
- HIRO, HAM

**Travaux Pratiques :**
- [ ] [Implémenter l'architecture option-critic](advanced_topics/hierarchical_rl/option_critic_explained.md)
- [ ] [Entraîner une politique conditionnée par des objectifs](advanced_topics/hierarchical_rl/goal_conditioned_policy_explained.md)
- [ ] [Tester sur des tâches à long horizon](advanced_topics/hierarchical_rl/long_horizon_tasks_explained.md)

**Ressources :**
- 📖 Sutton & Barto - Chapitre 12 (Options)
- 📄 [Article Option-Critic](https://arxiv.org/abs/1609.05140)
- 📄 [Article HIRO](https://arxiv.org/abs/1805.08296)

---

## Phase 5 : RLHF (RL à partir de retours humains)
Aligner les modèles avec les préférences humaines.

**Concepts :**
- Modélisation de la récompense à partir de comparaisons
- Optimisation de politique sous contrainte KL
- IA Constitutionnelle
- DPO (Direct Preference Optimization)

**Travaux Pratiques :**
- [ ] [Entraîner un modèle de récompense à partir de données de préférence](advanced_topics/rlhf/reward_modeling_explained.md)
- [ ] [Ajuster un petit modèle de langage (LM) avec PPO](advanced_topics/rlhf/ppo_fine_tuning_explained.md)
- [ ] [Implémenter DPO](advanced_topics/rlhf/dpo_implementation_explained.md)

**Ressources :**
- 📄 [Article InstructGPT](https://arxiv.org/abs/2203.02155)
- 📄 [IA Constitutionnelle](https://arxiv.org/abs/2212.08073)
- 📄 [Article DPO](https://arxiv.org/abs/2305.18290)
- 💻 [Bibliothèque TRL](https://github.com/huggingface/trl)

---

## Phase 6 : Niveau Recherche (Continu)

### Objectif
Contribuer par des travaux originaux au domaine.

### Activités
- Lire régulièrement les articles de NeurIPS, ICML, ICLR
- Reproduire les résultats clés des articles récents
- Identifier les problèmes ouverts et les limites
- Mener des expériences rigoureuses avec une évaluation appropriée
- Considérer l'efficacité des échantillons, la généralisation, la sécurité

### Articles Marquants à Étudier
- [ ] [AlphaGo (Silver et al., 2016)](https://www.nature.com/articles/nature16961)
- [ ] [AlphaZero (Silver et al., 2018)](https://www.science.org/doi/10.1126/science.aar6404)
- [ ] [MuZero (Schrittwieser et al., 2020)](https://arxiv.org/abs/1911.08265)
- [ ] [Decision Transformer (Chen et al., 2021)](https://arxiv.org/abs/2106.01345)
- [ ] [InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)
- [ ] [DreamerV3 (Hafner et al., 2023)](https://arxiv.org/abs/2301.04104)

### Où Publier
- NeurIPS, ICML, ICLR (meilleurs lieux)
- AAAI, IJCAI
- CoRL (focus robotique)
- Articles d'atelier (Workshop) pour les travaux précoces

---

## Résumé des Outils

| Phase | Outils Recommandés |
|-------|-------------------|
| 1-2 | NumPy, Matplotlib |
| 3 | PyTorch, Gymnasium, ALE-Py, W&B |
| 4 | Stable-Baselines3, MuJoCo/Box2D |
| 5-6 | Code personnalisé, Ray/RLlib, JAX (pour la vitesse) |

---

## Conseils pour Réussir

1. **Implémenter à partir de zéro d'abord** — N'utilisez des bibliothèques qu'après avoir compris l'algorithme
2. **Déboguer avec des environnements simples** — CartPole avant Atari, toujours
3. **Tout journaliser** — Récompenses, pertes, gradients, longueurs d'épisodes
4. **Visualiser l'apprentissage** — Tracer les courbes d'apprentissage, rendre les épisodes
5. **Lire le livre de Sutton & Barto** — C'est la bible du RL
6. **Comprendre les maths** — Au moins le théorème du gradient de politique et les équations de Bellman
7. **Être patient** — Le RL est notoirement instable ; les échecs sont normaux
8. **Utiliser des graines (seeds)** — La reproductibilité compte ; faites la moyenne sur plusieurs graines
9. **Rejoindre des communautés** — r/reinforcementlearning, Discord RL, Twitter/X

---

## Pièges Courants à Éviter

- ❌ Sauter les fondamentaux pour se lancer directement dans le deep RL
- ❌ Ne pas normaliser les observations/récompenses
- ❌ Utiliser des taux d'apprentissage trop grands ou trop petits
- ❌ Oublier de régler le mode évaluation pendant les tests
- ❌ Ne pas utiliser assez de graines pour les expériences
- ❌ Implémenter à partir d'articles sans vérifier le code de référence
- ❌ Abandonner après un échec d'entraînement

---

## Glossaire

| Terme | Définition |
|------|------------|
| **MDP** | Processus de Décision de Markov - cadre formel pour le RL |
| **Politique (π)** | Correspondance entre états et actions |
| **Fonction de Valeur (V)** | Retour attendu à partir d'un état |
| **Fonction Q** | Retour attendu à partir d'un couple état-action |
| **Erreur TD** | Différence entre valeur prédite et bootstrap |
| **GAE** | Estimation de l'Avantage Généralisée |
| **PPO** | Optimisation de Politique Proximale |
| **RLHF** | Apprentissage par Renforcement à partir de retours humains |

---

## Licence

Ce guide est fourni à des fins éducatives. N'hésitez pas à le partager et à l'adapter.
