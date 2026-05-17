# Versterkend leren: van beginner tot gevorderd

Een uitgebreide gids voor het begrijpen en bouwen van versterkende leersystemen, van fundamentele concepten tot begrip op onderzoeksniveau.

---

## Overzicht

| Phase | Focus | Duration |
|-------|-------|----------|
| 1 | [Funderingen](#phase-1-foundations-2-4-weeks) | 2-4 weken |
| 2 | [Tabulaire methoden](#phase-2-tabular-methods-3-4-weeks) | 3-4 weken |
| 3 | [Functiebenadering](#phase-3-function-approximation-3-4-weeks) | 3-4 weken |
| 4 | [Beleidsgradiëntmethoden](#phase-4-policy-gradient-methods-4-5-weeks) | 4-5 weken |
| 5 | [Geavanceerde onderwerpen](#phase-5-advanced-topics-6-8-weeks) | 6-8 weken |
| 6 | [Onderzoeksniveau](#phase-6-research-level-ongoing) | Lopend |

**Totale tijd tot vaardigheid:** ~6 maanden

---

## Fase 1: Fundering (2-4 weken) {#phase-1-foundations-2-4-weeks}

### Doel
Begrijp kernconcepten zonder diepgaande wiskunde.

### Concepten om te leren
- Agent, omgeving, staat, actie, beloning
- Markov-beslissingsprocessen (MDP)
- Rendement- en kortingsfactor (γ)
- Beleid versus waardefunctie
- Afweging tussen exploratie en exploitatie

### Praktisch werk
- [ ] [Implementeer een eenvoudig meerarmig bandietenprobleem vanaf het begin](foundations/multi_armed_bandit_explained.md)
- [ ] [Los Frozen Lake op met willekeurig beleid, bekijk de resultaten](foundations/frozen_lake_explained.md)
- [ ] [Visualiseer statuswaardefuncties op een eenvoudig raster](foundations/state_value_visualization_explained.md)

### Gereedschap
- Python
- NumPy
- Matplotlib (voor visualisatie)

### Bronnen
- 📖 [Sutton & Barto](http://incompleteideas.net/book/the-book.html) - Hoofdstukken 1-3
- 🎥 [David Silver-lezingen](https://www.youtube.com/watch?v=2pWv7GOvuf0) - Lezingen 1-2
- 📝 [OpenAI draait omhoog - Introductie](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)

### Mijlpaal
Je zou de RL-probleemformulering moeten kunnen uitleggen en waarom MDP's het standaardraamwerk zijn.

---

## Fase 2: Tabellarische methoden (3-4 weken) {#phase-2-tabular-methods-3-4-weeks}

### Doel
Beheers klassieke RL-algoritmen waarbij status-/actieruimten klein genoeg zijn om in tabellen op te slaan.

### Concepten om te leren
- Dynamische programmering
  - Beleidsevaluatie
  - Beleidsiteratie
  - Waarde-iteratie
- Monte Carlo-methoden
  - MC bij eerste bezoek versus MC bij elk bezoek
  - MC-besturing met verkenningsstarts
- Leren van temporele verschillen
  - TD(0)-voorspelling
  - SARSA (on-policy TD-controle)
  - Q-learning (TD-controle buiten het beleid)
- Geschiktheidssporen en TD(λ)

### Praktisch werk
- [ ] [Beleiditeratie implementeren voor GridWorld](tabular_methods/policy_iteration_gridworld_explained.md)
- [ ] [Q-learning-agent bouwen voor Frozen Lake](tabular_methods/q_learning_frozen_lake_explained.md)
- [ ] [SARSA implementeren voor klifwandelingen](tabular_methods/sarsa_cliff_walking_explained.md)
- [ ] [Vergelijk SARSA versus Q-leergedrag (veilig versus optimaal pad)](tabular_methods/sarsa_vs_qlearning_explained.md)
- [ ] [Monte Carlo-controle implementeren voor Blackjack](tabular_methods/monte_carlo_blackjack_explained.md)

### Belangrijk inzicht
Begrijp de back-updiagrammen: ze verduidelijken hoe elk algoritme waarden bijwerkt.

### Gereedschap
- NumPy
- Gymnasium (voor omgevingen)

### Bronnen
- 📖 Sutton & Barto - Hoofdstukken 4-7
- 🎥 David Silver-lezingen 3-5
- 💻 [De RL-repository van Denny Britz](https://github.com/dennybritz/reinforcement-learning)

### Mijlpaal
Implementeer Q-learning helemaal opnieuw en los Frozen Lake op met een succespercentage van >70%.

---

## Fase 3: Functie-aanpassing (3-4 weken) {#phase-3-function-approximation-3-4-weeks}

### Doel
Schaal RL voorbij kleine toestandsruimten met behulp van functiebenaderers.

### Concepten om te leren
- Waarom tabellarische methoden op grote schaal falen
- Lineaire functiebenadering
- Neurale netwerken als functiebenaderers
- Deep Q-netwerken (DQN)
  - Ervaar herhaling
  - Doelnetwerken
  - Beloning knippen
- DQN-verbeteringen
  - Dubbele DQN
  - Duellerende DQN
  - Geprioriteerde ervaringsherhaling
  - Regenboog DQN

### Praktisch werk
- [ ] [CarPole oplossen met lineaire Q-learning](function_approximation/linear_q_cartpole_explained.md)
- [ ] [DQN helemaal opnieuw implementeren voor CartPole](function_approximation/dqn_cartpole_explained.md)
- [ ] [Voeg ervaringsherhaling toe, observeer stabiliteitsverbetering](function_approximation/dqn_experience_replay_explained.md)
- [ ] [Doelnetwerk toevoegen, leercurves vergelijken](function_approximation/dqn_target_network_explained.md)
- [ ] [Train DQN op Atari Pong (gebruik ALE-Py)](function_approximation/dqn_atari_pong_explained.md)
- [ ] [Dubbele DQN implementeren, vergelijken met gewone DQN](function_approximation/double_dqn_cartpole_explained.md)

### Belangrijk inzicht
De ‘dodelijke triade’ (functiebenadering + bootstrapping + afwijkend beleid) veroorzaakt instabiliteit. DQN-innovaties pakken dit aan.

### Gereedschap
-PyTorch of TensorFlow
- Gymnasium
-ALE-Py (voor Atari)
- Gewichten en vooroordelen (voor het volgen van experimenten)

### Bronnen
- 📖 Sutton & Barto - Hoofdstukken 9-11
- 📄 [DQN Paper (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)
- 📄 [Regenboogpapier](https://arxiv.org/abs/1710.02298)
- 💻 [CleanRL-implementaties](https://github.com/vwxyzjn/cleanrl)

### Mijlpaal
Train een DQN-agent die een positieve beloning behaalt op Atari Pong.

---

## Fase 4: Beleidsgradiëntmethoden (4-5 weken) {#phase-4-policy-gradient-methods-4-5-weeks}

### Doel
Leer hoe u beleid rechtstreeks kunt optimaliseren zonder waardefuncties te berekenen.

### Concepten om te leren
- Beleidsgradiëntstelling
- REINFORCE-algoritme
- Variantiereductietechnieken
  - Basislijnen
  - Beloning voor onderweg
- Acteur-criticusmethoden
  - A2C (Advantage Acteur-Criticus)
  - A3C (asynchrone variant)
- Gegeneraliseerde voordeelschatting (GAE)
- Vertrouwenregiomethoden
  - TRPO (Trust Region Policy Optimization)
  - PPO (Proximale Beleidsoptimalisatie)

### Praktisch werk
- [ ] [REINFORCE implementeren voor CartPole](policy_gradients/reinforce_cartpole_explained.md)
- [ ] [Basislijn toevoegen, variantiereductie meten](policy_gradients/reinforce_baseline_explained.md)
- [ ] [Bouw A2C voor LunarLander](policy_gradients/a2c_lunarlander_explained.md)
- [ ] [PPO helemaal opnieuw implementeren](policy_gradients/ppo_scratch_explained.md)
- [ ] [PPO trainen op continue controle (BipedalWalker of MuJoCo)](policy_gradients/ppo_continuous_explained.md)
- [ ] [Vergelijk PPO-hyperparametergevoeligheid](policy_gradients/ppo_hyperparams_explained.md)

### Belangrijk inzicht
PPO is het werkpaard van de moderne RL. Begrijp het clipmechanisme grondig.

### Gereedschap
-PyTorch
- Gymnasium
- Stabiele basislijnen3 (ter referentie)
- MuJoCo of Box2D (voor continue controle)

### Bronnen
- 📖 Sutton & Barto - Hoofdstuk 13
- 📝 [OpenAI draait omhoog - Beleidsgradiënten](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
- 📄 [PPO-paper (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- 📄 [GAE-papier](https://arxiv.org/abs/1506.02438)
- 🎥 [Deep RL Bootcamp van Pieter Abbeel](https://sites.google.com/view/deep-rl-bootcamp/lectures)

### Mijlpaal
Implementeer PPO en los BipedalWalker-v3 op (beloning> 300).

---

## Fase 5: Geavanceerde onderwerpen (6-8 weken) {#phase-5-advanced-topics-6-8-weeks}

Kies 2-3 gebieden op basis van uw interesses.
- [Modelgebaseerde RL](#model-based-rl)
- [RL voor meerdere agenten](#multi-agent-rl)
- [Offline/Batch-RL](#offlinebatch-rl)
- [Verkenning](#exploration)
- [Hierarchische RL](#hierarchical-rl)
- [RLHF (RL van menselijke feedback)](#rlhf-rl-from-human-feedback)

### Modelgebaseerde RL {#model-based-rl}
Leer de omgevingsdynamiek om synthetische ervaringen te plannen of te genereren.

**Begrippen:**
- Dyna-architectuur
- Wereldmodellen
- Model voorspellende controle (MPC)
- MuZero, Dromer

**Praktisch werk:**
- [ ] [Dyna-Q implementeren](advanced_topics/model_based_rl/dyna_q_explained.md)
- [ ] [Train een wereldmodel in een eenvoudige omgeving](advanced_topics/model_based_rl/world_model_explained.md)
- [ ] [Gebruik het geleerde model voor planning](advanced_topics/model_based_rl/model_based_planning_explained.md)

**Bronnen:**
- 📖 Sutton & Barto - Hoofdstuk 8
- 📄 [Wereldmodellenpapier](https://arxiv.org/abs/1803.10122)
- 📄 [MuZero-papier](https://arxiv.org/abs/1911.08265)
- 📄 [DreamerV3-papier](https://arxiv.org/abs/2301.04104)

---

### RL voor meerdere agenten {#multi-agent-rl}
Meerdere agenten leren tegelijkertijd in gedeelde omgevingen.

**Begrippen:**
- Onafhankelijke leerlingen
- Gecentraliseerde training, gedecentraliseerde uitvoering (CTDE)
- Zelf spelen
- Opkomende communicatie

**Praktisch werk:**
- [ ] [Train agenten in eenvoudige matrixspellen](advanced_topics/multi_agent_rl/matrix_games_explained.md)
- [ ] [Zelfspel implementeren voor een bordspel](advanced_topics/multi_agent_rl/self_play_tic_tac_toe_explained.md)
- [ ] [Verken de kinderboerderij-omgevingen](advanced_topics/multi_agent_rl/pettingzoo_explore_explained.md)

**Bronnen:**
- 📄 [RL-enquête met meerdere agenten](https://arxiv.org/abs/1911.10635)
- 📄 [OpenAI Vijf](https://arxiv.org/abs/1912.06680)
- 💻 [Kinderboerderijbibliotheek](https://pettingzoo.farama.org/)

---

### Offline/Batch-RL {#offlinebatch-rl}
Leer van vaste datasets zonder interactie met de omgeving.

**Begrippen:**
- Probleem met distributieverschuiving
- Conservatief Q-Learning (CQL)
- Impliciete Q-Learning (IQL)
- Beslissingstransformator

**Praktisch werk:**
- [ ] [Trainen op D4RL-benchmarkgegevenssets](advanced_topics/offline_rl/d4rl_dataset_explained.md)
- [ ] [CQL implementeren](advanced_topics/offline_rl/cql_explained.md)
- [ ] [Vergelijken met gedragsmatig klonen](advanced_topics/offline_rl/behavioral_cloning_explained.md)

**Bronnen:**
- 📄 [Offline RL-zelfstudie](https://arxiv.org/abs/2005.01643)
- 📄 [CQL-papier](https://arxiv.org/abs/2006.04779)
- 📄 [Beslissingstransformator](https://arxiv.org/abs/2106.01345)
- 💻 [D4RL Benchmark](https://github.com/Farama-Foundation/D4RL)

---

### Verkenning {#exploration}
Pak schaarse belonings- en moeilijke verkenningsproblemen aan.

**Begrippen:**
- Intrinsieke motivatie
- Nieuwsgierigheidsgedreven verkenning (ICM)
- Willekeurige netwerkdestillatie (RND)
- Op tellingen gebaseerde verkenning
- Ga verkennen

**Praktisch werk:**
- [ ] [nieuwsgierigheidsbonus implementeren](advanced_topics/exploration/curiosity_bonus_explained.md)
- [ ] [Train op Montezuma's wraak](advanced_topics/exploration/montezuma_revenge_explained.md)
- [ ] [Verkenningsstrategieën vergelijken](advanced_topics/exploration/compare_exploration_explained.md)

**Bronnen:**
- 📄 [ICM-papier](https://arxiv.org/abs/1705.05363)
- 📄 [RND-papier](https://arxiv.org/abs/1810.12894)
- 📄 [Ga verkennen](https://arxiv.org/abs/1901.10995)
- 📄 [Deep Exploration via Bootstrapped DQN](https://arxiv.org/abs/1602.04621) — bron van de DeepSea-taak

---

### Hiërarchische RL {#hierarchical-rl}
Leer op meerdere niveaus van temporele abstractie.

**Begrippen:**
- Optiekader
- Doelgericht beleid
- Feodale netwerken
-HIRO, HAM

**Praktisch werk:**
- [ ] [Optiekritische architectuur implementeren](advanced_topics/hierarchical_rl/option_critic_explained.md)
- [ ] [Train doelgericht beleid](advanced_topics/hierarchical_rl/goal_conditioned_policy_explained.md)
- [ ] [Test op taken met een lange horizon](advanced_topics/hierarchical_rl/long_horizon_tasks_explained.md)

**Bronnen:**
- 📖 Sutton & Barto - Hoofdstuk 12 (Opties)
- 📄 [Option-Critic Paper](https://arxiv.org/abs/1609.05140)
- 📄 [HIRO-papier](https://arxiv.org/abs/1805.08296)

---

### RLHF (RL van menselijke feedback) {#rlhf-rl-from-human-feedback}
Stem modellen af op menselijke voorkeuren.

**Begrippen:**
- Beloningsmodellering op basis van vergelijkingen
- KL-beperkte beleidsoptimalisatie
- Constitutionele AI
- DPO (Directe Preferentie Optimalisatie)

**Praktisch werk:**
- [ ] [Train een beloningsmodel op basis van voorkeursgegevens](advanced_topics/rlhf/reward_modeling_explained.md)
- [ ] [Een kleine LM verfijnen met PPO](advanced_topics/rlhf/ppo_fine_tuning_explained.md)
- [ ] [DPO implementeren](advanced_topics/rlhf/dpo_implementation_explained.md)

**Bronnen:**
- 📄 [InstructGPT Paper](https://arxiv.org/abs/2203.02155)
- 📄 [Constitutionele AI](https://arxiv.org/abs/2212.08073)
- 📄 [DPO-papier](https://arxiv.org/abs/2305.18290)
- 💻 [TRL-bibliotheek](https://github.com/huggingface/trl)

---

## Fase 6: Onderzoeksniveau (lopend) {#phase-6-research-level-ongoing}

### Doel
Draag origineel werk bij aan het veld.

### Activiteiten
- Lees regelmatig artikelen van NeurIPS, ICML, ICLR
- Reproduceer de belangrijkste resultaten uit recente artikelen
- Identificeer open problemen en beperkingen
- Voer rigoureuze experimenten uit met de juiste evaluatie
- Denk aan monsterefficiëntie, generalisatie en veiligheid

### Landmark Papers om te studeren
- [ ] [AlphaGo (Silver et al., 2016)](https://www.nature.com/articles/nature16961)
- [ ] [AlphaZero (Silver et al., 2018)](https://www.science.org/doi/10.1126/science.aar6404)
- [ ] [MuZero (Schrittwieser et al., 2020)](https://arxiv.org/abs/1911.08265)
- [ ] [Decision Transformer (Chen et al., 2021)](https://arxiv.org/abs/2106.01345)
- [ ] [InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)
- [ ] [DreamerV3 (Hafner et al., 2023)](https://arxiv.org/abs/2301.04104)

### Waar te publiceren
- NeurIPS, ICML, ICLR (toplocaties)
- AAAI, IJCAI
- CoRL (focus op robotica)
- Workshoppapieren voor vroeg werk

---

## Hulpmiddelen Samenvatting

| Phase | Recommended Tools |
|-------|-------------------|
| 1-2 | NumPy, Matplotlib |
| 3 | PyTorch, Gymnasium, ALE-Py, W&B |
| 4 | Stabiel-Baselines3, MuJoCo/Box2D |
| 5-6 | Aangepaste code, Ray/RLlib, JAX (voor snelheid) |

---

## Tips voor succes

1. **Eerst helemaal opnieuw implementeren** — Gebruik bibliotheken pas nadat u het algoritme begrijpt
2. **Debuggen met eenvoudige omgevingen** — CartPole altijd vóór Atari
3. **Alles registreren** — Beloningen, verliezen, hellingen, lengte van afleveringen
4. **Leren visualiseren** — Leercurven uitzetten, afleveringen weergeven
5. **Lees het Sutton & Barto-boek** — Het is de bijbel van RL
6. **Begrijp de wiskunde** — Tenminste de beleidsgradiëntstelling en Bellman-vergelijkingen
7. **Wees geduldig** — RL is notoir onstabiel; mislukte runs zijn normaal
8. **Gebruik zaden** — Reproduceerbaarheid is belangrijk; gemiddeld over meerdere zaden
9. **Word lid van communities** — r/reinforcementlearning, RL Discord, Twitter/X

---

## Veelvoorkomende valkuilen die u moet vermijden

- ❌ Basisprincipes overslaan om in diepe RL te springen
- ❌ Observaties/beloningen niet normaliseren
- ❌ Te grote/kleine leersnelheden gebruiken
- ❌ Vergeten de evaluatiemodus in te stellen tijdens het testen
- ❌ Niet genoeg zaden gebruiken voor experimenten
- ❌ Implementeren vanuit papieren zonder de referentiecode te controleren
- ❌Opgeven na één mislukte trainingsloop

---

## Glossarium

| Term | Definition |
|------|------------|
| **MDP** | Markov-beslissingsproces - formeel raamwerk voor RL |
| **Beleid (π)** | In kaart brengen van toestanden naar acties |
| **Waardefunctie (V)** | Verwacht rendement van een staat |
| **Q-functie** | Verwacht rendement van een staat-actiepaar |
| **TD-fout** | Verschil tussen voorspelde en bootstrapped-waarde |
| **GAE** | Gegeneraliseerde voordeelschatting |
| **PPO** | Proximale beleidsoptimalisatie |
| **RLHF** | Versterking van het leren van menselijke feedback |

---

## Licentie

Deze handleiding is bedoeld voor educatieve doeleinden. Voel je vrij om te delen en aan te passen.

---
