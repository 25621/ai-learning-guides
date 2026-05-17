# Vahvistusoppiminen: aloittelijasta edistyneeseen

Kattava opas oppimisjärjestelmien ymmärtämiseen ja rakentamiseen peruskäsitteistä tutkimustason ymmärtämiseen.

---

## Yleiskatsaus

| Vaihe | Keskity | Kesto |
|-------|-------|----------|
| 1 | [Perusteet](#phase-1-foundations-2-4-weeks) | 2-4 viikkoa |
| 2 | [Taulukkomenetelmät](#phase-2-tabular-methods-3-4-weeks) | 3-4 viikkoa |
| 3 | [Funktion approksimointi](#phase-3-function-approximation-3-4-weeks) | 3-4 viikkoa |
| 4 | [Käytäntögradientit](#phase-4-policy-gradient-methods-4-5-weeks) | 4-5 viikkoa |
| 5 | [Edistyneet aiheet](#phase-5-advanced-topics-6-8-weeks) | 6-8 viikkoa |
| 6 | [Tutkimustaso](#phase-6-research-level-ongoing) | Jatkuva |

**Kokonaisaika pätevyyteen:** ~6 kuukautta

---

## Vaihe 1: Perusteet (2-4 viikkoa)

### Tavoite
Ymmärrä ydinkäsitteet ilman syvällistä matematiikkaa.

### Opittavia käsitteitä
- Agentti, ympäristö, tila, toiminta, palkinto
- Markovin päätösprosessit (MDP)
- Tuotto- ja alennuskerroin (γ)
- Käytäntö vs arvo -funktio
- Tutkimus vs hyödyntäminen kompromissi

### Käytännön työ
- [ ] [Toteuta yksinkertainen monikätinen rosvo-ongelma tyhjästä](foundations/multi_armed_bandit_explained.md)
- [ ] [Ratkaise Frozen Lake satunnaisella käytännöllä, tarkkaile tuloksia](foundations/frozen_lake_explained.md)
- [ ] [Visualisoi tila-arvofunktiot yksinkertaisessa ruudukossa](foundations/state_value_visualization_explained.md)

### Työkalut
- Python
- NumPy
- Matplotlib (visualisointia varten)

### Resurssit
- 📖 [Sutton & Barto](http://incompleteideas.net/book/the-book.html) - Luvut 1-3
- 🎥 [David Silverin luennot](https://www.youtube.com/watch?v=2pWv7GOvuf0) - Luennot 1-2
- 📝 [OpenAI Spinning Up - Johdanto](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)

### Virstanpylväs
Sinun pitäisi pystyä selittämään RL-ongelman muotoilu ja miksi MDP:t ovat vakiokehys.

---

## Vaihe 2: Taulukkomenetelmät (3–4 viikkoa)

### Tavoite
Hallitse klassisia RL-algoritmeja, joissa tila-/toiminta-avaruudet ovat tarpeeksi pieniä tallennettavaksi taulukoihin.

### Opittavia käsitteitä
- Dynaaminen ohjelmointi
  - Käytännön arviointi
  - Käytännön iteraatio
  - Arvo iterointi
- Monte Carlon menetelmät
  - Ensimmäinen käynti vs. joka käynti MC
  - MC-ohjaus ja tutkimuskäynnit
- Temporal Difference Learning
  - TD(0)-ennuste
  - SARSA (käytäntöön perustuva (on-policy) TD-hallinta)
  - Q-learning (käytännöstä riippumaton (off-policy) TD-hallinta)
- Kelpoisuusjäljet ja TD(λ)

### Käytännön työ
- [ ] [Toteuta käytäntöiteraatio GridWorldille](tabular_methods/policy_iteration_gridworld_explained.md)
- [ ] [Rakenna Q-oppimisagentti Frozen Lakelle](tabular_methods/q_learning_frozen_lake_explained.md)
- [ ] [Ota SARSA käyttöön Cliff Walkingissa](tabular_methods/sarsa_cliff_walking_explained.md)
- [ ] [Vertaa SARSA- ja Q-oppimiskäyttäytymistä (turvalliset vs. optimaaliset polut)](tabular_methods/sarsa_vs_qlearning_explained.md)
- [ ] [Ota käyttöön Monte Carlo -hallinta Blackjackille](tabular_methods/monte_carlo_blackjack_explained.md)

### Keskeinen oivallus
Ymmärrä varmuuskopiokaaviot – ne selventävät, kuinka kukin algoritmi päivittää arvoja.

### Työkalut
- NumPy
- Gymnasium (ympäristöihin)

### Resurssit
- 📖 Sutton & Barto - Luvut 4-7
- 🎥 David Silver Luennot 3-5
- 💻 [Denny Britzin RL-varasto](https://github.com/dennybritz/reinforcement-learning)

### Virstanpylväs
Ota Q-oppiminen käyttöön tyhjästä ja ratkaise Frozen Lake >70 % onnistumisprosentilla.

---

## Vaihe 3: Funktion approksimointi (3-4 viikkoa)

### Tavoite
Skaalaa RL pienten tilaavaruuksien yli käyttämällä funktioapproksimaattoreita.

### Opittavia käsitteitä
- Miksi taulukkomenetelmät epäonnistuvat mittakaavassa
- Lineaarinen funktion approksimaatio
- Neuroverkot funktion approksimaattoreina
- Deep Q-Networks (DQN)
  - Koe toisto
  - Kohdeverkostot
  - Palkkion leikkaaminen
- DQN-parannukset
  - Double DQN
  - Dueling DQN
  - Priorisoitu kokemustoisto
  - Rainbow DQN

### Käytännön työ
- [ ] [Ratkaise CartPole lineaarisella Q-oppimisella](function_approximation/linear_q_cartpole_explained.md)
- [ ] [Toteuta DQN tyhjästä CartPolelle](function_approximation/dqn_cartpole_explained.md)
- [ ] [Lisää kokemuksen toisto, tarkkaile vakauden paranemista](function_approximation/dqn_experience_replay_explained.md)
- [ ] [Lisää kohdeverkko, vertaa oppimiskäyriä](function_approximation/dqn_target_network_explained.md)
- [ ] [Harjoittele DQN:ää Atari Pongilla (käytä ALE-Py:tä)](function_approximation/dqn_atari_pong_explained.md)
- [ ] [Toteuta Double DQN, vertaa vanilja DQN:ään](function_approximation/double_dqn_cartpole_explained.md)

### Keskeinen oivallus
"Tappava kolmikko" (funktion approksimaatio + bootstrapping + off-policy) aiheuttaa epävakautta. DQN-innovaatiot ratkaisevat tämän.

### Työkalut
- PyTorch tai TensorFlow
- Gymnasium
- ALE-Py (Atarille)
- Weights & Biases (kokeilun seurantaan)

### Resurssit
- 📖 Sutton & Barto - Luvut 9-11
- 📄 [DQN-paperi (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)
- 📄 [Rainbow-paperi](https://arxiv.org/abs/1710.02298)
- 💻 [CleanRL-toteutukset](https://github.com/vwxyzjn/cleanrl)

### Virstanpylväs
Kouluta DQN-agentti, joka saa positiivisen palkinnon Atari Pongissa.

---

## Vaihe 4: Käytäntögradientit (4–5 viikkoa)

### Tavoite
Opi optimoimaan käytännöt suoraan ilman arvofunktioiden laskemista.

### Opittavia käsitteitä
- Käytännön gradienttilause
- REINFORCE-algoritmi
- Varianssin vähentämistekniikat
  - Perustasot
  - Palkinto matkaan
- Toimija-arvioijan menetelmät
  - A2C (toimija-arvioija)
  - A3C (asynkroninen variantti)
- yleinen etuarvio (GAE)
- Luottamusalueen menetelmät
  - TRPO (Trust Region Policy Optimization)
  - PPO (Proximal Policy Optimization)

### Käytännön työ
- [ ] [Toteuta REINFORCE CartPolelle](policy_gradients/reinforce_cartpole_explained.md)
- [ ] [Lisää perusviiva, mittaa varianssin pienenemistä](policy_gradients/reinforce_baseline_explained.md)
- [ ] [Rakenna A2C LunarLanderille](policy_gradients/a2c_lunarlander_explained.md)
- [ ] [Ota PPO käyttöön tyhjästä](policy_gradients/ppo_scratch_explained.md)
- [ ] [Harjoittele PPO:ta jatkuvalla ohjauksella (BipedalWalker tai MuJoCo)](policy_gradients/ppo_continuous_explained.md)
- [ ] [Vertaa PPO-hyperparametrin herkkyyttä](policy_gradients/ppo_hyperparams_explained.md)

### Keskeinen oivallus
PPO on nykyaikaisen RL:n työhevonen – ymmärrä sen leikkausmekanismi syvästi.

### Työkalut
- PyTorch
- Gymnasium
- Stable-Baselines3 (viite)
- MuJoCo tai Box2D (jatkuvaan ohjaukseen)

### Resurssit
- 📖 Sutton & Barto - Luku 13
- 📝 [OpenAI Spinning Up - Policy Gradients](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
- 📄 [PPO-paperi (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- 📄 [GAE-paperi](https://arxiv.org/abs/1506.02438)
- 🎥 [Pieter Abbeelin Deep RL Bootcamp](https://sites.google.com/view/deep-rl-bootcamp/lectures)

### Virstanpylväs
Ota PPO käyttöön ja ratkaise BipedalWalker-v3 (palkinto > 300).

---

## Vaihe 5: Edistyneet aiheet (6–8 viikkoa)

Valitse 2-3 aluetta kiinnostuksen kohteidesi perusteella.
- [Mallipohjainen RL](#model-based-rl)
- [Monen agentin RL](#multi-agent-rl)
- [Offline/Batch RL](#offlinebatch-rl)
- [Eksploraatio](#exploration)
- [Hierarkkinen RL](#hierarchical-rl)
- [RLHF (RL ihmispalautteesta)](#rlhf-rl-from-human-feedback)

### Mallipohjainen RL
Opi ympäristön dynamiikkaa suunnittelemaan tai luomaan synteettistä kokemusta.

**Konseptit:**
- Dyna-arkkitehtuuri
- Maailmanmallit
- Mallin ennakoiva ohjaus (MPC)
- MuZero, Dreamer

**Käytännön työ:**
- [ ] [Ota Dyna-Q käyttöön](advanced_topics/model_based_rl/dyna_q_explained.md)
- [ ] [Kouluta maailmanmalli yksinkertaisessa ympäristössä](advanced_topics/model_based_rl/world_model_explained.md)
- [ ] [Käytä opittua mallia suunnittelussa](advanced_topics/model_based_rl/model_based_planning_explained.md)

**Resurssit:**
- 📖 Sutton & Barto - Luku 8
- 📄 [Maailman mallien paperi](https://arxiv.org/abs/1803.10122)
- 📄 [MuZero paperi](https://arxiv.org/abs/1911.08265)
- 📄 [DreamerV3 paperi](https://arxiv.org/abs/2301.04104)

---

### Monen agentin RL
Useat agentit oppivat samanaikaisesti jaetuissa ympäristöissä.

**Konseptit:**
- Itsenäiset oppijat
- Keskitetty koulutus, hajautettu toteutus (CTDE)
- Itseleikkiä
- Hätäviestintä

**Käytännön työ:**
- [ ] [Kouluta agentteja yksinkertaisissa matriisipeleissä](advanced_topics/multi_agent_rl/matrix_games_explained.md)
- [ ] [Toteuta itsepeli lautapeliä varten](advanced_topics/multi_agent_rl/self_play_tic_tac_toe_explained.md)
- [ ] [Tutustu PettingZoo-ympäristöihin](advanced_topics/multi_agent_rl/pettingzoo_explore_explained.md)

**Resurssit:**
- 📄 [Multi-Agent RL Survey](https://arxiv.org/abs/1911.10635)
- 📄 [OpenAI Five](https://arxiv.org/abs/1912.06680)
- 💻 [PettingZoon kirjasto](https://pettingzoo.farama.org/)

---

### Offline/Batch RL
Opi kiinteistä tietojoukoista ilman ympäristövuorovaikutusta.

**Konseptit:**
- Jakeluvaihteen ongelma
- Konservatiivinen Q-Learning (CQL)
- Implisiittinen Q-oppiminen (IQL)
- Päätöksenmuuntaja

**Käytännön työ:**
- [ ] [Harjoittele D4RL-vertailutietojoukoissa](advanced_topics/offline_rl/d4rl_dataset_explained.md)
- [ ] [Ota käyttöön CQL](advanced_topics/offline_rl/cql_explained.md)
- [ ] [Vertaa käyttäytymiskloonaukseen](advanced_topics/offline_rl/behavioral_cloning_explained.md)

**Resurssit:**
- 📄 [Offline RL -opetusohjelma](https://arxiv.org/abs/2005.01643)
- 📄 [CQL-paperi](https://arxiv.org/abs/2006.04779)
- 📄 [Päätöksen muuntaja](https://arxiv.org/abs/2106.01345)
- 💻 [D4RL-vertailu](https://github.com/Farama-Foundation/D4RL)

---

### Eksploraatio
Käsittele niukkoja palkkioita ja kovia etsintäongelmia.

**Konseptit:**
- Sisäinen motivaatio
- Uteliaisuusvetoinen eksploraatio (ICM)
- Random Network Distillation (RND)
- Laskuriin perustuva eksploraatio
- Go-Explore

**Käytännön työ:**
- [ ] [Ota käyttöön uteliaisuusbonus](advanced_topics/exploration/curiosity_bonus_explained.md)
- [ ] [Harjoittele Montezuman kostoa](advanced_topics/exploration/montezuma_revenge_explained.md)
- [ ] [Vertaa etsintästrategioita](advanced_topics/exploration/compare_exploration_explained.md)

**Resurssit:**
- 📄 [ICM-paperi](https://arxiv.org/abs/1705.05363)
- 📄 [RND-paperi](https://arxiv.org/abs/1810.12894)
- 📄 [Go-Explore](https://arxiv.org/abs/1901.10995)
- 📄 [Deep Exploration Bootstrapped DQN:n kautta](https://arxiv.org/abs/1602.04621) — DeepSea-tehtävän lähde

---

### Hierarkkinen RL
Opi useilla temporaalisen abstraktion tasoilla.

**Konseptit:**
- Optio-kehys
- Tavoitteisiin perustuvat käytännöt
- Feodaaliverkot
- HIRO, HAM

**Käytännön työ:**
- [ ] [Toteuta vaihtoehtokriittinen arkkitehtuuri](advanced_topics/hierarchical_rl/option_critic_explained.md)
- [ ] [Harjoittele tavoitteellista käytäntöä](advanced_topics/hierarchical_rl/goal_conditioned_policy_explained.md)
- [ ] [Testaa pitkän horisontin tehtäviä](advanced_topics/hierarchical_rl/long_horizon_tasks_explained.md)

**Resurssit:**
- 📖 Sutton & Barto - Luku 12 (Vaihtoehdot)
- 📄 [Option-Critic Paper](https://arxiv.org/abs/1609.05140)
- 📄 [HIRO-paperi](https://arxiv.org/abs/1805.08296)

---

### RLHF (RL ihmispalautteesta)
Kohdista mallit ihmisten mieltymyksiin.

**Konseptit:**
- Palkitse mallinnus vertailuista
- KL-rajoitettu käytännön optimointi
- Perustuslaillinen AI
- DPO (Direct Preference Optimization)

**Käytännön työ:**
- [ ] [Harjoittele palkkiomalli preferenssitiedoista](advanced_topics/rlhf/reward_modeling_explained.md)
- [ ] [Hienosäädä pieni LM PPO:lla](advanced_topics/rlhf/ppo_fine_tuning_explained.md)
- [ ] [Toteuta DPO](advanced_topics/rlhf/dpo_implementation_explained.md)

**Resurssit:**
- 📄 [InstructGPT Paper](https://arxiv.org/abs/2203.02155)
- 📄 [Perustuslaillinen AI](https://arxiv.org/abs/2212.08073)
- 📄 [DPO-paperi](https://arxiv.org/abs/2305.18290)
- 💻 [TRL-kirjasto](https://github.com/huggingface/trl)

---

## Vaihe 6: Tutkimustaso (jatkuva)

### Tavoite
Anna omaperäistä työtä kentälle.

### Aktiviteetit
- Lue säännöllisesti NeurIPS-, ICML- ja ICLR-artikkeleita
- Toista tärkeimmät tulokset viimeaikaisista papereista
- Tunnista avoimet ongelmat ja rajoitukset
- Suorita tiukkoja kokeita asianmukaisella arvioinnilla
- Harkitse näytteen tehokkuutta, yleistämistä, turvallisuutta

### Maamerkkipaperit opiskeluun
- [ ] [AlphaGo (Silver et al., 2016)](https://www.nature.com/articles/nature16961)
- [ ] [AlphaZero (Silver et al., 2018)](https://www.science.org/doi/10.1126/science.aar6404)
- [ ] [MuZero (Schrittwieser et al., 2020)](https://arxiv.org/abs/1911.08265)
- [ ] [Decision Transformer (Chen et al., 2021)](https://arxiv.org/abs/2106.01345)
- [ ] [InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)
- [ ] [DreamerV3 (Hafner ym., 2023)](https://arxiv.org/abs/2301.04104)

### Missä julkaista
- NeurIPS, ICML, ICLR (huippupaikat)
- AAAI, IJCAI
- CoRL (robotiikan fokus)
- Työpajapaperit varhaiseen työhön

---

## Työkalut yhteenveto

| Vaihe | Suositellut työkalut |
|-------|-------------------|
| 1-2 | NumPy, Matplotlib |
| 3 | PyTorch, Gymnasium, ALE-Py, W&B |
| 4 | Stable-Baselines3, MuJoCo/Box2D |
| 5-6 | Mukautettu koodi, Ray/RLlib, JAX (nopeus) |

---

## Vinkkejä menestykseen

1. **Ota käyttöön tyhjästä ensin** — Käytä kirjastoja vasta, kun olet ymmärtänyt algoritmin
2. **Virheenkorjaus yksinkertaisissa ympäristöissä** — CartPole ennen Ataria, aina
3. **Kirjaa kaikki lokiin** — Palkinnot, tappiot, kaltevuus, jaksojen pituudet
4. **Visualisoi oppiminen** — Piirrä oppimiskäyrät, renderöi jaksot
5. **Lue Sutton & Barto -kirja** — Se on RL:n raamattu
6. **Ymmärrä matematiikka** — ainakin käytännön gradienttilause ja Bellman-yhtälöt
7. **Ole kärsivällinen** — RL on tunnetusti epävakaa; epäonnistuneet ajot ovat normaaleja
8. **Käytä siemeniä** — Uusittavuus on tärkeää; useiden siementen keskiarvo
9. **Liity yhteisöihin** — r/reinforcementlearning, RL Discord, Twitter/X

---

## Yleisiä sudenkuoppia vältettävänä

- ❌ Perusasiat väliin hypätäksesi syvään RL:ään
- ❌ Ei normalisoi havaintoja/palkintoja
- ❌ Liian suuria/pieniä oppimisnopeuksia käytetään
- ❌ Unohdin asettaa arviointitilan testauksen aikana
- ❌ Ei käytetä tarpeeksi siemeniä kokeisiin
- ❌ Toteutus papereista tarkistamatta viitekoodia
- ❌ Luopuminen yhden epäonnistuneen harjoituslenkin jälkeen

---

## Sanasto

| Termi | Määritelmä |
|------|------------|
| **MDP** | Markovin päätösprosessi - muodollinen kehys RL:lle |
| **Käytäntö (π)** | Kartoitus tiloista tekoihin |
| **Arvofunktio (V)** | Odotettu paluu osavaltiosta |
| **Q-toiminto** | Odotettu tuotto tila-toiminto-parista |
| **TD-virhe** | Ennustetun ja bootstrapped-arvon välinen ero |
| **GAE** | Yleistetty etuarvio |
| **PPO** | Proksimaalinen käytännön optimointi |
| **RLHF** | Ihmisten palautteesta oppimisen vahvistaminen |

---

## Lisenssi

Tämä opas on tarkoitettu koulutustarkoituksiin. Voit vapaasti jakaa ja mukauttaa.

---
