# रिएनफोर्समेंट लर्निंग: शुरुआती से उन्नत तक (Reinforcement Learning: From Beginner to Advanced)

मौलिक अवधारणाओं से लेकर शोध-स्तर की समझ तक, रिएनफोर्समेंट लर्निंग में महारत हासिल करने के लिए एक संरचित मार्गदर्शिका।

---

## अवलोकन (Overview)

| चरण (Phase) | फोकस (Focus) | अवधि (Duration) |
|-------|-------|----------|
| 1 | [बुनियाद (Foundations)](#चरण-1-बुनियाद-2-4-सप्ताह) | 2-4 सप्ताह |
| 2 | [सारणीबद्ध तरीके (Tabular Methods)](#चरण-2-सारणीबद्ध-तरीके-3-4-सप्ताह) | 3-4 सप्ताह |
| 3 | [फंक्शन एप्रोक्सीमेशन (Function Approximation)](#चरण-3-फंक्शन-एप्रोक्सीमेशन-3-4-सप्ताह) | 3-4 सप्ताह |
| 4 | [पॉलिसी ग्रेडिएंट तरीके (Policy Gradient Methods)](#चरण-4-पॉलिसी-ग्रेडिएंट-तरीके-4-5-सप्ताह) | 4-5 सप्ताह |
| 5 | [उन्नत विषय (Advanced Topics)](#चरण-5-उन्नत-विषय-6-8-सप्ताह) | 6-8 सप्ताह |
| 6 | [शोध-स्तर (Research-Level)](#चरण-6-शोध-स्तर-निरंतर) | निरंतर |

**प्रवीणता हासिल करने का कुल समय:** ~6 महीने

---

## चरण 1: बुनियाद (2-4 सप्ताह) (Phase 1: Foundations)

### लक्ष्य (Goal)
गहन गणित के बिना मुख्य अवधारणाओं को समझना।

### सीखने योग्य अवधारणाएं (Concepts to Learn)
- एजेंट (Agent), वातावरण (environment), स्थिति (state), क्रिया (action), इनाम (reward)
- मार्कोव डिसीजन प्रोसेस (Markov Decision Processes - MDP)
- रिटर्न (Return) और डिस्काउंट फैक्टर (γ)
- पॉलिसी (Policy) बनाम वैल्यू फंक्शन (value function)
- एक्सप्लोरेशन (Exploration) बनाम एक्सप्लोइटेशन (exploitation) ट्रेडऑफ

### व्यावहारिक कार्य (Practical Work)
- [ ] [शुरुआत से एक सरल मल्टी-आर्म्ड बैंडिट समस्या लागू करें](foundations/multi_armed_bandit_explained.md)
- [ ] [रैंडम पॉलिसी के साथ फ्रोजन लेक को हल करें, परिणाम देखें](foundations/frozen_lake_explained.md)
- [ ] [एक सरल ग्रिड पर स्टेट-वैल्यू फंक्शन की कल्पना (Visualize) करें](foundations/state_value_visualization_explained.md)

### उपकरण (Tools)
- Python
- NumPy
- Matplotlib (विजुअलाइजेशन के लिए)

### संसाधन (Resources)
- 📖 [Sutton & Barto](http://incompleteideas.net/book/the-book.html) - अध्याय 1-3
- 🎥 [David Silver Lectures](https://www.youtube.com/watch?v=2pWv7GOvuf0) - व्याख्यान 1-2
- 📝 [OpenAI Spinning Up - परिचय](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)

### माइलस्टोन (Milestone)
आपको RL समस्या के निर्माण और यह समझाने में सक्षम होना चाहिए कि MDP मानक ढांचा (standard framework) क्यों है।

---

## चरण 2: सारणीबद्ध तरीके (3-4 सप्ताह) (Phase 2: Tabular Methods)

### लक्ष्य (Goal)
उन शास्त्रीय RL एल्गोरिदम में महारत हासिल करना जहाँ स्थिति/क्रिया रिक्त स्थान (state/action spaces) तालिकाओं में संग्रहीत करने के लिए पर्याप्त छोटे हैं।

### सीखने योग्य अवधारणाएं (Concepts to Learn)
- डायनेमिक प्रोग्रामिंग (Dynamic Programming)
  - पॉलिसी इवैल्यूएशन (Policy Evaluation)
  - पॉलिसी इटरेशन (Policy Iteration)
  - वैल्यू इटरेशन (Value Iteration)
- मोंटे कार्लो तरीके (Monte Carlo Methods)
  - First-visit बनाम every-visit MC
  - एक्सप्लोरिंग स्टार्ट्स के साथ MC कंट्रोल
- टेम्पोरल डिफरेंस लर्निंग (Temporal Difference Learning)
  - TD(0) प्रेडिक्शन
  - SARSA (ऑन-पॉलिसी TD कंट्रोल)
  - Q-learning (ऑफ-पॉलिसी TD कंट्रोल)
- एलिजिबिलिटी ट्रेसेस (Eligibility Traces) और TD(λ)

### व्यावहारिक कार्य (Practical Work)
- [ ] [ग्रिडवर्ल्ड (GridWorld) के लिए पॉलिसी इटरेशन लागू करें](tabular_methods/policy_iteration_gridworld_explained.md)
- [ ] [फ्रोजन लेक (Frozen Lake) के लिए Q-learning एजेंट बनाएं](tabular_methods/q_learning_frozen_lake_explained.md)
- [ ] [क्लिफ वॉकिंग (Cliff Walking) के लिए SARSA लागू करें](tabular_methods/sarsa_cliff_walking_explained.md)
- [ ] [SARSA बनाम Q-learning व्यवहार की तुलना करें (सुरक्षित बनाम इष्टतम पथ)](tabular_methods/sarsa_vs_qlearning_explained.md)
- [ ] [ब्लैकजैक (Blackjack) के लिए मोंटे कार्लो कंट्रोल लागू करें](tabular_methods/monte_carlo_blackjack_explained.md)

### मुख्य अंतर्दृष्टि (Key Insight)
बैकअप आरेखों (backup diagrams) को समझें—वे स्पष्ट करते हैं कि प्रत्येक एल्गोरिदम वैल्यू को कैसे अपडेट करता है।

### उपकरण (Tools)
- NumPy
- Gymnasium (वातावरण के लिए)

### संसाधन (Resources)
- 📖 Sutton & Barto - अध्याय 4-7
- 🎥 David Silver व्याख्यान 3-5
- 💻 [Denny Britz का RL रिपॉजिटरी](https://github.com/dennybritz/reinforcement-learning)

### माइलस्टोन (Milestone)
शुरुआत से Q-learning लागू करें और >70% सफलता दर के साथ फ्रोजन लेक को हल करें।

---

## चरण 3: फंक्शन एप्रोक्सीमेशन (3-4 सप्ताह) (Phase 3: Function Approximation)

### लक्ष्य (Goal)
फंक्शन एप्रोक्सीमेटर्स का उपयोग करके RL को छोटे स्टेट स्पेस से आगे बढ़ाना।

### सीखने योग्य अवधारणाएं (Concepts to Learn)
- बड़े पैमाने पर सारणीबद्ध तरीके क्यों विफल होते हैं
- लीनियर फंक्शन एप्रोक्सीमेशन (Linear function approximation)
- फंक्शन एप्रोक्सीमेटर्स के रूप में न्यूरल नेटवर्क
- डीप Q-नेटवर्क (Deep Q-Networks - DQN)
  - एक्सपीरियंस रिप्ले (Experience replay)
  - टारगेट नेटवर्क (Target networks)
  - रिवॉर्ड क्लिपिंग (Reward clipping)
- DQN सुधार
  - Double DQN
  - Dueling DQN
  - Prioritized Experience Replay
  - Rainbow DQN

### व्यावहारिक कार्य (Practical Work)
- [ ] [लीनियर Q-learning के साथ कार्टपोल (CartPole) को हल करें](function_approximation/linear_q_cartpole_explained.md)
- [ ] [कार्टपोल के लिए शुरुआत से DQN लागू करें](function_approximation/dqn_cartpole_explained.md)
- [ ] [अनुभव रिप्ले जोड़ें, स्थिरता में सुधार देखें](function_approximation/dqn_experience_replay_explained.md)
- [ ] [टार्गेट नेटवर्क जोड़ें, लर्निंग कर्व की तुलना करें](function_approximation/dqn_target_network_explained.md)
- [ ] [Atari Pong पर DQN प्रशिक्षित करें (ALE-Py का उपयोग करें)](function_approximation/dqn_atari_pong_explained.md)
- [ ] [Double DQN लागू करें, वेनिला DQN के साथ तुलना करें](function_approximation/double_dqn_cartpole_explained.md)

### मुख्य अंतर्दृष्टि (Key Insight)
"डेडली ट्रायड" (फंक्शन एप्रोक्सीमेशन + बूटस्ट्रैपिंग + ऑफ-पॉलिसी) अस्थिरता का कारण बनता है। DQN नवाचार इसे संबोधित करते हैं।

### उपकरण (Tools)
- PyTorch या TensorFlow
- Gymnasium
- ALE-Py (Atari के लिए)
- Weights & Biases (प्रयोग ट्रैकिंग के लिए)

### संसाधन (Resources)
- 📖 Sutton & Barto - अध्याय 9-11
- 📄 [DQN पेपर (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)
- 📄 [Rainbow पेपर](https://arxiv.org/abs/1710.02298)
- 💻 [CleanRL कार्यान्वयन](https://github.com/vwxyzjn/cleanrl)

### माइलस्टोन (Milestone)
एक DQN एजेंट को प्रशिक्षित करें जो Atari Pong पर सकारात्मक इनाम प्राप्त करता है।

---

## चरण 4: पॉलिसी ग्रेडिएंट तरीके (4-5 सप्ताह) (Phase 4: Policy Gradient Methods)

### लक्ष्य (Goal)
वैल्यू फंक्शन की गणना किए बिना सीधे पॉलिसियों को अनुकूलित (optimize) करना सीखें।

### सीखने योग्य अवधारणाएं (Concepts to Learn)
- पॉलिसी ग्रेडिएंट थ्योरम (Policy Gradient Theorem)
- REINFORCE एल्गोरिदम
- वेरिएंस रिडक्शन (Variance reduction) तकनीक
  - बेसलाइन (Baselines)
  - रिवॉर्ड-टू-गो (Reward-to-go)
- एक्टर-क्रिटिक तरीके (Actor-Critic Methods)
  - A2C (Advantage Actor-Critic)
  - A3C (Asynchronous variant)
- जनरलाइज्ड एडवांटेज एस्टीमेशन (Generalized Advantage Estimation - GAE)
- ट्रस्ट रीजन तरीके (Trust Region Methods)
  - TRPO (Trust Region Policy Optimization)
  - PPO (Proximal Policy Optimization)

### व्यावहारिक कार्य (Practical Work)
- [ ] [कार्टपोल के लिए REINFORCE लागू करें](policy_gradients/reinforce_cartpole_explained.md)
- [ ] [बेसलाइन जोड़ें, वेरिएंस रिडक्शन को मापें](policy_gradients/reinforce_baseline_explained.md)
- [ ] [लूनरलैंडर (LunarLander) के लिए A2C बनाएं](policy_gradients/a2c_lunarlander_explained.md)
- [ ] [शुरुआत से PPO लागू करें](policy_gradients/ppo_scratch_explained.md)
- [ ] [सतत नियंत्रण (continuous control - BipedalWalker या MuJoCo) पर PPO को प्रशिक्षित करें](policy_gradients/ppo_continuous_explained.md)
- [ ] [PPO हाइपरपैरामीटर संवेदनशीलता की तुलना करें](policy_gradients/ppo_hyperparams_explained.md)

### मुख्य अंतर्दृष्टि (Key Insight)
PPO आधुनिक RL का वर्कहॉर्स है—इसके क्लिपिंग तंत्र (clipping mechanism) को गहराई से समझें।

### उपकरण (Tools)
- PyTorch
- Gymnasium
- Stable-Baselines3 (संदर्भ के लिए)
- MuJoCo या Box2D (सतत नियंत्रण के लिए)

### संसाधन (Resources)
- 📖 Sutton & Barto - अध्याय 13
- 📝 [OpenAI Spinning Up - पॉलिसी ग्रेडिएंट्स](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
- 📄 [PPO पेपर (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- 📄 [GAE पेपर](https://arxiv.org/abs/1506.02438)
- 🎥 [Pieter Abbeel का Deep RL Bootcamp](https://sites.google.com/view/deep-rl-bootcamp/lectures)

### माइलस्टोन (Milestone)
PPO लागू करें और BipedalWalker-v3 (इनाम > 300) को हल करें।

---

## चरण 5: उन्नत विषय (6-8 सप्ताह) (Phase 5: Advanced Topics)

अपनी रुचियों के आधार पर 2-3 क्षेत्र चुनें।
- [मॉडल-आधारित RL (Model-Based RL)](#मॉडल-आधारित-rl)
- [मल्टी-एजेंट RL (Multi-Agent RL)](#मल्टी-एजेंट-rl)
- [ऑफलाइन/बैच RL (Offline/Batch RL)](#ऑफलाइनबैच-rl)
- [खोज (Exploration)](#खोज)
- [पदानुक्रमित RL (Hierarchical RL)](#पदानुक्रमित-rl)
- [RLHF (मानव फीडबैक से RL)](#rlhf-मानव-फीडबैक-से-rl)

### मॉडल-आधारित RL (Model-Based RL)
सिंथेटिक अनुभव की योजना बनाने या उत्पन्न करने के लिए वातावरण की गतिशीलता (dynamics) सीखें।

**अवधारणाएं:**
- Dyna आर्किटेक्चर
- वर्ल्ड मॉडल (World models)
- मॉडल प्रेडिक्टिव कंट्रोल (Model Predictive Control - MPC)
- MuZero, Dreamer

**व्यावहारिक कार्य:**
- [ ] [Dyna-Q लागू करें](advanced_topics/model_based_rl/dyna_q_explained.md)
- [ ] [एक सरल वातावरण पर वर्ल्ड मॉडल को प्रशिक्षित करें](advanced_topics/model_based_rl/world_model_explained.md)
- [ ] [योजना बनाने के लिए सीखे गए मॉडल का उपयोग करें](advanced_topics/model_based_rl/model_based_planning_explained.md)

**संसाधन:**
- 📖 Sutton & Barto - अध्याय 8
- 📄 [वर्ल्ड मॉडल्स पेपर](https://arxiv.org/abs/1803.10122)
- 📄 [MuZero पेपर](https://arxiv.org/abs/1911.08265)
- 📄 [DreamerV3 पेपर](https://arxiv.org/abs/2301.04104)

---

### मल्टी-एजेंट RL (Multi-Agent RL)
साझा वातावरण में एक साथ सीखने वाले कई एजेंट।

**अवधारणाएं:**
- स्वतंत्र शिक्षार्थी (Independent learners)
- सेंट्रलाइज्ड ट्रेनिंग, डिसेंट्रलाइज्ड एग्जीक्यूशन (CTDE)
- सेल्फ-प्ले (Self-play)
- उभरता हुआ संचार (Emergent communication)

**व्यावहारिक कार्य:**
- [ ] [एजेंटों को सरल मैट्रिक्स गेम में प्रशिक्षित करें](advanced_topics/multi_agent_rl/matrix_games_explained.md)
- [ ] [बोर्ड गेम के लिए सेल्फ-प्ले लागू करें](advanced_topics/multi_agent_rl/self_play_tic_tac_toe_explained.md)
- [ ] [PettingZoo वातावरण का अन्वेषण करें](advanced_topics/multi_agent_rl/pettingzoo_explore_explained.md)

**संसाधन:**
- 📄 [मल्टी-एजेंट RL सर्वेक्षण](https://arxiv.org/abs/1911.10635)
- 📄 [OpenAI Five](https://arxiv.org/abs/1912.06680)
- 💻 [PettingZoo लाइब्रेरी](https://pettingzoo.farama.org/)

---

### ऑफलाइन/बैच RL (Offline/Batch RL)
वातावरण की बातचीत के बिना निश्चित डेटासेट से सीखें।

**अवधारणाएं:**
- डिस्ट्रीब्यूशन शिफ्ट की समस्या
- कंजर्वेटिव Q-लर्निंग (Conservative Q-Learning - CQL)
- इंप्लिसिट Q-लर्निंग (Implicit Q-Learning - IQL)
- डिसीजन ट्रांसफार्मर (Decision Transformer)

**व्यावहारिक कार्य:**
- [ ] [D4RL बेंचमार्क डेटासेट पर प्रशिक्षित करें](advanced_topics/offline_rl/d4rl_dataset_explained.md)
- [ ] [CQL लागू करें](advanced_topics/offline_rl/cql_explained.md)
- [ ] [व्यवहार क्लोनिंग (behavioral cloning) के साथ तुलना करें](advanced_topics/offline_rl/behavioral_cloning_explained.md)

**संसाधन:**
- 📄 [ऑफलाइन RL ट्यूटोरियल](https://arxiv.org/abs/2005.01643)
- 📄 [CQL पेपर](https://arxiv.org/abs/2006.04779)
- 📄 [डिसीजन ट्रांसफार्मर](https://arxiv.org/abs/2106.01345)
- 💻 [D4RL बेंचमार्क](https://github.com/Farama-Foundation/D4RL)

---

### खोज (Exploration)
विरल इनाम (sparse reward) और कठिन खोज समस्याओं को संबोधित करें।

**अवधारणाएं:**
- आंतरिक प्रेरणा (Intrinsic motivation)
- जिज्ञासा-संचालित खोज (Curiosity-driven exploration - ICM)
- रैंडम नेटवर्क डिस्टिलेशन (Random Network Distillation - RND)
- काउंट-आधारित खोज
- गो-एक्सप्लोर (Go-Explore)

**व्यावहारिक कार्य:**
- [ ] [जिज्ञासा बोनस (curiosity bonus) लागू करें](advanced_topics/exploration/curiosity_bonus_explained.md)
- [ ] [Montezuma's Revenge पर प्रशिक्षित करें](advanced_topics/exploration/montezuma_revenge_explained.md)
- [ ] [खोज रणनीतियों की तुलना करें](advanced_topics/exploration/compare_exploration_explained.md)

**संसाधन:**
- 📄 [ICM पेपर](https://arxiv.org/abs/1705.05363)
- 📄 [RND पेपर](https://arxiv.org/abs/1810.12894)
- 📄 [Go-Explore](https://arxiv.org/abs/1901.10995)
- 📄 [Deep Exploration via Bootstrapped DQN](https://arxiv.org/abs/1602.04621) — DeepSea कार्य का स्रोत

---

### पदानुक्रमित RL (Hierarchical RL)
लौकिक अमूर्तता (temporal abstraction) के कई स्तरों पर सीखें।

**अवधारणाएं:**
- ऑप्शंस फ्रेमवर्क (Options framework)
- लक्ष्य-आधारित पॉलिसियाँ (Goal-conditioned policies)
- फ्यूडल नेटवर्क (Feudal networks)
- HIRO, HAM

**व्यावहारिक कार्य:**
- [ ] [ऑप्शन-क्रिटिक आर्किटेक्चर लागू करें](advanced_topics/hierarchical_rl/option_critic_explained.md)
- [ ] [लक्ष्य-आधारित पॉलिसी को प्रशिक्षित करें](advanced_topics/hierarchical_rl/goal_conditioned_policy_explained.md)
- [ ] [लंबे समय तक चलने वाले कार्यों पर परीक्षण करें](advanced_topics/hierarchical_rl/long_horizon_tasks_explained.md)

**संसाधन:**
- 📖 Sutton & Barto - अध्याय 12 (विकल्प)
- 📄 [Option-Critic पेपर](https://arxiv.org/abs/1609.05140)
- 📄 [HIRO पेपर](https://arxiv.org/abs/1805.08296)

---

### RLHF (मानव फीडबैक से RL) (RL from Human Feedback)
मॉडल को मानवीय प्राथमिकताओं के साथ संरेखित (align) करें।

**अवधारणाएं:**
- तुलनाओं से रिवॉर्ड मॉडलिंग
- KL-बाधित पॉलिसी अनुकूलन (KL-constrained policy optimization)
- संवैधानिक AI (Constitutional AI)
- DPO (Direct Preference Optimization)

**व्यावहारिक कार्य:**
- [ ] [वरीयता डेटा से एक रिवॉर्ड मॉडल को प्रशिक्षित करें](advanced_topics/rlhf/reward_modeling_explained.md)
- [ ] [PPO के साथ एक छोटे LM को फाइन-ट्यून करें](advanced_topics/rlhf/ppo_fine_tuning_explained.md)
- [ ] [DPO लागू करें](advanced_topics/rlhf/dpo_implementation_explained.md)

**संसाधन:**
- 📄 [InstructGPT पेपर](https://arxiv.org/abs/2203.02155)
- 📄 [संवैधानिक AI](https://arxiv.org/abs/2212.08073)
- 📄 [DPO पेपर](https://arxiv.org/abs/2305.18290)
- 💻 [TRL लाइब्रेरी](https://github.com/huggingface/trl)

---

## चरण 6: शोध-स्तर (निरंतर) (Phase 6: Research-Level)

### लक्ष्य (Goal)
क्षेत्र में मूल कार्य का योगदान करें।

### गतिविधियां (Activities)
- NeurIPS, ICML, ICLR के पेपर नियमित रूप से पढ़ें
- हाल के पेपरों के मुख्य परिणामों को फिर से पेश करें
- खुली समस्याओं और सीमाओं की पहचान करें
- उचित मूल्यांकन के साथ कठोर प्रयोग चलाएं
- नमूना दक्षता (sample efficiency), सामान्यीकरण (generalization), सुरक्षा पर विचार करें

### अध्ययन के लिए ऐतिहासिक पेपर (Landmark Papers to Study)
- [ ] [AlphaGo (Silver et al., 2016)](https://www.nature.com/articles/nature16961)
- [ ] [AlphaZero (Silver et al., 2018)](https://www.science.org/doi/10.1126/science.aar6404)
- [ ] [MuZero (Schrittwieser et al., 2020)](https://arxiv.org/abs/1911.08265)
- [ ] [Decision Transformer (Chen et al., 2021)](https://arxiv.org/abs/2106.01345)
- [ ] [InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)
- [ ] [DreamerV3 (Hafner et al., 2023)](https://arxiv.org/abs/2301.04104)

### कहाँ प्रकाशित करें (Where to Publish)
- NeurIPS, ICML, ICLR (शीर्ष स्थान)
- AAAI, IJCAI
- CoRL (रोबोटिक्स फोकस)
- प्रारंभिक कार्य के लिए कार्यशाला (Workshop) पेपर

---

## उपकरणों का सारांश (Tools Summary)

| चरण (Phase) | अनुशंसित उपकरण (Recommended Tools) |
|-------|-------------------|
| 1-2 | NumPy, Matplotlib |
| 3 | PyTorch, Gymnasium, ALE-Py, W&B |
| 4 | Stable-Baselines3, MuJoCo/Box2D |
| 5-6 | कस्टम कोड, Ray/RLlib, JAX (गति के लिए) |

---

## सफलता के लिए सुझाव (Tips for Success)

1. **पहले शुरुआत से लागू करें** — लाइब्रेरी का उपयोग केवल एल्गोरिदम को समझने के बाद ही करें
2. **सरल वातावरण के साथ डिबग करें** — अटाारी से पहले कार्टपोल, हमेशा
3. **सब कुछ लॉग करें** — इनाम, नुकसान, ग्रेडिएंट, एपिसोड की लंबाई
4. **सीखने की कल्पना करें** — लर्निंग कर्व प्लॉट करें, एपिसोड रेंडर करें
5. **सटन और बार्टो पुस्तक पढ़ें** — यह RL की बाइबिल है
6. **गणित को समझें** — कम से कम पॉलिसी ग्रेडिएंट थ्योरम और बेलमैन समीकरण
7. **धैर्य रखें** — RL कुख्यात रूप से अस्थिर है; विफल रन सामान्य हैं
8. **बीज (Seeds) का उपयोग करें** — प्रतिलिपि प्रस्तुत करने योग्यता (reproducibility) मायने रखती है; कई बीजों पर औसत लें
9. **समुदायों में शामिल हों** — r/reinforcementlearning, RL Discord, Twitter/X

---

## बचने योग्य सामान्य गलतियाँ (Common Pitfalls to Avoid)

- ❌ डीप RL में कूदने के लिए बुनियादी बातों को छोड़ना
- ❌ ऑब्जर्वेशन/रिवॉर्ड को सामान्य (normalize) न करना
- ❌ बहुत बड़ी/छोटी लर्निंग रेट का उपयोग करना
- ❌ परीक्षण के दौरान इवैल्यूएशन मोड सेट करना भूल जाना
- ❌ प्रयोगों के लिए पर्याप्त बीजों का उपयोग न करना
- ❌ संदर्भ कोड की जाँच किए बिना पेपरों से लागू करना
- ❌ एक विफल प्रशिक्षण रन के बाद हार मान लेना

---

## शब्दावली (Glossary)

| शब्द (Term) | परिभाषा (Definition) |
|------|------------|
| **MDP** | मार्कोव डिसीजन प्रोसेस - RL के लिए औपचारिक ढांचा |
| **पॉलिसी (π)** | स्थितियों से क्रियाओं तक मानचित्रण (Mapping) |
| **वैल्यू फंक्शन (V)** | एक स्थिति से अपेक्षित रिटर्न |
| **Q-फंक्शन** | एक स्थिति-क्रिया जोड़ी से अपेक्षित रिटर्न |
| **TD एरर** | अनुमानित और बूटस्ट्रैप्ड वैल्यू के बीच अंतर |
| **GAE** | जनरलाइज्ड एडवांटेज एस्टीमेशन |
| **PPO** | प्रॉक्सिमल पॉलिसी ऑप्टिमाइज़ेशन |
| **RLHF** | मानव फीडबैक से रिएनफोर्समेंट लर्निंग |

---

## लाइसेंस (License)

यह मार्गदर्शिका शैक्षिक उद्देश्यों के लिए प्रदान की गई है। साझा करने और अनुकूलित करने के लिए स्वतंत्र महसूस करें।

---
