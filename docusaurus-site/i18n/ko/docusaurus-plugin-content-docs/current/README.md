# 강화학습: 기초부터 심화까지

기초 개념부터 연구 수준의 이해에 이르기까지 강화학습 시스템을 이해하고 구축하기 위한 포괄적인 가이드입니다.

---

## 개요

| 단계 | 집중 분야 | 소요 기간 |
|-------|-------|----------|
| 1 | [기초 (Foundations)](#phase-1-foundations-2-4-weeks) | 2-4주 |
| 2 | [테이블 방식 (Tabular Methods)](#phase-2-tabular-methods-3-4-weeks) | 3-4주 |
| 3 | [함수 근사 (Function Approximation)](#phase-3-function-approximation-3-4-weeks) | 3-4주 |
| 4 | [정책 그래디언트 방법 (Policy Gradient Methods)](#phase-4-policy-gradient-methods-4-5-weeks) | 4-5주 |
| 5 | [심화 주제 (Advanced Topics)](#phase-5-advanced-topics-6-8-weeks) | 6-8주 |
| 6 | [연구 수준 (Research-Level)](#phase-6-research-level-ongoing) | 진행 중 |

**숙련까지의 총 예상 시간:** 약 6개월

---

## Phase 1: 기초 (2-4주) {#phase-1-foundations-2-4-weeks}

### 목표
수학적 깊이보다는 핵심 개념을 이해합니다.

### 학습 개념
- 에이전트(Agent), 환경(Environment), 상태(State), 행동(Action), 보상(Reward)
- 마르코프 결정 과정 (Markov Decision Processes, MDP)
- 리턴(Return)과 할인 인자(Discount Factor, γ)
- 정책(Policy) vs 가치 함수(Value Function)
- 탐험(Exploration) vs 활용(Exploitation)의 트레이드오프

### 실습 과제
- [ ] [밑바닥부터 간단한 멀티 암드 밴딧 문제 구현하기](foundations/multi_armed_bandit_explained.md)
- [ ] [무작위 정책으로 Frozen Lake 해결하고 결과 관찰하기](foundations/frozen_lake_explained.md)
- [ ] [단순한 그리드에서 상태 가치 함수 시각화하기](foundations/state_value_visualization_explained.md)

### 도구
- Python
- NumPy
- Matplotlib (시각화용)

### 추천 자료
- 📖 [Sutton & Barto](http://incompleteideas.net/book/the-book.html) - Chapters 1-3
- 🎥 [David Silver 강좌](https://www.youtube.com/watch?v=2pWv7GOvuf0) - Lectures 1-2
- 📝 [OpenAI Spinning Up - Introduction](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)

### 이정표 (Milestone)
강화학습 문제의 공식화 과정과 왜 MDP가 표준 프레임워크인지 설명할 수 있어야 합니다.

---

## Phase 2: 테이블 방식 (3-4주) {#phase-2-tabular-methods-3-4-weeks}

### 목표
상태/행동 공간이 테이블에 저장할 수 있을 정도로 작은 고전적인 강화학습 알고리즘을 마스터합니다.

### 학습 개념
- 동적 계획법 (Dynamic Programming)
  - 정책 평가 (Policy Evaluation)
  - 정책 반복 (Policy Iteration)
  - 가치 반복 (Value Iteration)
- 몬테카를로 방법 (Monte Carlo Methods)
  - First-visit vs Every-visit MC
  - Exploring starts를 이용한 MC 제어
- 시간차 학습 (Temporal Difference Learning, TD)
  - TD(0) 예측
  - SARSA (온-폴리시 TD 제어)
  - Q-러닝 (오프-폴리시 TD 제어)
- Eligibility Traces와 TD(λ)

### 실습 과제
- [ ] [GridWorld를 위한 정책 반복 구현하기](tabular_methods/policy_iteration_gridworld_explained.md)
- [ ] [Frozen Lake를 위한 Q-러닝 에이전트 구축하기](tabular_methods/q_learning_frozen_lake_explained.md)
- [ ] [Cliff Walking을 위한 SARSA 구현하기](tabular_methods/sarsa_cliff_walking_explained.md)
- [ ] [SARSA vs Q-러닝의 행동 비교 (안전한 경로 vs 최적 경로)](tabular_methods/sarsa_vs_qlearning_explained.md)
- [ ] [블랙잭을 위한 몬테카를로 제어 구현하기](tabular_methods/monte_carlo_blackjack_explained.md)

### 핵심 통찰
백업 다이어그램(Backup diagrams)을 이해하세요. 각 알고리즘이 가치를 어떻게 업데이트하는지 명확히 해줍니다.

### 도구
- NumPy
- Gymnasium (환경용)

### 추천 자료
- 📖 Sutton & Barto - Chapters 4-7
- 🎥 David Silver 강좌 3-5
- 💻 [Denny Britz의 RL 저장소](https://github.com/dennybritz/reinforcement-learning)

### 이정표
밑바닥부터 Q-러닝을 구현하여 Frozen Lake를 70% 이상의 성공률로 해결합니다.

---

## Phase 3: 함수 근사 (3-4주) {#phase-3-function-approximation-3-4-weeks}

### 목표
함수 근사기(Function Approximator)를 사용하여 작은 상태 공간을 넘어 강화학습을 확장합니다.

### 학습 개념
- 테이블 방식이 큰 규모에서 실패하는 이유
- 선형 함수 근사 (Linear Function Approximation)
- 함수 근사기로서의 신경망
- Deep Q-Networks (DQN)
  - 경험 재현 (Experience Replay)
  - 타겟 네트워크 (Target Networks)
  - 보상 클리핑 (Reward Clipping)
- DQN 개선 사항
  - Double DQN
  - Dueling DQN
  - Prioritized Experience Replay
  - Rainbow DQN

### 실습 과제
- [ ] [선형 Q-러닝으로 CartPole 해결하기](function_approximation/linear_q_cartpole_explained.md)
- [ ] [CartPole을 위한 DQN 밑바닥부터 구현하기](function_approximation/dqn_cartpole_explained.md)
- [ ] [경험 재현 추가 및 안정성 향상 관찰하기](function_approximation/dqn_experience_replay_explained.md)
- [ ] [타겟 네트워크 추가 및 학습 곡선 비교하기](function_approximation/dqn_target_network_explained.md)
- [ ] [Atari Pong에서 DQN 훈련하기 (ALE-Py 사용)](function_approximation/dqn_atari_pong_explained.md)
- [ ] [Double DQN 구현 및 일반 DQN과 비교하기](function_approximation/double_dqn_cartpole_explained.md)

### 핵심 통찰
"치명적인 삼중주(Deadly Triad)"(함수 근사 + 부트스트래핑 + 오프-폴리시)는 불안정성을 유발합니다. DQN의 혁신은 이를 해결합니다.

### 도구
- PyTorch 또는 TensorFlow
- Gymnasium
- ALE-Py (Atari용)
- Weights & Biases (실험 추적용)

### 추천 자료
- 📖 Sutton & Barto - Chapters 9-11
- 📄 [DQN 논문 (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)
- 📄 [Rainbow 논문](https://arxiv.org/abs/1710.02298)
- 💻 [CleanRL 구현체](https://github.com/vwxyzjn/cleanrl)

### 이정표
Atari Pong에서 양수(+)의 보상을 얻는 DQN 에이전트를 훈련합니다.

---

## Phase 4: 정책 그래디언트 방법 (4-5주) {#phase-4-policy-gradient-methods-4-5-weeks}

### 목표
가치 함수를 계산하지 않고 정책을 직접 최적화하는 방법을 배웁니다.

### 학습 개념
- 정책 그래디언트 정리 (Policy Gradient Theorem)
- REINFORCE 알고리즘
- 분산 감소 기법
  - 베이스라인 (Baselines)
  - Reward-to-go
- 액터-크리틱 방법 (Actor-Critic Methods)
  - A2C (Advantage Actor-Critic)
  - A3C (비동기 변형)
- 일반화된 이득 추정 (Generalized Advantage Estimation, GAE)
- 신뢰 영역 방법 (Trust Region Methods)
  - TRPO (Trust Region Policy Optimization)
  - PPO (Proximal Policy Optimization)

### 실습 과제
- [ ] [CartPole을 위한 REINFORCE 구현하기](policy_gradients/reinforce_cartpole_explained.md)
- [ ] [베이스라인 추가 및 분산 감소 측정하기](policy_gradients/reinforce_baseline_explained.md)
- [ ] [LunarLander를 위한 A2C 구축하기](policy_gradients/a2c_lunarlander_explained.md)
- [ ] [PPO 밑바닥부터 구현하기](policy_gradients/ppo_scratch_explained.md)
- [ ] [연속 제어(BipedalWalker 또는 MuJoCo)에서 PPO 훈련하기](policy_gradients/ppo_continuous_explained.md)
- [ ] [PPO 하이퍼파라미터 민감도 비교하기](policy_gradients/ppo_hyperparams_explained.md)

### 핵심 통찰
PPO는 현대 강화학습의 주역입니다. 클리핑(clipping) 메커니즘을 깊이 있게 이해하세요.

### 도구
- PyTorch
- Gymnasium
- Stable-Baselines3 (참고용)
- MuJoCo 또는 Box2D (연속 제어용)

### 추천 자료
- 📖 Sutton & Barto - Chapter 13
- 📝 [OpenAI Spinning Up - Policy Gradients](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
- 📄 [PPO 논문 (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- 📄 [GAE 논문](https://arxiv.org/abs/1506.02438)
- 🎥 [Pieter Abbeel의 Deep RL Bootcamp](https://sites.google.com/view/deep-rl-bootcamp/lectures)

### 이정표
PPO를 구현하여 BipedalWalker-v3를 해결합니다 (보상 > 300).

---

## Phase 5: 심화 주제 (6-8주) {#phase-5-advanced-topics-6-8-weeks}

관심 분야에 따라 2~3개를 선택합니다.
- [모델 기반 강화학습 (Model-Based RL)](#model-based-rl)
- [다중 에이전트 강화학습 (Multi-Agent RL)](#multi-agent-rl)
- [오프라인 강화학습 (Offline/Batch RL)](#offlinebatch-rl)
- [탐험 (Exploration)](#exploration)
- [계층적 강화학습 (Hierarchical RL)](#hierarchical-rl)
- [RLHF (인간 피드백을 통한 강화학습)](#rlhf-rl-from-human-feedback)

### 모델 기반 강화학습 (Model-Based RL) {#model-based-rl}
환경의 역학을 학습하여 계획을 세우거나 가상의 경험을 생성합니다.

**개념:**
- Dyna 아키텍처
- 세계 모델 (World Models)
- 모델 예측 제어 (Model Predictive Control, MPC)
- MuZero, Dreamer

**실습 과제:**
- [ ] [Dyna-Q 구현하기](advanced_topics/model_based_rl/dyna_q_explained.md)
- [ ] [단순한 환경에서 세계 모델 훈련하기](advanced_topics/model_based_rl/world_model_explained.md)
- [ ] [계획 수립에 학습된 모델 활용하기](advanced_topics/model_based_rl/model_based_planning_explained.md)

**추천 자료:**
- 📖 Sutton & Barto - Chapter 8
- 📄 [World Models 논문](https://arxiv.org/abs/1803.10122)
- 📄 [MuZero 논문](https://arxiv.org/abs/1911.08265)
- 📄 [DreamerV3 논문](https://arxiv.org/abs/2301.04104)

---

### 다중 에이전트 강화학습 (Multi-Agent RL) {#multi-agent-rl}
공유된 환경에서 여러 에이전트가 동시에 학습합니다.

**개념:**
- 독립적 학습자 (Independent learners)
- 중앙 집중식 훈련, 분산식 실행 (CTDE)
- 셀프 플레이 (Self-play)
- 창발적 의사소통 (Emergent communication)

**실습 과제:**
- [ ] [단순한 행렬 게임에서 에이전트 훈련하기](advanced_topics/multi_agent_rl/matrix_games_explained.md)
- [ ] [보드게임을 위한 셀프 플레이 구현하기](advanced_topics/multi_agent_rl/self_play_tic_tac_toe_explained.md)
- [ ] [PettingZoo 환경 탐색하기](advanced_topics/multi_agent_rl/pettingzoo_explore_explained.md)

**추천 자료:**
- 📄 [Multi-Agent RL Survey](https://arxiv.org/abs/1911.10635)
- 📄 [OpenAI Five](https://arxiv.org/abs/1912.06680)
- 💻 [PettingZoo 라이브러리](https://pettingzoo.farama.org/)

---

### 오프라인 강화학습 (Offline/Batch RL) {#offlinebatch-rl}
환경과의 상호작용 없이 고정된 데이터셋으로부터 학습합니다.

**개념:**
- 분포 변화(Distribution shift) 문제
- 보수적 Q-러닝 (Conservative Q-Learning, CQL)
- 암묵적 Q-러닝 (Implicit Q-Learning, IQL)
- Decision Transformer

**실습 과제:**
- [ ] [D4RL 벤치마크 데이터셋에서 훈련하기](advanced_topics/offline_rl/d4rl_dataset_explained.md)
- [ ] [CQL 구현하기](advanced_topics/offline_rl/cql_explained.md)
- [ ] [행동 복제(Behavioral Cloning)와 비교하기](advanced_topics/offline_rl/behavioral_cloning_explained.md)

**추천 자료:**
- 📄 [Offline RL Tutorial](https://arxiv.org/abs/2005.01643)
- 📄 [CQL 논문](https://arxiv.org/abs/2006.04779)
- 📄 [Decision Transformer](https://arxiv.org/abs/2106.01345)
- 💻 [D4RL Benchmark](https://github.com/Farama-Foundation/D4RL)

---

### 탐험 (Exploration) {#exploration}
희소 보상 및 어려운 탐험 문제를 다룹니다.

**개념:**
- 내재적 동기부여 (Intrinsic motivation)
- 호기심 기반 탐험 (ICM)
- Random Network Distillation (RND)
- 카운트 기반 탐험 (Count-based exploration)
- Go-Explore

**실습 과제:**
- [ ] [호기심 보너스 구현하기](advanced_topics/exploration/curiosity_bonus_explained.md)
- [ ] [Montezuma's Revenge에서 훈련하기](advanced_topics/exploration/montezuma_revenge_explained.md)
- [ ] [탐험 전략 비교하기](advanced_topics/exploration/compare_exploration_explained.md)

**추천 자료:**
- 📄 [ICM 논문](https://arxiv.org/abs/1705.05363)
- 📄 [RND 논문](https://arxiv.org/abs/1810.12894)
- 📄 [Go-Explore](https://arxiv.org/abs/1901.10995)
- 📄 [Deep Exploration via Bootstrapped DQN](https://arxiv.org/abs/1602.04621) — DeepSea 과제의 출처

---

### 계층적 강화학습 (Hierarchical RL) {#hierarchical-rl}
여러 수준의 시간적 추상화 단계에서 학습합니다.

**개념:**
- 옵션 프레임워크 (Options framework)
- 목표 조건부 정책 (Goal-conditioned policies)
- Feudal networks
- HIRO, HAM

**실습 과제:**
- [ ] [옵션-크리틱 아키텍처 구현하기](advanced_topics/hierarchical_rl/option_critic_explained.md)
- [ ] [목표 조건부 정책 훈련하기](advanced_topics/hierarchical_rl/goal_conditioned_policy_explained.md)
- [ ] [장기 의존성 과제 테스트하기](advanced_topics/hierarchical_rl/long_horizon_tasks_explained.md)

**추천 자료:**
- 📖 Sutton & Barto - Chapter 12 (Options)
- 📄 [Option-Critic 논문](https://arxiv.org/abs/1609.05140)
- 📄 [HIRO 논문](https://arxiv.org/abs/1805.08296)

---

### RLHF (인간 피드백을 통한 강화학습) {#rlhf-rl-from-human-feedback}
인간의 선호도에 맞춰 모델을 정렬합니다.

**개념:**
- 비교 데이터를 통한 보상 모델링
- KL 제약 조건이 있는 정책 최적화
- Constitutional AI
- DPO (Direct Preference Optimization)

**실습 과제:**
- [ ] [선호도 데이터로부터 보상 모델 훈련하기](advanced_topics/rlhf/reward_modeling_explained.md)
- [ ] [PPO로 소형 언어 모델 미세 조정하기](advanced_topics/rlhf/ppo_fine_tuning_explained.md)
- [ ] [DPO 구현하기](advanced_topics/rlhf/dpo_implementation_explained.md)

**추천 자료:**
- 📄 [InstructGPT 논문](https://arxiv.org/abs/2203.02155)
- 📄 [Constitutional AI](https://arxiv.org/abs/2212.08073)
- 📄 [DPO 논문](https://arxiv.org/abs/2305.18290)
- 💻 [TRL 라이브러리](https://github.com/huggingface/trl)

---

## Phase 6: 연구 수준 (진행 중) {#phase-6-research-level-ongoing}

### 목표
해당 분야에 독창적인 기여를 합니다.

### 활동
- NeurIPS, ICML, ICLR 논문 정기적으로 읽기
- 최신 논문의 주요 결과 재현하기
- 공개된 문제 및 한계점 식별하기
- 적절한 평가 방식과 함께 엄격한 실험 수행하기
- 샘플 효율성, 일반화, 안전성 고려하기

### 연구해볼 만한 기념비적인 논문들
- [ ] [AlphaGo (Silver et al., 2016)](https://www.nature.com/articles/nature16961)
- [ ] [AlphaZero (Silver et al., 2018)](https://www.science.org/doi/10.1126/science.aar6404)
- [ ] [MuZero (Schrittwieser et al., 2020)](https://arxiv.org/abs/1911.08265)
- [ ] [Decision Transformer (Chen et al., 2021)](https://arxiv.org/abs/2106.01345)
- [ ] [InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)
- [ ] [DreamerV3 (Hafner et al., 2023)](https://arxiv.org/abs/2301.04104)

### 논문 게재처
- NeurIPS, ICML, ICLR (최우수 학회)
- AAAI, IJCAI
- CoRL (로봇 공학 중심)
- 초기 연구를 위한 워크숍 논문

---

## 도구 요약

| 단계 | 추천 도구 |
|-------|-------------------|
| 1-2 | NumPy, Matplotlib |
| 3 | PyTorch, Gymnasium, ALE-Py, W&B |
| 4 | Stable-Baselines3, MuJoCo/Box2D |
| 5-6 | 커스텀 코드, Ray/RLlib, JAX (속도 향상용) |

---

## 성공을 위한 팁

1. **밑바닥부터 직접 구현해 보기** — 라이브러리는 알고리즘을 완전히 이해한 후에 사용하세요.
2. **단순한 환경에서 디버깅하기** — 항상 Atari 이전에 CartPole부터 시작하세요.
3. **모든 것을 기록하기** — 보상, 손실, 그래디언트, 에피소드 길이 등.
4. **학습 과정 시각화하기** — 학습 곡선을 그리고 에피소드를 렌더링하세요.
5. **Sutton & Barto 책 읽기** — 강화학습의 바이블입니다.
6. **수학적 원리 이해하기** — 최소한 정책 그래디언트 정리와 벨만 방정식은 알아야 합니다.
7. **인내심 갖기** — 강화학습은 불안정하기로 유명합니다. 실패하는 실행은 지극히 정상입니다.
8. **시드(Seed) 사용하기** — 재현성이 중요합니다. 여러 시드의 평균을 사용하세요.
9. **커뮤니티 참여하기** — r/reinforcementlearning, RL Discord 등.

---

## 피해야 할 일반적인 실수

- ❌ 기초를 건너뛰고 바로 딥 강화학습으로 뛰어들기
- ❌ 관측값이나 보상을 정규화하지 않기
- ❌ 너무 크거나 작은 학습률 사용하기
- ❌ 테스트 시 평가 모드(evaluation mode) 설정을 잊기
- ❌ 실험 시 충분한 시드를 사용하지 않기
- ❌ 참조 코드 확인 없이 논문만 보고 구현하기
- ❌ 한 번의 훈련 실패로 포기하기

---

## 용어 사전

| 용어 | 정의 |
|------|------------|
| **MDP** | 마르코프 결정 과정 - 강화학습을 위한 공식 프레임워크 |
| **정책 (π)** | 상태에서 행동으로의 매핑 (어떻게 행동할지 결정) |
| **가치 함수 (V)** | 특정 상태에서 기대되는 리턴 |
| **Q-함수** | 특정 상태에서 특정 행동을 취했을 때 기대되는 리턴 |
| **TD 오차** | 예측값과 부트스트랩된 가치 사이의 차이 |
| **GAE** | 일반화된 이득 추정 (Generalized Advantage Estimation) |
| **PPO** | 근사 정책 최적화 (Proximal Policy Optimization) |
| **RLHF** | 인간 피드백을 통한 강화학습 (Reinforcement Learning from Human Feedback) |

---

## 라이선스

이 가이드는 교육 목적으로 제공됩니다. 자유롭게 공유하고 수정하셔도 좋습니다.

---
