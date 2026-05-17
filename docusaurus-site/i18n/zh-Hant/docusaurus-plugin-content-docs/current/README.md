# 強化學習：從入門到進階

一份從基礎概念到研究級理解，全面掌握強化學習系統建構的綜合指南。

---

## 概覽

| 階段 | 重點 | 持續時間 |
|-------|-------|----------|
| 1 | [基礎](#phase-1-foundations-2-4-weeks) | 2-4 週 |
| 2 | [表格方法](#phase-2-tabular-methods-3-4-weeks) | 3-4 週 |
| 3 | [函數近似](#phase-3-function-approximation-3-4-weeks) | 3-4 週 |
| 4 | [策略梯度方法](#phase-4-policy-gradient-methods-4-5-weeks) | 4-5 週 |
| 5 | [高級主題](#phase-5-advanced-topics-6-8-weeks) | 6-8 週 |
| 6 | [研究級別](#phase-6-research-level-ongoing) | 持續進行 |

**達到熟練所需的總時間：** 約 6 個月

---

## 階段 1：基礎 (2-4 週)

### 目標
在不深入數學的情況下理解核心概念。

### 學習概念
- 代理人 (Agent)、環境 (Environment)、狀態 (State)、動作 (Action)、獎勵 (Reward)
- 馬爾可夫決策過程 (MDP)
- 回報 (Return) 和折扣因子 (γ)
- 策略 (Policy) 與價值函數 (Value function)
- 探索 (Exploration) 與利用 (Exploitation) 的權衡

### 實踐工作
- [ ] [從零開始實作一個簡單的多臂拉霸機問題](foundations/multi_armed_bandit_explained.md)
- [ ] [使用隨機策略解決 Frozen Lake 並觀察結果](foundations/frozen_lake_explained.md)
- [ ] [在簡單網格上視覺化狀態價值函數](foundations/state_value_visualization_explained.md)

### 工具
- Python
- NumPy
- Matplotlib (用於視覺化)

### 資源
- 📖 [Sutton & Barto](http://incompleteideas.net/book/the-book.html) - 第 1-3 章
- 🎥 [David Silver 講座](https://www.youtube.com/watch?v=2pWv7GOvuf0) - 第 1-2 講
- 📝 [OpenAI Spinning Up - 介紹](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)

### 里程碑
你應該能夠解釋強化學習問題的形式化定義，以及為什麼 MDP 是標準框架。

---

## 階段 2：表格方法 (3-4 週)

### 目標
掌握狀態/動作空間足夠小、可以存儲在表格中的經典強化學習演算法。

### 學習概念
- 動態規劃 (Dynamic Programming)
  - 策略評估 (Policy Evaluation)
  - 策略迭代 (Policy Iteration)
  - 價值迭代 (Value Iteration)
- 蒙地卡羅方法 (Monte Carlo Methods)
  - 首次訪問 vs 每次訪問 MC
  - 帶有探索開始的 MC 控制
- 時序差分學習 (Temporal Difference Learning)
  - TD(0) 預測
  - SARSA (同策略 TD 控制)
  - Q-learning (異策略 TD 控制)
- 資格跡 (Eligibility Traces) 和 TD(λ)

### 實踐工作
- [ ] [為 GridWorld 實作策略迭代](tabular_methods/policy_iteration_gridworld_explained.md)
- [ ] [為 Frozen Lake 建構 Q-learning 代理人](tabular_methods/q_learning_frozen_lake_explained.md)
- [ ] [為 Cliff Walking 實作 SARSA](tabular_methods/sarsa_cliff_walking_explained.md)
- [ ] [比較 SARSA 與 Q-learning 的行為（安全路徑 vs 最優路徑）](tabular_methods/sarsa_vs_qlearning_explained.md)
- [ ] [為 21 點遊戲實作蒙地卡羅控制](tabular_methods/monte_carlo_blackjack_explained.md)

### 關鍵見解
理解回溯圖 (Backup diagrams)——它們闡明了每種演算法如何更新價值。

### 工具
- NumPy
- Gymnasium (用於環境)

### 資源
- 📖 Sutton & Barto - 第 4-7 章
- 🎥 David Silver 講座 3-5
- 💻 [Denny Britz 的強化學習倉庫](https://github.com/dennybritz/reinforcement-learning)

### 里程碑
從零開始實作 Q-learning 並解決 Frozen Lake，成功率 > 70%。

---

## 階段 3：函數近似 (3-4 週)

### 目標
使用函數近似器將強化學習擴展到小型狀態空間之外。

### 學習概念
- 為什麼表格方法在大規模下會失敗
- 線性函數近似
- 神經網路作為函數近似器
- 深度 Q 網路 (DQN)
  - 經驗回放 (Experience replay)
  - 目標網路 (Target networks)
  - 獎勵裁剪 (Reward clipping)
- DQN 的改進
  - Double DQN
  - Dueling DQN
  - 優先經驗回放 (Prioritized Experience Replay)
  - Rainbow DQN

### 實踐工作
- [ ] [使用線性 Q-learning 解決 CartPole](function_approximation/linear_q_cartpole_explained.md)
- [ ] [從零開始為 CartPole 實作 DQN](function_approximation/dqn_cartpole_explained.md)
- [ ] [添加經驗回放，觀察穩定性提升](function_approximation/dqn_experience_replay_explained.md)
- [ ] [添加目標網路，比較學習曲線](function_approximation/dqn_target_network_explained.md)
- [ ] [在 Atari Pong 上訓練 DQN (使用 ALE-Py)](function_approximation/dqn_atari_pong_explained.md)
- [ ] [實作 Double DQN，並與原始 DQN 進行比較](function_approximation/double_dqn_cartpole_explained.md)

### 關鍵見解
“致命三元組”(Deadly Triad：函數近似 + 自舉 + 異策略) 會導致不穩定。DQN 的創新解決了這些問題。

### 工具
- PyTorch 或 TensorFlow
- Gymnasium
- ALE-Py (用於 Atari)
- Weights & Biases (用於實驗跟踪)

### 資源
- 📖 Sutton & Barto - 第 9-11 章
- 📄 [DQN 論文 (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)
- 📄 [Rainbow 論文](https://arxiv.org/abs/1710.02298)
- 💻 [CleanRL 實作](https://github.com/vwxyzjn/cleanrl)

### 里程碑
訓練一個在 Atari Pong 上獲得正獎勵的 DQN 代理人。

---

## 階段 4：策略梯度方法 (4-5 週)

### 目標
學習直接優化策略，而不計算價值函數。

### 學習概念
- 策略梯度定理 (Policy Gradient Theorem)
- REINFORCE 演算法
- 方差縮減技術
  - 基準 (Baselines)
  - 獎勵待收 (Reward-to-go)
- 演員-評論家方法 (Actor-Critic Methods)
  - A2C (Advantage Actor-Critic)
  - A3C (異步變體)
- 廣義優勢估計 (GAE)
- 信任區域方法 (Trust Region Methods)
  - TRPO (Trust Region Policy Optimization)
  - PPO (Proximal Policy Optimization)

### 實踐工作
- [ ] [為 CartPole 實作 REINFORCE](policy_gradients/reinforce_cartpole_explained.md)
- [ ] [添加基準，衡量方差縮減效果](policy_gradients/reinforce_baseline_explained.md)
- [ ] [為 LunarLander 建構 A2C](policy_gradients/a2c_lunarlander_explained.md)
- [ ] [從零開始實作 PPO](policy_gradients/ppo_scratch_explained.md)
- [ ] [在連續控制任務上訓練 PPO (BipedalWalker 或 MuJoCo)](policy_gradients/ppo_continuous_explained.md)
- [ ] [比較 PPO 的超參數敏感性](policy_gradients/ppo_hyperparams_explained.md)

### 關鍵見解
PPO 是現代強化學習的主力演算法——深入理解其裁剪 (Clipping) 機制。

### 工具
- PyTorch
- Gymnasium
- Stable-Baselines3 (參考用)
- MuJoCo 或 Box2D (用於連續控制)

### 資源
- 📖 Sutton & Barto - 第 13 章
- 📝 [OpenAI Spinning Up - 策略梯度](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
- 📄 [PPO 論文 (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- 📄 [GAE 論文](https://arxiv.org/abs/1506.02438)
- 🎥 [Pieter Abbeel 的深度強化學習訓練營](https://sites.google.com/view/deep-rl-bootcamp/lectures)

### 里程碑
實作 PPO 並解決 BipedalWalker-v3 (獎勵 > 300)。

---

## 階段 5：高級主題 (6-8 週)

根據你的興趣選擇 2-3 個領域。
- [基於模型的強化學習](#model-based-rl)
- [多代理人強化學習](#multi-agent-rl)
- [離線/批量強化學習](#offlinebatch-rl)
- [探索](#exploration)
- [分層強化學習](#hierarchical-rl)
- [RLHF (來自人類回饋的強化學習)](#rlhf-rl-from-human-feedback)

### 基於模型的強化學習 (Model-Based RL)
學習環境動力學以進行規劃或生成合成經驗。

**概念：**
- Dyna 架構
- 世界模型 (World models)
- 模型預測控制 (MPC)
- MuZero, Dreamer

**實踐工作：**
- [ ] [實作 Dyna-Q](advanced_topics/model_based_rl/dyna_q_explained.md)
- [ ] [在簡單環境上訓練世界模型](advanced_topics/model_based_rl/world_model_explained.md)
- [ ] [使用學習到的模型進行規劃](advanced_topics/model_based_rl/model_based_planning_explained.md)

**資源：**
- 📖 Sutton & Barto - 第 8 章
- 📄 [世界模型論文](https://arxiv.org/abs/1803.10122)
- 📄 [MuZero 論文](https://arxiv.org/abs/1911.08265)
- 📄 [DreamerV3 論文](https://arxiv.org/abs/2301.04104)

---

### 多代理人強化學習 (Multi-Agent RL)
多個代理人在共享環境中同時學習。

**概念：**
- 獨立學習者
- 集中式訓練，分佈式執行 (CTDE)
- 自對弈 (Self-play)
- 湧現通訊 (Emergent communication)

**實踐工作：**
- [ ] [在簡單博弈論任務中訓練代理人](advanced_topics/multi_agent_rl/matrix_games_explained.md)
- [ ] [為棋類遊戲實作自對弈](advanced_topics/multi_agent_rl/self_play_tic_tac_toe_explained.md)
- [ ] [探索 PettingZoo 環境](advanced_topics/multi_agent_rl/pettingzoo_explore_explained.md)

**資源：**
- 📄 [多代理人強化學習綜述](https://arxiv.org/abs/1911.10635)
- 📄 [OpenAI Five](https://arxiv.org/abs/1912.06680)
- 💻 [PettingZoo 庫](https://pettingzoo.farama.org/)

---

### 離線/批量強化學習 (Offline/Batch RL)
在不與環境交互的情況下，從固定資料集中學習。

**概念：**
- 分佈偏移 (Distribution shift) 問題
- 保守 Q 學習 (CQL)
- 隱式 Q 學習 (IQL)
- Decision Transformer

**實踐工作：**
- [ ] [在 D4RL 基準資料集中進行訓練](advanced_topics/offline_rl/d4rl_dataset_explained.md)
- [ ] [實作 CQL](advanced_topics/offline_rl/cql_explained.md)
- [ ] [與行為克隆進行比較](advanced_topics/offline_rl/behavioral_cloning_explained.md)

**資源：**
- 📄 [離線強化學習教程](https://arxiv.org/abs/2005.01643)
- 📄 [CQL 論文](https://arxiv.org/abs/2006.04779)
- 📄 [Decision Transformer](https://arxiv.org/abs/2106.01345)
- 💻 [D4RL 基準](https://github.com/Farama-Foundation/D4RL)

---

### 探索 (Exploration)
解決稀疏獎勵和極難探索的問題。

**概念：**
- 內在動機 (Intrinsic motivation)
- 好奇心驅動的探索 (ICM)
- 隨機網路蒸餾 (RND)
- 基於計數的探索
- Go-Explore

**實踐工作：**
- [ ] [實作好奇心獎勵](advanced_topics/exploration/curiosity_bonus_explained.md)
- [ ] [在 Montezuma's Revenge 上訓練](advanced_topics/exploration/montezuma_revenge_explained.md)
- [ ] [比較各種探索策略](advanced_topics/exploration/compare_exploration_explained.md)

**資源：**
- 📄 [ICM 論文](https://arxiv.org/abs/1705.05363)
- 📄 [RND 論文](https://arxiv.org/abs/1810.12894)
- 📄 [Go-Explore](https://arxiv.org/abs/1901.10995)
- 📄 [通過自舉 DQN 進行深度探索 (Deep Exploration via Bootstrapped DQN)](https://arxiv.org/abs/1602.04621) — DeepSea 任務來源

---

### 分層強化學習 (Hierarchical RL)
在多個時間抽象層級上進行學習。

**概念：**
- 選項 (Options) 框架
- 目標條件策略 (Goal-conditioned policies)
- 封建網路 (Feudal networks)
- HIRO, HAM

**實踐工作：**
- [ ] [實作 option-critic 架構](advanced_topics/hierarchical_rl/option_critic_explained.md)
- [ ] [訓練目標條件策略](advanced_topics/hierarchical_rl/goal_conditioned_policy_explained.md)
- [ ] [在長時程任務上測試](advanced_topics/hierarchical_rl/long_horizon_tasks_explained.md)

**資源：**
- 📖 Sutton & Barto - 第 12 章 (Options)
- 📄 [Option-Critic 論文](https://arxiv.org/abs/1609.05140)
- 📄 [HIRO 論文](https://arxiv.org/abs/1805.08296)

---

### RLHF (來自人類回饋的強化學習)
使模型與人類偏好對齊。

**概念：**
- 通過比較進行獎勵建模
- KL 約束策略優化
- 憲法 AI (Constitutional AI)
- DPO (直接偏好優化)

**實踐工作：**
- [ ] [從偏好資料訓練獎勵模型](advanced_topics/rlhf/reward_modeling_explained.md)
- [ ] [使用 PPO 微調小型語言模型](advanced_topics/rlhf/ppo_fine_tuning_explained.md)
- [ ] [實作 DPO](advanced_topics/rlhf/dpo_implementation_explained.md)

**資源：**
- 📄 [InstructGPT 論文](https://arxiv.org/abs/2203.02155)
- 📄 [憲法 AI](https://arxiv.org/abs/2212.08073)
- 📄 [DPO 論文](https://arxiv.org/abs/2305.18290)
- 💻 [TRL 庫](https://github.com/huggingface/trl)

---

## 階段 6：研究級別 (持續進行)

### 目標
為該領域貢獻原創性工作。

### 活動
- 定期閱讀 NeurIPS, ICML, ICLR 的論文
- 重現近期論文的關鍵結果
- 識別開放性問題和局限性
- 進行嚴謹的實驗並進行適當評估
- 考慮樣本效率、泛化性、安全性

### 必讀的里程碑式論文
- [ ] [AlphaGo (Silver et al., 2016)](https://www.nature.com/articles/nature16961)
- [ ] [AlphaZero (Silver et al., 2018)](https://www.science.org/doi/10.1126/science.aar6404)
- [ ] [MuZero (Schrittwieser et al., 2020)](https://arxiv.org/abs/1911.08265)
- [ ] [Decision Transformer (Chen et al., 2021)](https://arxiv.org/abs/2106.01345)
- [ ] [InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)
- [ ] [DreamerV3 (Hafner et al., 2023)](https://arxiv.org/abs/2301.04104)

### 發表途徑
- NeurIPS, ICML, ICLR (頂級會議)
- AAAI, IJCAI
- CoRL (專注於機器人)
- 早期工作的研討會論文 (Workshop papers)

---

## 工具總結

| 階段 | 推薦工具 |
|-------|-------------------|
| 1-2 | NumPy, Matplotlib |
| 3 | PyTorch, Gymnasium, ALE-Py, W&B |
| 4 | Stable-Baselines3, MuJoCo/Box2D |
| 5-6 | 自定義代碼, Ray/RLlib, JAX (追求速度) |

---

## 成功技巧

1. **先從零開始實作** — 只有在你理解演算法後才使用庫
2. **使用簡單的環境進行調試** — 始終先嘗試 CartPole，然後再嘗試 Atari
3. **記錄一切** — 獎勵、損失、梯度、回合長度
4. **視覺化學習過程** — 繪製學習曲線，渲染回合
5. **閱讀 Sutton & Barto 的書** — 這是強化學習的“聖經”
6. **理解數學** — 至少掌握策略梯度定理和貝爾曼方程
7. **保持耐心** — 強化學習以不穩定著稱；訓練失敗是常態
8. **使用隨機種子** — 可重複性至關重要；在多個種子上取平均值
9. **加入社群** — r/reinforcementlearning, RL Discord, Twitter/X

---

## 應避免的常見陷阱

- ❌ 跳過基礎知識直接跳到深度強化學習
- ❌ 沒有對觀察/獎勵進行歸一化
- ❌ 使用過大或過小的學習率
- ❌ 在測試期間忘記設置評估模式
- ❌ 實驗時沒有使用足夠的隨機種子
- ❌ 直接根據論文實作而不檢查參考代碼
- ❌ 一次訓練失敗就放棄

---

## 術語表

| 術語 | 定義 |
|------|------------|
| **MDP** | 馬爾可夫決策過程 - 強化學習的形式化框架 |
| **策略 (π)** | 從狀態到動作的映射 |
| **價值函數 (V)** | 來自某個狀態的預期回報 |
| **Q 函數** | 來自某個狀態-動作對的預期回報 |
| **TD 誤差** | 預測值與自舉值之間的差異 |
| **GAE** | 廣義優勢估計 |
| **PPO** | 近端策略優化 |
| **RLHF** | 來自人類回饋的強化學習 |

---

## 許可證

本指南僅用於教學目的。歡迎分享和改編。
