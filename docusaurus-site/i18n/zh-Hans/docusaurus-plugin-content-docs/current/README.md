# 强化学习：从入门到进阶

一份从基础概念到研究级理解，全面掌握强化学习系统构建的综合指南。

---

## 概览

| 阶段 | 重点 | 持续时间 |
|-------|-------|----------|
| 1 | [基础](#phase-1-foundations-2-4-weeks) | 2-4 周 |
| 2 | [表格方法](#phase-2-tabular-methods-3-4-weeks) | 3-4 周 |
| 3 | [函数近似](#phase-3-function-approximation-3-4-weeks) | 3-4 周 |
| 4 | [策略梯度方法](#phase-4-policy-gradient-methods-4-5-weeks) | 4-5 周 |
| 5 | [高级主题](#phase-5-advanced-topics-6-8-weeks) | 6-8 周 |
| 6 | [研究级别](#phase-6-research-level-ongoing) | 持续进行 |

**达到熟练所需的总时间：** 约 6 个月

---

## 阶段 1：基础 (2-4 周) {#phase-1-foundations-2-4-weeks}

### 目标
在不深入数学的情况下理解核心概念。

### 学习概念
- 智能体 (Agent)、环境 (Environment)、状态 (State)、动作 (Action)、奖励 (Reward)
- 马尔可夫决策过程 (MDP)
- 回报 (Return) 和折扣因子 (γ)
- 策略 (Policy) 与价值函数 (Value function)
- 探索 (Exploration) 与利用 (Exploitation) 的权衡

### 实践工作
- [ ] [从零开始实现一个简单的多臂老虎机问题](foundations/multi_armed_bandit_explained.md)
- [ ] [使用随机策略解决 Frozen Lake 并观察结果](foundations/frozen_lake_explained.md)
- [ ] [在简单网格上可视化状态价值函数](foundations/state_value_visualization_explained.md)

### 工具
- Python
- NumPy
- Matplotlib (用于可视化)

### 资源
- 📖 [Sutton & Barto](http://incompleteideas.net/book/the-book.html) - 第 1-3 章
- 🎥 [David Silver 讲座](https://www.youtube.com/watch?v=2pWv7GOvuf0) - 第 1-2 讲
- 📝 [OpenAI Spinning Up - 介绍](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)

### 里程碑
你应该能够解释强化学习问题的形式化定义，以及为什么 MDP 是标准框架。

---

## 阶段 2：表格方法 (3-4 周) {#phase-2-tabular-methods-3-4-weeks}

### 目标
掌握状态/动作空间足够小、可以存储在表格中的经典强化学习算法。

### 学习概念
- 动态规划 (Dynamic Programming)
  - 策略评估 (Policy Evaluation)
  - 策略迭代 (Policy Iteration)
  - 价值迭代 (Value Iteration)
- 蒙特卡洛方法 (Monte Carlo Methods)
  - 首次访问 vs 每次访问 MC
  - 带有探索开始的 MC 控制
- 时序差分学习 (Temporal Difference Learning)
  - TD(0) 预测
  - SARSA (同策略 TD 控制)
  - Q-learning (异策略 TD 控制)
- 资格迹 (Eligibility Traces) 和 TD(λ)

### 实践工作
- [ ] [为 GridWorld 实现策略迭代](tabular_methods/policy_iteration_gridworld_explained.md)
- [ ] [为 Frozen Lake 构建 Q-learning 智能体](tabular_methods/q_learning_frozen_lake_explained.md)
- [ ] [为 Cliff Walking 实现 SARSA](tabular_methods/sarsa_cliff_walking_explained.md)
- [ ] [比较 SARSA 与 Q-learning 的行为（安全路径 vs 最优路径）](tabular_methods/sarsa_vs_qlearning_explained.md)
- [ ] [为 21 点游戏实现蒙特卡洛控制](tabular_methods/monte_carlo_blackjack_explained.md)

### 关键见解
理解回溯图 (Backup diagrams)——它们阐明了每种算法如何更新价值。

### 工具
- NumPy
- Gymnasium (用于环境)

### 资源
- 📖 Sutton & Barto - 第 4-7 章
- 🎥 David Silver 讲座 3-5
- 💻 [Denny Britz 的强化学习仓库](https://github.com/dennybritz/reinforcement-learning)

### 里程碑
从零开始实现 Q-learning 并解决 Frozen Lake，成功率 > 70%。

---

## 阶段 3：函数近似 (3-4 周) {#phase-3-function-approximation-3-4-weeks}

### 目标
使用函数近似器将强化学习扩展到小型状态空间之外。

### 学习概念
- 为什么表格方法在大规模下会失败
- 线性函数近似
- 神经网络作为函数近似器
- 深度 Q 网络 (DQN)
  - 经验回放 (Experience replay)
  - 目标网络 (Target networks)
  - 奖励裁剪 (Reward clipping)
- DQN 的改进
  - Double DQN
  - Dueling DQN
  - 优先经验回放 (Prioritized Experience Replay)
  - Rainbow DQN

### 实践工作
- [ ] [使用线性 Q-learning 解决 CartPole](function_approximation/linear_q_cartpole_explained.md)
- [ ] [从零开始为 CartPole 实现 DQN](function_approximation/dqn_cartpole_explained.md)
- [ ] [添加经验回放，观察稳定性提升](function_approximation/dqn_experience_replay_explained.md)
- [ ] [添加目标网络，比较学习曲线](function_approximation/dqn_target_network_explained.md)
- [ ] [在 Atari Pong 上训练 DQN (使用 ALE-Py)](function_approximation/dqn_atari_pong_explained.md)
- [ ] [实现 Double DQN，并与原始 DQN 进行比较](function_approximation/double_dqn_cartpole_explained.md)

### 关键见解
“致命三元组”(Deadly Triad：函数近似 + 自举 + 异策略) 会导致不稳定。DQN 的创新解决了这些问题。

### 工具
- PyTorch 或 TensorFlow
- Gymnasium
- ALE-Py (用于 Atari)
- Weights & Biases (用于实验跟踪)

### 资源
- 📖 Sutton & Barto - 第 9-11 章
- 📄 [DQN 论文 (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)
- 📄 [Rainbow 论文](https://arxiv.org/abs/1710.02298)
- 💻 [CleanRL 实现](https://github.com/vwxyzjn/cleanrl)

### 里程碑
训练一个在 Atari Pong 上获得正奖励的 DQN 智能体。

---

## 阶段 4：策略梯度方法 (4-5 周) {#phase-4-policy-gradient-methods-4-5-weeks}

### 目标
学习直接优化策略，而不计算价值函数。

### 学习概念
- 策略梯度定理 (Policy Gradient Theorem)
- REINFORCE 算法
- 方差缩减技术
  - 基准 (Baselines)
  - 奖励待收 (Reward-to-go)
- 演员-评论家方法 (Actor-Critic Methods)
  - A2C (Advantage Actor-Critic)
  - A3C (异步变体)
- 广义优势估计 (GAE)
- 置信区域方法 (Trust Region Methods)
  - TRPO (Trust Region Policy Optimization)
  - PPO (Proximal Policy Optimization)

### 实践工作
- [ ] [为 CartPole 实现 REINFORCE](policy_gradients/reinforce_cartpole_explained.md)
- [ ] [添加基准，衡量方差缩减效果](policy_gradients/reinforce_baseline_explained.md)
- [ ] [为 LunarLander 构建 A2C](policy_gradients/a2c_lunarlander_explained.md)
- [ ] [从零开始实现 PPO](policy_gradients/ppo_scratch_explained.md)
- [ ] [在连续控制任务上训练 PPO (BipedalWalker 或 MuJoCo)](policy_gradients/ppo_continuous_explained.md)
- [ ] [比较 PPO 的超参数敏感性](policy_gradients/ppo_hyperparams_explained.md)

### 关键见解
PPO 是现代强化学习的主力算法——深入理解其裁剪 (Clipping) 机制。

### 工具
- PyTorch
- Gymnasium
- Stable-Baselines3 (参考用)
- MuJoCo 或 Box2D (用于连续控制)

### 资源
- 📖 Sutton & Barto - 第 13 章
- 📝 [OpenAI Spinning Up - 策略梯度](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
- 📄 [PPO 论文 (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- 📄 [GAE 论文](https://arxiv.org/abs/1506.02438)
- 🎥 [Pieter Abbeel 的深度强化学习训练营](https://sites.google.com/view/deep-rl-bootcamp/lectures)

### 里程碑
实现 PPO 并解决 BipedalWalker-v3 (奖励 > 300)。

---

## 阶段 5：高级主题 (6-8 周) {#phase-5-advanced-topics-6-8-weeks}

根据你的兴趣选择 2-3 个领域。
- [基于模型的强化学习](#model-based-rl)
- [多智能体强化学习](#multi-agent-rl)
- [离线/批量强化学习](#offlinebatch-rl)
- [探索](#exploration)
- [分层强化学习](#hierarchical-rl)
- [RLHF (来自人类反馈的强化学习)](#rlhf-rl-from-human-feedback)

### 基于模型的强化学习 (Model-Based RL) {#model-based-rl}
学习环境动力学以进行规划或生成合成经验。

**概念：**
- Dyna 架构
- 世界模型 (World models)
- 模型预测控制 (MPC)
- MuZero, Dreamer

**实践工作：**
- [ ] [实现 Dyna-Q](advanced_topics/model_based_rl/dyna_q_explained.md)
- [ ] [在简单环境上训练世界模型](advanced_topics/model_based_rl/world_model_explained.md)
- [ ] [使用学习到的模型进行规划](advanced_topics/model_based_rl/model_based_planning_explained.md)

**资源：**
- 📖 Sutton & Barto - 第 8 章
- 📄 [世界模型论文](https://arxiv.org/abs/1803.10122)
- 📄 [MuZero 论文](https://arxiv.org/abs/1911.08265)
- 📄 [DreamerV3 论文](https://arxiv.org/abs/2301.04104)

---

### 多智能体强化学习 (Multi-Agent RL) {#multi-agent-rl}
多个智能体在共享环境中同时学习。

**概念：**
- 独立学习者
- 集中式训练，分布式执行 (CTDE)
- 自博弈 (Self-play)
- 涌现通信 (Emergent communication)

**实践工作：**
- [ ] [在简单博弈论任务中训练智能体](advanced_topics/multi_agent_rl/matrix_games_explained.md)
- [ ] [为棋类游戏实现自博弈](advanced_topics/multi_agent_rl/self_play_tic_tac_toe_explained.md)
- [ ] [探索 PettingZoo 环境](advanced_topics/multi_agent_rl/pettingzoo_explore_explained.md)

**资源：**
- 📄 [多智能体强化学习综述](https://arxiv.org/abs/1911.10635)
- 📄 [OpenAI Five](https://arxiv.org/abs/1912.06680)
- 💻 [PettingZoo 库](https://pettingzoo.farama.org/)

---

### 离线/批量强化学习 (Offline/Batch RL) {#offlinebatch-rl}
在不与环境交互的情况下，从固定数据集中学习。

**概念：**
- 分布偏移 (Distribution shift) 问题
- 保守 Q 学习 (CQL)
- 隐式 Q 学习 (IQL)
- Decision Transformer

**实践工作：**
- [ ] [在 D4RL 基准数据集中进行训练](advanced_topics/offline_rl/d4rl_dataset_explained.md)
- [ ] [实现 CQL](advanced_topics/offline_rl/cql_explained.md)
- [ ] [与行为克隆进行比较](advanced_topics/offline_rl/behavioral_cloning_explained.md)

**资源：**
- 📄 [离线强化学习教程](https://arxiv.org/abs/2005.01643)
- 📄 [CQL 论文](https://arxiv.org/abs/2006.04779)
- 📄 [Decision Transformer](https://arxiv.org/abs/2106.01345)
- 💻 [D4RL 基准](https://github.com/Farama-Foundation/D4RL)

---

### 探索 (Exploration) {#exploration}
解决稀疏奖励和极难探索的问题。

**概念：**
- 内在动机 (Intrinsic motivation)
- 好奇心驱动的探索 (ICM)
- 随机网络蒸馏 (RND)
- 基于计数的探索
- Go-Explore

**实践工作：**
- [ ] [实现好奇心奖励](advanced_topics/exploration/curiosity_bonus_explained.md)
- [ ] [在 Montezuma's Revenge 上训练](advanced_topics/exploration/montezuma_revenge_explained.md)
- [ ] [比较各种探索策略](advanced_topics/exploration/compare_exploration_explained.md)

**资源：**
- 📄 [ICM 论文](https://arxiv.org/abs/1705.05363)
- 📄 [RND 论文](https://arxiv.org/abs/1810.12894)
- 📄 [Go-Explore](https://arxiv.org/abs/1901.10995)
- 📄 [通过自举 DQN 进行深度探索 (Deep Exploration via Bootstrapped DQN)](https://arxiv.org/abs/1602.04621) — DeepSea 任务来源

---

### 分层强化学习 (Hierarchical RL) {#hierarchical-rl}
在多个时间抽象层级上进行学习。

**概念：**
- 选项 (Options) 框架
- 目标条件策略 (Goal-conditioned policies)
- 封建网络 (Feudal networks)
- HIRO, HAM

**实践工作：**
- [ ] [实现 option-critic 架构](advanced_topics/hierarchical_rl/option_critic_explained.md)
- [ ] [训练目标条件策略](advanced_topics/hierarchical_rl/goal_conditioned_policy_explained.md)
- [ ] [在长时程任务上测试](advanced_topics/hierarchical_rl/long_horizon_tasks_explained.md)

**资源：**
- 📖 Sutton & Barto - 第 12 章 (Options)
- 📄 [Option-Critic 论文](https://arxiv.org/abs/1609.05140)
- 📄 [HIRO 论文](https://arxiv.org/abs/1805.08296)

---

### RLHF (来自人类反馈的强化学习) {#rlhf-rl-from-human-feedback}
使模型与人类偏好对齐。

**概念：**
- 通过比较进行奖励建模
- KL 约束策略优化
- 宪法 AI (Constitutional AI)
- DPO (直接偏好优化)

**实践工作：**
- [ ] [从偏好数据训练奖励模型](advanced_topics/rlhf/reward_modeling_explained.md)
- [ ] [使用 PPO 微调小型语言模型](advanced_topics/rlhf/ppo_fine_tuning_explained.md)
- [ ] [实现 DPO](advanced_topics/rlhf/dpo_implementation_explained.md)

**资源：**
- 📄 [InstructGPT 论文](https://arxiv.org/abs/2203.02155)
- 📄 [宪法 AI](https://arxiv.org/abs/2212.08073)
- 📄 [DPO 论文](https://arxiv.org/abs/2305.18290)
- 💻 [TRL 库](https://github.com/huggingface/trl)

---

## 阶段 6：研究级别 (持续进行) {#phase-6-research-level-ongoing}

### 目标
为该领域贡献原创性工作。

### 活动
- 定期阅读 NeurIPS, ICML, ICLR 的论文
- 复现近期论文的关键结果
- 识别开放性问题和局限性
- 进行严谨的实验并进行适当评估
- 考虑样本效率、泛化性、安全性

### 必读的里程碑式论文
- [ ] [AlphaGo (Silver et al., 2016)](https://www.nature.com/articles/nature16961)
- [ ] [AlphaZero (Silver et al., 2018)](https://www.science.org/doi/10.1126/science.aar6404)
- [ ] [MuZero (Schrittwieser et al., 2020)](https://arxiv.org/abs/1911.08265)
- [ ] [Decision Transformer (Chen et al., 2021)](https://arxiv.org/abs/2106.01345)
- [ ] [InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)
- [ ] [DreamerV3 (Hafner et al., 2023)](https://arxiv.org/abs/2301.04104)

### 发表途径
- NeurIPS, ICML, ICLR (顶级会议)
- AAAI, IJCAI
- CoRL (专注于机器人)
- 早期工作的研讨会论文 (Workshop papers)

---

## 工具总结

| 阶段 | 推荐工具 |
|-------|-------------------|
| 1-2 | NumPy, Matplotlib |
| 3 | PyTorch, Gymnasium, ALE-Py, W&B |
| 4 | Stable-Baselines3, MuJoCo/Box2D |
| 5-6 | 自定义代码, Ray/RLlib, JAX (追求速度) |

---

## 成功技巧

1. **先从零开始实现** — 只有在你理解算法后才使用库
2. **使用简单的环境进行调试** — 始终先尝试 CartPole，然后再尝试 Atari
3. **记录一切** — 奖励、损失、梯度、回合长度
4. **可视化学习过程** — 绘制学习曲线，渲染回合
5. **阅读 Sutton & Barto 的书** — 这是强化学习的“圣经”
6. **理解数学** — 至少掌握策略梯度定理和贝尔曼方程
7. **保持耐心** — 强化学习以不稳定著称；训练失败是常态
8. **使用随机种子** — 可复现性至关重要；在多个种子上取平均值
9. **加入社区** — r/reinforcementlearning, RL Discord, Twitter/X

---

## 应避免的常见陷阱

- ❌ 跳过基础知识直接跳到深度强化学习
- ❌ 没有对观察/奖励进行归一化
- ❌ 使用过大或过小的学习率
- ❌ 在测试期间忘记设置评估模式
- ❌ 实验时没有使用足够的随机种子
- ❌ 直接根据论文实现而不检查参考代码
- ❌ 一次训练失败就放弃

---

## 术语表

| 术语 | 定义 |
|------|------------|
| **MDP** | 马尔可夫决策过程 - 强化学习的形式化框架 |
| **策略 (π)** | 从状态到动作的映射 |
| **价值函数 (V)** | 来自某个状态的预期回报 |
| **Q 函数** | 来自某个状态-动作对的预期回报 |
| **TD 误差** | 预测值与自举值之间的差异 |
| **GAE** | 广义优势估计 |
| **PPO** | 近端策略优化 |
| **RLHF** | 来自人类反馈的强化学习 |

---

## 许可证

本指南仅用于教学目的。欢迎分享和改编。

---
