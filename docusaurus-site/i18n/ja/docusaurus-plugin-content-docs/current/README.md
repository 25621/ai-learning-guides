# 強化学習：初級から上級まで

基本概念から研究レベルの理解まで、強化学習システムを理解し構築するための包括的なガイドです。

---

## 概要

| フェーズ | 焦点 | 期間 |
|-------|-------|----------|
| 1 | [基礎](#phase-1-foundations-2-4-weeks) | 2-4週間 |
| 2 | [テーブル手法](#phase-2-tabular-methods-3-4-weeks) | 3-4週間 |
| 3 | [関数近似](#phase-3-function-approximation-3-4-weeks) | 3-4週間 |
| 4 | [方策勾配法](#phase-4-policy-gradient-methods-4-5-weeks) | 4-5週間 |
| 5 | [高度なトピック](#phase-5-advanced-topics-6-8-weeks) | 6-8週間 |
| 6 | [研究レベル](#phase-6-research-level-ongoing) | 継続的 |

**習得までの合計時間:** 約6ヶ月

---

## フェーズ 1: 基礎 (2-4週間) {#phase-1-foundations-2-4-weeks}

### 目標
深い数学を使わずにコア概念を理解する。

### 学習する概念
- エージェント、環境、状態、行動、報酬
- マルコフ決定過程 (MDP)
- 収益と割引率 (γ)
- 方策 vs 価値関数
- 探索と利用のトレードオフ

### 実践ワーク
- [ ] [シンプルな多腕バンディット問題をゼロから実装する](foundations/multi_armed_bandit_explained.md)
- [ ] [ランダム方策で Frozen Lake を解き、結果を観察する](foundations/frozen_lake_explained.md)
- [ ] [シンプルなグリッド上で状態価値関数を可視化する](foundations/state_value_visualization_explained.md)

### ツール
- Python
- NumPy
- Matplotlib (可視化用)

### リソース
- 📖 [Sutton & Barto](http://incompleteideas.net/book/the-book.html) - 第1-3章
- 🎥 [David Silver 講義](https://www.youtube.com/watch?v=2pWv7GOvuf0) - 第1-2回
- 📝 [OpenAI Spinning Up - 導入](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)

### マイルストーン
強化学習の問題定式化と、なぜ MDP が標準的なフレームワークであるかを説明できること。

---

## フェーズ 2: テーブル手法 (3-4週間) {#phase-2-tabular-methods-3-4-weeks}

### 目標
状態/行動空間がテーブルに保存できるほど小さい古典的な強化学習アルゴリズムをマスターする。

### 学習する概念
- 動的計画法
  - 方策評価
  - 方策反復
  - 価値反復
- モンテカルロ法
  - 逐次訪問 (First-visit) vs 毎訪問 (Every-visit) MC
  - 探索開始付き MC 制御
- 時間的差分学習 (TD学習)
  - TD(0) 予測
  - SARSA (オンポリス TD 制御)
  - Q学習 (オフポリス TD 制御)
- 適格度トレースと TD(λ)

### 実践ワーク
- [ ] [GridWorld の方策反復を実装する](tabular_methods/policy_iteration_gridworld_explained.md)
- [ ] [Frozen Lake 用の Q学習エージェントを構築する](tabular_methods/q_learning_frozen_lake_explained.md)
- [ ] [Cliff Walking 用に SARSA を実装する](tabular_methods/sarsa_cliff_walking_explained.md)
- [ ] [SARSA vs Q学習の挙動を比較する (安全な経路 vs 最適な経路)](tabular_methods/sarsa_vs_qlearning_explained.md)
- [ ] [ブラックジャック用のモンテカルロ制御を実装する](tabular_methods/monte_carlo_blackjack_explained.md)

### 重要な洞察
バックアップ図を理解すること。これらは各アルゴリズムがどのように価値を更新するかを明確にします。

### ツール
- NumPy
- Gymnasium (環境用)

### リソース
- 📖 Sutton & Barto - 第4-7章
- 🎥 David Silver 講義 3-5
- 💻 [Denny Britz の RL リポジトリ](https://github.com/dennybritz/reinforcement-learning)

### マイルストーン
Q学習をゼロから実装し、Frozen Lake を 70% 以上の成功率で解決すること。

---

## フェーズ 3: 関数近似 (3-4週間) {#phase-3-function-approximation-3-4-weeks}

### 目標
関数近似器を使用して、小規模な状態空間を超えて強化学習をスケールさせる。

### 学習する概念
- なぜテーブル手法が大規模環境で失敗するのか
- 線形関数近似
- 関数近似器としてのニューラルネットワーク
- Deep Q-Networks (DQN)
  - 経験再生 (Experience replay)
  - ターゲットネットワーク
  - 報酬クリッピング
- DQN の改善
  - Double DQN
  - Dueling DQN
  - 優先度付き経験再生 (Prioritized Experience Replay)
  - Rainbow DQN

### 実践ワーク
- [ ] [線形 Q学習で CartPole を解く](function_approximation/linear_q_cartpole_explained.md)
- [ ] [CartPole 用の DQN をゼロから実装する](function_approximation/dqn_cartpole_explained.md)
- [ ] [経験再生を追加し、安定性の向上を観察する](function_approximation/dqn_experience_replay_explained.md)
- [ ] [ターゲットネットワークを追加し、学習曲線を比較する](function_approximation/dqn_target_network_explained.md)
- [ ] [Atari Pong で DQN を訓練する (ALE-Py を使用)](function_approximation/dqn_atari_pong_explained.md)
- [ ] [Double DQN を実装し、通常の DQN と比較する](function_approximation/double_dqn_cartpole_explained.md)

### 重要な洞察
「死の三つ組」(関数近似 + ブートストラップ + オフポリス) は不安定性を引き起こします。DQN の革新はこれに対処するものです。

### ツール
- PyTorch または TensorFlow
- Gymnasium
- ALE-Py (Atari 用)
- Weights & Biases (実験管理用)

### リソース
- 📖 Sutton & Barto - 第9-11章
- 📄 [DQN 論文 (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)
- 📄 [Rainbow 論文](https://arxiv.org/abs/1710.02298)
- 💻 [CleanRL の実装](https://github.com/vwxyzjn/cleanrl)

### マイルストーン
Atari Pong でプラスの報酬を獲得する DQN エージェントを訓練すること。

---

## フェーズ 4: 方策勾配法 (4-5週間) {#phase-4-policy-gradient-methods-4-5-weeks}

### 目標
価値関数を計算せずに、方策を直接最適化する方法を学ぶ。

### 学習する概念
- 方策勾配定理
- REINFORCE アルゴリズム
- 分散減少手法
  - ベースライン
  - Reward-to-go
- アクター・クリティック法
  - A2C (Advantage Actor-Critic)
  - A3C (非同期版)
- 一般化アドバンテージ推定 (GAE)
- 信賴領域手法
  - TRPO (Trust Region Policy Optimization)
  - PPO (Proximal Policy Optimization)

### 実践ワーク
- [ ] [CartPole 用に REINFORCE を実装する](policy_gradients/reinforce_cartpole_explained.md)
- [ ] [ベースラインを追加し、分散減少を測定する](policy_gradients/reinforce_baseline_explained.md)
- [ ] [LunarLander 用に A2C を構築する](policy_gradients/a2c_lunarlander_explained.md)
- [ ] [PPO をゼロから実装する](policy_gradients/ppo_scratch_explained.md)
- [ ] [連続値制御 (BipedalWalker または MuJoCo) で PPO を訓練する](policy_gradients/ppo_continuous_explained.md)
- [ ] [PPO のハイパーパラメータ感度を比較する](policy_gradients/ppo_hyperparams_explained.md)

### 重要な洞察
PPO は現代の強化学習の主力です。そのクリッピング・メカニズムを深く理解しましょう。

### ツール
- PyTorch
- Gymnasium
- Stable-Baselines3 (参照用)
- MuJoCo または Box2D (連続値制御用)

### リソース
- 📖 Sutton & Barto - 第13章
- 📝 [OpenAI Spinning Up - 方策勾配](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
- 📄 [PPO 論文 (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- 📄 [GAE 論文](https://arxiv.org/abs/1506.02438)
- 🎥 [Pieter Abbeel の Deep RL Bootcamp](https://sites.google.com/view/deep-rl-bootcamp/lectures)

### マイルストーン
PPO を実装し、BipedalWalker-v3 を解決すること (報酬 > 300)。

---

## フェーズ 5: 高度なトピック (6-8週間) {#phase-5-advanced-topics-6-8-weeks}

興味に基づいて 2-3 の分野を選択してください。
- [モデルベース強化学習](#model-based-rl)
- [マルチエージェント強化学習](#multi-agent-rl)
- [オフライン/バッチ強化学習](#offlinebatch-rl)
- [探索](#exploration)
- [階層的強化学習](#hierarchical-rl)
- [RLHF (人間からのフィードバックによる強化学習)](#rlhf-rl-from-human-feedback)

### モデルベース強化学習 {#model-based-rl}
環境のダイナミクスを学習し、計画を立てたり合成経験を生成したりする。

**概念:**
- Dyna アーキテクチャ
- 世界モデル (World models)
- モデル予測制御 (MPC)
- MuZero, Dreamer

**実践ワーク:**
- [ ] [Dyna-Q の実装](advanced_topics/model_based_rl/dyna_q_explained.md)
- [ ] [シンプルな環境で世界モデルを訓練する](advanced_topics/model_based_rl/world_model_explained.md)
- [ ] [学習したモデルを計画に使用する](advanced_topics/model_based_rl/model_based_planning_explained.md)

**リソース:**
- 📖 Sutton & Barto - 第8章
- 📄 [World Models 論文](https://arxiv.org/abs/1803.10122)
- 📄 [MuZero 論文](https://arxiv.org/abs/1911.08265)
- 📄 [DreamerV3 論文](https://arxiv.org/abs/2301.04104)

---

### マルチエージェント強化学習 {#multi-agent-rl}
共有環境で複数のエージェントが同時に学習する。

**概念:**
- 独立学習者 (Independent learners)
- 集中訓練・分散実行 (CTDE)
- 自己対戦 (Self-play)
- 創発的なコミュニケーション

**実践ワーク:**
- [ ] [シンプルな行列ゲームでエージェントを訓練する](advanced_topics/multi_agent_rl/matrix_games_explained.md)
- [ ] [ボードゲーム用に自己対戦を実装する](advanced_topics/multi_agent_rl/self_play_tic_tac_toe_explained.md)
- [ ] [PettingZoo 環境を探索する](advanced_topics/multi_agent_rl/pettingzoo_explore_explained.md)

**リソース:**
- 📄 [マルチエージェント強化学習のサーベイ](https://arxiv.org/abs/1911.10635)
- 📄 [OpenAI Five](https://arxiv.org/abs/1912.06680)
- 💻 [PettingZoo ライブラリ](https://pettingzoo.farama.org/)

---

### オフライン/バッチ強化学習 {#offlinebatch-rl}
環境との相互作用なしに、固定データセットから学習する。

**概念:**
- 分布シフト問題
- 保守的 Q学習 (CQL)
- 暗黙的 Q学習 (IQL)
- Decision Transformer

**実践ワーク:**
- [ ] [D4RL ベンチマーク・データセットで訓練する](advanced_topics/offline_rl/d4rl_dataset_explained.md)
- [ ] [CQL を実装する](advanced_topics/offline_rl/cql_explained.md)
- [ ] [行動クローニングと比較する](advanced_topics/offline_rl/behavioral_cloning_explained.md)

**リソース:**
- 📄 [オフライン強化学習チュートリアル](https://arxiv.org/abs/2005.01643)
- 📄 [CQL 論文](https://arxiv.org/abs/2006.04779)
- 📄 [Decision Transformer](https://arxiv.org/abs/2106.01345)
- 💻 [D4RL ベンチマーク](https://github.com/Farama-Foundation/D4RL)

---

### 探索 {#exploration}
疎な報酬や困難な探索問題に対処する。

**概念:**
- 内発的動機付け
- 好奇心駆動の探索 (ICM)
- ランダム・ネットワーク蒸留 (RND)
- カウントベースの探索
- Go-Explore

**実践ワーク:**
- [ ] [好奇心ボーナスの実装](advanced_topics/exploration/curiosity_bonus_explained.md)
- [ ] [Montezuma's Revenge で訓練する](advanced_topics/exploration/montezuma_revenge_explained.md)
- [ ] [探索戦略を比較する](advanced_topics/exploration/compare_exploration_explained.md)

**リソース:**
- 📄 [ICM 論文](https://arxiv.org/abs/1705.05363)
- 📄 [RND 論文](https://arxiv.org/abs/1810.12894)
- 📄 [Go-Explore](https://arxiv.org/abs/1901.10995)
- 📄 [Deep Exploration via Bootstrapped DQN](https://arxiv.org/abs/1602.04621) — DeepSea タスクの出典

---

### 階層的強化学習 {#hierarchical-rl}
複数の時間的抽象化レベルで学習する。

**概念:**
- オプション・フレームワーク (Options framework)
- ゴール条件付き方策
- 封建的ネットワーク (Feudal networks)
- HIRO, HAM

**実践ワーク:**
- [ ] [オプション・クリティック・アーキテクチャを実装する](advanced_topics/hierarchical_rl/option_critic_explained.md)
- [ ] [ゴール条件付き方策を訓練する](advanced_topics/hierarchical_rl/goal_conditioned_policy_explained.md)
- [ ] [長期ホライゾンのタスクでテストする](advanced_topics/hierarchical_rl/long_horizon_tasks_explained.md)

**リソース:**
- 📖 Sutton & Barto - 第12章 (オプション)
- 📄 [Option-Critic 論文](https://arxiv.org/abs/1609.05140)
- 📄 [HIRO 論文](https://arxiv.org/abs/1805.08296)

---

### RLHF (人間からのフィードバックによる強化学習) {#rlhf-rl-from-human-feedback}
モデルを人間の好みに合わせる。

**概念:**
- 比較データからの報酬モデリング
- KL 制約付き方策最適化
- 憲法 AI (Constitutional AI)
- DPO (Direct Preference Optimization)

**実践ワーク:**
- [ ] [好みのデータから報酬モデルを訓練する](advanced_topics/rlhf/reward_modeling_explained.md)
- [ ] [PPO で小規模言語モデルをファインチューニングする](advanced_topics/rlhf/ppo_fine_tuning_explained.md)
- [ ] [DPO を実装する](advanced_topics/rlhf/dpo_implementation_explained.md)

**リソース:**
- 📄 [InstructGPT 論文](https://arxiv.org/abs/2203.02155)
- 📄 [Constitutional AI](https://arxiv.org/abs/2212.08073)
- 📄 [DPO 論文](https://arxiv.org/abs/2305.18290)
- 💻 [TRL ライブラリ](https://github.com/huggingface/trl)

---

## フェーズ 6: 研究レベル (継続的) {#phase-6-research-level-ongoing}

### 目標
この分野に独自の研究成果で貢献する。

### 活動
- NeurIPS, ICML, ICLR の論文を定期的に読む
- 最近の論文の主要な結果を再現する
- 未解決の問題や限界を特定する
- 適切な評価を伴う厳密な実験を行う
- サンプル効率、汎化性能、安全性を考慮する

### 学習すべき画期的な論文
- [ ] [AlphaGo (Silver et al., 2016)](https://www.nature.com/articles/nature16961)
- [ ] [AlphaZero (Silver et al., 2018)](https://www.science.org/doi/10.1126/science.aar6404)
- [ ] [MuZero (Schrittwieser et al., 2020)](https://arxiv.org/abs/1911.08265)
- [ ] [Decision Transformer (Chen et al., 2021)](https://arxiv.org/abs/2106.01345)
- [ ] [InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)
- [ ] [DreamerV3 (Hafner et al., 2023)](https://arxiv.org/abs/2301.04104)

### 発表の場
- NeurIPS, ICML, ICLR (トップ会議)
- AAAI, IJCAI
- CoRL (ロボティクス焦点)
- 初期の成果のためのワークショップ論文

---

## ツール概要

| フェーズ | 推奨ツール |
|-------|-------------------|
| 1-2 | NumPy, Matplotlib |
| 3 | PyTorch, Gymnasium, ALE-Py, W&B |
| 4 | Stable-Baselines3, MuJoCo/Box2D |
| 5-6 | カスタムコード, Ray/RLlib, JAX (高速化用) |

---

## 成功のためのヒント

1. **まずゼロから実装する** — ライブラリの使用は、アルゴリズムを理解してからにする
2. **シンプルな環境でデバッグする** — Atari の前に必ず CartPole で試す
3. **すべてをログに記録する** — 報酬、損失、勾配、エピソードの長さなど
4. **学習を可視化する** — 学習曲線をプロットし、エピソードをレンダリングする
5. **Sutton & Barto 本を読む** — これは強化学習の聖書です
6. **数学を理解する** — 少なくとも方策勾配定理とベルマン方程式は理解する
7. **忍耐強く取り組む** — 強化学習は不安定なことで有名です。失敗する実行は普通のことです
8. **シード値を使用する** — 再現性が重要です。複数のシードで平均をとってください
9. **コミュニティに参加する** — r/reinforcementlearning, RL Discord, Twitter/X

---

## 避けるべき一般的な落とし穴

- ❌ 基礎を飛ばして深層強化学習に飛び込む
- ❌ 観測値や報酬を正規化しない
- ❌ 学習率が大きすぎる、または小さすぎる
- ❌ テスト中に評価モードに設定し忘れる
- ❌ 実験に十分なシード値を使用しない
- ❌ 参照コードを確認せずに論文から実装する
- ❌ 1回の訓練失敗で諦めてしまう

---

## 用語集

| 用語 | 定義 |
|------|------------|
| **MDP** | マルコフ決定過程 - 強化学習の形式的なフレームワーク |
| **方策 (π)** | 状態から行動へのマッピング |
| **価値関数 (V)** | ある状態からの期待収益 |
| **Q関数** | ある状態と行動のペアからの期待収益 |
| **TD誤差** | 予測値とブートストラップされた値の差 |
| **GAE** | 一般化アドバンテージ推定 |
| **PPO** | 近接方策最適化 (Proximal Policy Optimization) |
| **RLHF** | 人間からのフィードバックによる強化学習 |

---

## ライセンス

このガイドは教育目的で提供されています。共有や改変は自由に行ってください。

---
