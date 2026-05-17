# Pembelajaran Penguatan: Dari Pemula hingga Lanjutan

Panduan komprehensif untuk memahami dan membangun sistem pembelajaran penguatan (reinforcement learning), mulai dari konsep dasar hingga pemahaman tingkat riset.

---

## Ikhtisar

| Fase | Fokus | Durasi |
|-------|-------|----------|
| 1 | [Dasar-Dasar](#phase-1-foundations-2-4-weeks) | 2-4 minggu |
| 2 | [Metode Tabular](#phase-2-tabular-methods-3-4-weeks) | 3-4 minggu |
| 3 | [Aproksimasi Fungsi](#phase-3-function-approximation-3-4-weeks) | 3-4 minggu |
| 4 | [Metode Gradien Kebijakan](#phase-4-policy-gradient-methods-4-5-weeks) | 4-5 minggu |
| 5 | [Topik Lanjutan](#phase-5-advanced-topics-6-8-weeks) | 6-8 minggu |
| 6 | [Tingkat Riset](#phase-6-research-level-ongoing) | Berkelanjutan |

**Total Waktu hingga Mahir:** ~6 bulan

---

## Fase 1: Dasar-Dasar (2-4 minggu) {#phase-1-foundations-2-4-weeks}

### Tujuan
Memahami konsep inti tanpa matematika yang mendalam.

### Konsep yang Harus Dipelajari
- Agen, lingkungan, status (state), tindakan (action), imbalan (reward)
- Proses Keputusan Markov (Markov Decision Processes / MDP)
- Return dan faktor diskon (γ)
- Kebijakan (policy) vs fungsi nilai (value function)
- Tradeoff eksplorasi vs eksploitasi

### Pekerjaan Praktis
- [ ] [Implementasi masalah multi-armed bandit sederhana dari nol](foundations/multi_armed_bandit_explained.md)
- [ ] [Selesaikan Frozen Lake dengan kebijakan acak, amati hasilnya](foundations/frozen_lake_explained.md)
- [ ] [Visualisasikan fungsi nilai-status pada kisi sederhana](foundations/state_value_visualization_explained.md)

### Alat
- Python
- NumPy
- Matplotlib (untuk visualisasi)

### Sumber Daya
- 📖 [Sutton & Barto](http://incompleteideas.net/book/the-book.html) - Bab 1-3
- 🎥 [Kuliah David Silver](https://www.youtube.com/watch?v=2pWv7GOvuf0) - Kuliah 1-2
- 📝 [OpenAI Spinning Up - Pendahuluan](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)

### Pencapaian
Anda harus mampu menjelaskan formulasi masalah RL dan mengapa MDP merupakan kerangka kerja standar.

---

## Fase 2: Metode Tabular (3-4 minggu) {#phase-2-tabular-methods-3-4-weeks}

### Tujuan
Menguasai algoritma RL klasik di mana ruang status/tindakan cukup kecil untuk disimpan dalam tabel.

### Konsep yang Harus Dipelajari
- Pemrograman Dinamis (Dynamic Programming)
  - Evaluasi Kebijakan
  - Iterasi Kebijakan
  - Iterasi Nilai
- Metode Monte Carlo
  - First-visit vs every-visit MC
  - Kontrol MC dengan awal eksplorasi (exploring starts)
- Temporal Difference Learning
  - Prediksi TD(0)
  - SARSA (kontrol TD on-policy)
  - Q-learning (kontrol TD off-policy)
- Eligibility Traces dan TD(λ)

### Pekerjaan Praktis
- [ ] [Implementasi iterasi kebijakan untuk GridWorld](tabular_methods/policy_iteration_gridworld_explained.md)
- [ ] [Bangun agen Q-learning untuk Frozen Lake](tabular_methods/q_learning_frozen_lake_explained.md)
- [ ] [Implementasi SARSA untuk Cliff Walking](tabular_methods/sarsa_cliff_walking_explained.md)
- [ ] [Bandingkan perilaku SARSA vs Q-learning (jalur aman vs optimal)](tabular_methods/sarsa_vs_qlearning_explained.md)
- [ ] [Implementasi kontrol Monte Carlo untuk Blackjack](tabular_methods/monte_carlo_blackjack_explained.md)

### Wawasan Utama
Pahami diagram backup — diagram tersebut memperjelas bagaimana setiap algoritma memperbarui nilai.

### Alat
- NumPy
- Gymnasium (untuk lingkungan)

### Sumber Daya
- 📖 Sutton & Barto - Bab 4-7
- 🎥 Kuliah David Silver 3-5
- 💻 [Repositori RL Denny Britz](https://github.com/dennybritz/reinforcement-learning)

### Pencapaian
Implementasikan Q-learning dari nol dan selesaikan Frozen Lake dengan tingkat keberhasilan >70%.

---

## Fase 3: Aproksimasi Fungsi (3-4 minggu) {#phase-3-function-approximation-3-4-weeks}

### Tujuan
Meningkatkan skala RL melampaui ruang status kecil menggunakan aproksimator fungsi.

### Konsep yang Harus Dipelajari
- Mengapa metode tabular gagal pada skala besar
- Aproksimasi fungsi linear
- Jaringan saraf sebagai aproksimator fungsi
- Deep Q-Networks (DQN)
  - Experience replay
  - Jaringan target (target networks)
  - Pemotongan imbalan (reward clipping)
- Peningkatan DQN
  - Double DQN
  - Dueling DQN
  - Prioritized Experience Replay
  - Rainbow DQN

### Pekerjaan Praktis
- [ ] [Selesaikan CartPole dengan Q-learning linear](function_approximation/linear_q_cartpole_explained.md)
- [ ] [Implementasi DQN dari nol untuk CartPole](function_approximation/dqn_cartpole_explained.md)
- [ ] [Tambahkan experience replay, amati peningkatan stabilitas](function_approximation/dqn_experience_replay_explained.md)
- [ ] [Tambahkan jaringan target, bandingkan kurva pembelajaran](function_approximation/dqn_target_network_explained.md)
- [ ] [Latih DQN pada Atari Pong (gunakan ALE-Py)](function_approximation/dqn_atari_pong_explained.md)
- [ ] [Implementasi Double DQN, bandingkan dengan vanilla DQN](function_approximation/double_dqn_cartpole_explained.md)

### Wawasan Utama
"Tiga serangkai maut" (aproksimasi fungsi + bootstrapping + off-policy) menyebabkan ketidakstablian. Inovasi DQN mengatasi hal ini.

### Alat
- PyTorch atau TensorFlow
- Gymnasium
- ALE-Py (untuk Atari)
- Weights & Biases (untuk pelacakan eksperimen)

### Sumber Daya
- 📖 Sutton & Barto - Bab 9-11
- 📄 [Makalah DQN (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)
- 📄 [Makalah Rainbow](https://arxiv.org/abs/1710.02298)
- 💻 [Implementasi CleanRL](https://github.com/vwxyzjn/cleanrl)

### Pencapaian
Latih agen DQN yang mencapai imbalan positif pada Atari Pong.

---

## Fase 4: Metode Gradien Kebijakan (4-5 minggu) {#phase-4-policy-gradient-methods-4-5-weeks}

### Tujuan
Belajar mengoptimalkan kebijakan secara langsung tanpa menghitung fungsi nilai.

### Konsep yang Harus Dipelajari
- Teorema Gradien Kebijakan (Policy Gradient Theorem)
- Algoritma REINFORCE
- Teknik pengurangan variansi
  - Baseline
  - Reward-to-go
- Metode Actor-Critic
  - A2C (Advantage Actor-Critic)
  - A3C (Varian asinkron)
- Generalized Advantage Estimation (GAE)
- Metode Trust Region
  - TRPO (Trust Region Policy Optimization)
  - PPO (Proximal Policy Optimization)

### Pekerjaan Praktis
- [ ] [Implementasi REINFORCE untuk CartPole](policy_gradients/reinforce_cartpole_explained.md)
- [ ] [Tambahkan baseline, ukur pengurangan variansi](policy_gradients/reinforce_baseline_explained.md)
- [ ] [Bangun A2C untuk LunarLander](policy_gradients/a2c_lunarlander_explained.md)
- [ ] [Implementasi PPO dari nol](policy_gradients/ppo_scratch_explained.md)
- [ ] [Latih PPO pada kontrol kontinu (BipedalWalker atau MuJoCo)](policy_gradients/ppo_continuous_explained.md)
- [ ] [Bandingkan sensitivitas hiperparameter PPO](policy_gradients/ppo_hyperparams_explained.md)

### Wawasan Utama
PPO adalah kuda beban RL modern — pahami mekanisme clipping-nya secara mendalam.

### Alat
- PyTorch
- Gymnasium
- Stable-Baselines3 (untuk referensi)
- MuJoCo atau Box2D (untuk kontrol kontinu)

### Sumber Daya
- 📖 Sutton & Barto - Bab 13
- 📝 [OpenAI Spinning Up - Policy Gradients](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
- 📄 [Makalah PPO (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- 📄 [Makalah GAE](https://arxiv.org/abs/1506.02438)
- 🎥 [Deep RL Bootcamp Pieter Abbeel](https://sites.google.com/view/deep-rl-bootcamp/lectures)

### Pencapaian
Implementasikan PPO dan selesaikan BipedalWalker-v3 (imbalan > 300).

---

## Fase 5: Topik Lanjutan (6-8 minggu) {#phase-5-advanced-topics-6-8-weeks}

Pilih 2-3 bidang berdasarkan minat Anda.
- [Model-Based RL](#model-based-rl)
- [Multi-Agent RL](#multi-agent-rl)
- [Offline/Batch RL](#offlinebatch-rl)
- [Eksplorasi](#eksplorasi)
- [RL Hierarkis](#rl-hierarkis)
- [RLHF (RL dari Umpan Balik Manusia)](#rlhf-rl-dari-umpan-balik-manusia)

### RL Berbasis Model (Model-Based RL) {#model-based-rl}
Pelajari dinamika lingkungan untuk merencanakan atau menghasilkan pengalaman sintetis.

**Konsep:**
- Arsitektur Dyna
- World models
- Model Predictive Control (MPC)
- MuZero, Dreamer

**Pekerjaan Praktis:**
- [ ] [Implementasi Dyna-Q](advanced_topics/model_based_rl/dyna_q_explained.md)
- [ ] [Latih world model pada lingkungan sederhana](advanced_topics/model_based_rl/world_model_explained.md)
- [ ] [Gunakan model terpelajari untuk perencanaan](advanced_topics/model_based_rl/model_based_planning_explained.md)

**Sumber Daya:**
- 📖 Sutton & Barto - Bab 8
- 📄 [Makalah World Models](https://arxiv.org/abs/1803.10122)
- 📄 [Makalah MuZero](https://arxiv.org/abs/1911.08265)
- 📄 [Makalah DreamerV3](https://arxiv.org/abs/2301.04104)

---

### RL Multi-Agen (Multi-Agent RL) {#multi-agent-rl}
Beberapa agen belajar secara simultan di lingkungan bersama.

**Konsep:**
- Pembelajar independen
- Centralized training, decentralized execution (CTDE)
- Self-play
- Komunikasi darurat (emergent communication)

**Pekerjaan Praktis:**
- [ ] [Latih agen dalam matrix games sederhana](advanced_topics/multi_agent_rl/matrix_games_explained.md)
- [ ] [Implementasi self-play untuk permainan papan](advanced_topics/multi_agent_rl/self_play_tic_tac_toe_explained.md)
- [ ] [Jelajahi lingkungan PettingZoo](advanced_topics/multi_agent_rl/pettingzoo_explore_explained.md)

**Sumber Daya:**
- 📄 [Survei RL Multi-Agen](https://arxiv.org/abs/1911.10635)
- 📄 [OpenAI Five](https://arxiv.org/abs/1912.06680)
- 💻 [Pustaka PettingZoo](https://pettingzoo.farama.org/)

---

### RL Offline/Batch (Offline/Batch RL) {#offlinebatch-rl}
Belajar dari dataset tetap tanpa interaksi lingkungan.

**Konsep:**
- Masalah pergeseran distribusi (distribution shift)
- Conservative Q-Learning (CQL)
- Implicit Q-Learning (IQL)
- Decision Transformer

**Pekerjaan Praktis:**
- [ ] [Latih pada dataset benchmark D4RL](advanced_topics/offline_rl/d4rl_dataset_explained.md)
- [ ] [Implementasi CQL](advanced_topics/offline_rl/cql_explained.md)
- [ ] [Bandingkan dengan behavioral cloning](advanced_topics/offline_rl/behavioral_cloning_explained.md)

**Sumber Daya:**
- 📄 [Tutorial RL Offline](https://arxiv.org/abs/2005.01643)
- 📄 [Makalah CQL](https://arxiv.org/abs/2006.04779)
- 📄 [Decision Transformer](https://arxiv.org/abs/2106.01345)
- 💻 [Benchmark D4RL](https://github.com/Farama-Foundation/D4RL)

---

### Eksplorasi {#eksplorasi}
Menangani imbalan jarang dan masalah eksplorasi sulit.

**Konsep:**
- Motivasi intrinsik
- Eksplorasi berbasis keingintahuan (ICM)
- Random Network Distillation (RND)
- Eksplorasi berbasis hitungan
- Go-Explore

**Pekerjaan Praktis:**
- [ ] [Implementasi bonus keingintahuan](advanced_topics/exploration/curiosity_bonus_explained.md)
- [ ] [Latih pada Montezuma's Revenge](advanced_topics/exploration/montezuma_revenge_explained.md)
- [ ] [Bandingkan strategi eksplorasi](advanced_topics/exploration/compare_exploration_explained.md)

**Sumber Daya:**
- 📄 [Makalah ICM](https://arxiv.org/abs/1705.05363)
- 📄 [Makalah RND](https://arxiv.org/abs/1810.12894)
- 📄 [Go-Explore](https://arxiv.org/abs/1901.10995)
- 📄 [Deep Exploration via Bootstrapped DQN](https://arxiv.org/abs/1602.04621) — sumber tugas DeepSea

---

### RL Hierarkis {#rl-hierarkis}
Belajar pada beberapa tingkat abstraksi temporal.

**Konsep:**
- Kerangka kerja Options
- Kebijakan terkondisi-tujuan (goal-conditioned policies)
- Feudal networks
- HIRO, HAM

**Pekerjaan Praktis:**
- [ ] [Implementasi arsitektur option-critic](advanced_topics/hierarchical_rl/option_critic_explained.md)
- [ ] [Latih kebijakan terkondisi-tujuan](advanced_topics/hierarchical_rl/goal_conditioned_policy_explained.md)
- [ ] [Uji pada tugas cakrawala-panjang](advanced_topics/hierarchical_rl/long_horizon_tasks_explained.md)

**Sumber Daya:**
- 📖 Sutton & Barto - Bab 12 (Options)
- 📄 [Makalah Option-Critic](https://arxiv.org/abs/1609.05140)
- 📄 [Makalah HIRO](https://arxiv.org/abs/1805.08296)

---

### RLHF (RL dari Umpan Balik Manusia) {#rlhf-rl-dari-umpan-balik-manusia}
Menyelaraskan model dengan preferensi manusia.

**Konsep:**
- Pemodelan imbalan dari perbandingan
- Optimasi kebijakan dengan batasan KL
- AI Konstitusional
- DPO (Direct Preference Optimization)

**Pekerjaan Praktis:**
- [ ] [Latih model imbalan dari data preferensi](advanced_topics/rlhf/reward_modeling_explained.md)
- [ ] [Fine-tune LM kecil dengan PPO](advanced_topics/rlhf/ppo_fine_tuning_explained.md)
- [ ] [Implementasi DPO](advanced_topics/rlhf/dpo_implementation_explained.md)

**Sumber Daya:**
- 📄 [Makalah InstructGPT](https://arxiv.org/abs/2203.02155)
- 📄 [AI Konstitusional](https://arxiv.org/abs/2212.08073)
- 📄 [Makalah DPO](https://arxiv.org/abs/2305.18290)
- 💻 [Pustaka TRL](https://github.com/huggingface/trl)

---

## Fase 6: Tingkat Riset (Berkelanjutan) {#phase-6-research-level-ongoing}

### Tujuan
Memberikan kontribusi karya orisinal di bidang ini.

### Aktivitas
- Baca makalah dari NeurIPS, ICML, ICLR secara rutin
- Reproduksi hasil utama dari makalah terbaru
- Identifikasi masalah terbuka dan batasan
- Jalankan eksperimen yang ketat dengan evaluasi yang tepat
- Pertimbangkan efisiensi sampel, generalisasi, keamanan

### Makalah Penting untuk Dipelajari
- [ ] [AlphaGo (Silver et al., 2016)](https://www.nature.com/articles/nature16961)
- [ ] [AlphaZero (Silver et al., 2018)](https://www.science.org/doi/10.1126/science.aar6404)
- [ ] [MuZero (Schrittwieser et al., 2020)](https://arxiv.org/abs/1911.08265)
- [ ] [Decision Transformer (Chen et al., 2021)](https://arxiv.org/abs/2106.01345)
- [ ] [InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)
- [ ] [DreamerV3 (Hafner et al., 2023)](https://arxiv.org/abs/2301.04104)

### Tempat Publikasi
- NeurIPS, ICML, ICLR (tempat teratas)
- AAAI, IJCAI
- CoRL (fokus robotika)
- Makalah workshop untuk karya awal

---

## Ringkasan Alat

| Fase | Alat yang Direkomendasikan |
|-------|-------------------|
| 1-2 | NumPy, Matplotlib |
| 3 | PyTorch, Gymnasium, ALE-Py, W&B |
| 4 | Stable-Baselines3, MuJoCo/Box2D |
| 5-6 | Kode kustom, Ray/RLlib, JAX (untuk kecepatan) |

---

## Tips untuk Sukses

1. **Implementasi dari nol terlebih dahulu** — Gunakan pustaka hanya setelah Anda memahami algoritmanya
2. **Debug dengan lingkungan sederhana** — CartPole sebelum Atari, selalu
3. **Catat (log) semuanya** — Imbalan, loss, gradien, panjang episode
4. **Visualisasikan pembelajaran** — Plot kurva pembelajaran, render episode
5. **Baca buku Sutton & Barto** — Ini adalah kitab suci RL
6. **Pahami matematikanya** — Setidaknya teorema gradien kebijakan dan persamaan Bellman
7. **Bersabarlah** — RL dikenal tidak stabil; kegagalan menjalankan program adalah hal normal
8. **Gunakan seed** — Reproduksibilitas itu penting; rata-ratakan di beberapa seed
9. **Bergabunglah dengan komunitas** — r/reinforcementlearning, Discord RL, Twitter/X

---

## Kesalahan Umum yang Harus Dihindari

- ❌ Melewatkan dasar-dasar untuk langsung lompat ke deep RL
- ❌ Tidak menormalisasi observasi/imbalan
- ❌ Menggunakan tingkat pembelajaran yang terlalu besar/kecil
- ❌ Lupa mengatur mode evaluasi selama pengujian
- ❌ Tidak menggunakan cukup seed untuk eksperimen
- ❌ Mengimplementasikan dari makalah tanpa memeriksa kode referensi
- ❌ Menyerah setelah satu kegagalan pelatihan

---

## Glosarium

| Istilah | Definisi |
|------|------------|
| **MDP** | Proses Keputusan Markov - kerangka kerja formal untuk RL |
| **Kebijakan (π)** | Pemetaan dari status ke tindakan |
| **Fungsi Nilai (V)** | Imbalan yang diharapkan dari suatu status |
| **Fungsi-Q** | Imbalan yang diharapkan dari pasangan status-tindakan |
| **Kesalahan TD** | Selisih antara nilai prediksi dan nilai bootstrap |
| **GAE** | Generalized Advantage Estimation |
| **PPO** | Proximal Policy Optimization |
| **RLHF** | Pembelajaran Penguatan dari Umpan Balik Manusia |

---

## Lisensi

Panduan ini disediakan untuk tujuan pendidikan. Jangan ragu untuk membagikan dan mengadaptasinya.

---
