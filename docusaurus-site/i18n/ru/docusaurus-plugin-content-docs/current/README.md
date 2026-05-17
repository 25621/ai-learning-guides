# Обучение с подкреплением: от новичка до эксперта

Всеобъемлющее руководство по пониманию и созданию систем обучения с подкреплением, от фундаментальных концепций до исследовательского уровня.

---

## Обзор

| Этап | Фокус | Продолжительность |
|-------|-------|----------|
| 1 | [Основы](#phase-1-foundations-2-4-weeks) | 2-4 недели |
| 2 | [Табличные методы](#phase-2-tabular-methods-3-4-weeks) | 3-4 недели |
| 3 | [Аппроксимация функций](#phase-3-function-approximation-3-4-weeks) | 3-4 недели |
| 4 | [Методы градиента политики](#phase-4-policy-gradient-methods-4-5-weeks) | 4-5 недель |
| 5 | [Продвинутые темы](#phase-5-advanced-topics-6-8-weeks) | 6-8 недель |
| 6 | [Исследовательский уровень](#phase-6-research-level-ongoing) | Постоянно |

**Общее время до мастерства:** ~6 месяцев

---

## Этап 1: Основы (2-4 недели) {#phase-1-foundations-2-4-weeks}

### Цель
Понять основные концепции без глубокой математики.

### Концепции для изучения
- Агент, среда, состояние, действие, награда
- Марковские процессы принятия решений (MDP)
- Доход (Return) и коэффициент дисконтирования (γ)
- Политика против функции ценности
- Компромисс между исследованием и использованием (Exploration vs exploitation)

### Практическая работа
- [ ] [Реализовать простую задачу о многоруком бандите с нуля](foundations/multi_armed_bandit_explained.md)
- [ ] [Решить задачу Frozen Lake со случайной политикой, наблюдать результаты](foundations/frozen_lake_explained.md)
- [ ] [Визуализировать функции ценности состояний на простой сетке](foundations/state_value_visualization_explained.md)

### Инструменты
- Python
- NumPy
- Matplotlib (для визуализации)

### Ресурсы
- 📖 [Саттон и Барто](http://incompleteideas.net/book/the-book.html) - Главы 1-3
- 🎥 [Лекции Дэвида Сильвера](https://www.youtube.com/watch?v=2pWv7GOvuf0) - Лекции 1-2
- 📝 [OpenAI Spinning Up - Введение](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)

### Результат
Вы должны уметь объяснить постановку задачи RL и почему MDP являются стандартной основой.

---

## Этап 2: Табличные методы (3-4 недели) {#phase-2-tabular-methods-3-4-weeks}

### Цель
Освоить классические алгоритмы RL, где пространства состояний и действий достаточно малы, чтобы хранить их в таблицах.

### Концепции для изучения
- Динамическое программирование
  - Итерация политики
  - Итерация ценности
  - Оценка политики
- Методы Монте-Карло
  - MC при первом посещении против MC при каждом посещении
  - MC-управление с исследующими стартами
- Обучение на основе временных различий (Temporal Difference Learning)
  - TD(0) предсказание
  - SARSA (on-policy TD управление)
  - Q-learning (off-policy TD управление)
- Следы пригодности (Eligibility Traces) и TD(λ)

### Практическая работа
- [ ] [Реализовать итерацию политики для GridWorld](tabular_methods/policy_iteration_gridworld_explained.md)
- [ ] [Создать Q-learning агента для Frozen Lake](tabular_methods/q_learning_frozen_lake_explained.md)
- [ ] [Реализовать SARSA для Cliff Walking](tabular_methods/sarsa_cliff_walking_explained.md)
- [ ] [Сравнить поведение SARSA и Q-learning (безопасные против оптимальных путей)](tabular_methods/sarsa_vs_qlearning_explained.md)
- [ ] [Реализовать управление Монте-Карло для Блэкджека](tabular_methods/monte_carlo_blackjack_explained.md)

### Ключевое понимание
Поймите диаграммы обновлений (backup diagrams) — они проясняют, как каждый алгоритм обновляет значения.

### Инструменты
- NumPy
- Gymnasium (для сред)

### Ресурсы
- 📖 Саттон и Барто - Главы 4-7
- 🎥 Лекции Дэвида Сильвера 3-5
- 💻 [Репозиторий Денни Бритца по RL](https://github.com/dennybritz/reinforcement-learning)

### Результат
Реализовать Q-learning с нуля и решить Frozen Lake с вероятностью успеха >70%.

---

## Этап 3: Аппроксимация функций (3-4 недели) {#phase-3-function-approximation-3-4-weeks}

### Цель
Масштабировать RL за пределы малых пространств состояний с помощью аппроксиматоров функций.

### Концепции для изучения
- Почему табличные методы не масштабируются
- Линейная аппроксимация функций
- Нейронные сети как аппроксиматоры функций
- Глубокие Q-сети (DQN)
  - Буфер воспроизведения опыта (Experience replay)
  - Целевые сети (Target networks)
  - Обрезка наград (Reward clipping)
- Улучшения DQN
  - Double DQN
  - Dueling DQN
  - Приоритизированное воспроизведение опыта (Prioritized Experience Replay)
  - Rainbow DQN

### Практическая работа
- [ ] [Решить CartPole с помощью линейного Q-learning](function_approximation/linear_q_cartpole_explained.md)
- [ ] [Реализовать DQN с нуля для CartPole](function_approximation/dqn_cartpole_explained.md)
- [ ] [Добавить воспроизведение опыта, наблюдать улучшение стабильности](function_approximation/dqn_experience_replay_explained.md)
- [ ] [Добавить целевую сеть, сравнить кривые обучения](function_approximation/dqn_target_network_explained.md)
- [ ] [Обучить DQN на Atari Pong (используйте ALE-Py)](function_approximation/dqn_atari_pong_explained.md)
- [ ] [Реализовать Double DQN, сравнить с обычным DQN](function_approximation/double_dqn_cartpole_explained.md)

### Ключевое понимание
"Смертельная триада" (аппроксимация функций + бутстрапинг + off-policy) вызывает нестабильность. Инновации DQN решают эту проблему.

### Инструменты
- PyTorch или TensorFlow
- Gymnasium
- ALE-Py (для Atari)
- Weights & Biases (для отслеживания экспериментов)

### Ресурсы
- 📖 Саттон и Барто - Главы 9-11
- 📄 [Статья о DQN (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)
- 📄 [Статья о Rainbow](https://arxiv.org/abs/1710.02298)
- 💻 [Реализации CleanRL](https://github.com/vwxyzjn/cleanrl)

### Результат
Обучить DQN агента, который достигает положительной награды в Atari Pong.

---

## Этап 4: Методы градиента политики (4-5 недель) {#phase-4-policy-gradient-methods-4-5-weeks}

### Цель
Научиться оптимизировать политики напрямую без вычисления функций ценности.

### Концепции для изучения
- Теорема о градиенте политики
- Алгоритм REINFORCE
- Методы снижения дисперсии
  - Базовые линии (Baselines)
  - Награда до конца (Reward-to-go)
- Методы Актёр-Критик (Actor-Critic)
  - A2C (Advantage Actor-Critic)
  - A3C (Asynchronous variant)
- Обобщенная оценка преимущества (GAE)
- Методы доверительной области
  - TRPO (Trust Region Policy Optimization)
  - PPO (Proximal Policy Optimization)

### Практическая работа
- [ ] [Реализовать REINFORCE для CartPole](policy_gradients/reinforce_cartpole_explained.md)
- [ ] [Добавить базовую линию, измерить снижение дисперсии](policy_gradients/reinforce_baseline_explained.md)
- [ ] [Построить A2C для LunarLander](policy_gradients/a2c_lunarlander_explained.md)
- [ ] [Реализовать PPO с нуля](policy_gradients/ppo_scratch_explained.md)
- [ ] [Обучить PPO на непрерывном управлении (BipedalWalker или MuJoCo)](policy_gradients/ppo_continuous_explained.md)
- [ ] [Сравнить чувствительность PPO к гиперпараметрам](policy_gradients/ppo_hyperparams_explained.md)

### Клювое понимание
PPO является основным рабочим инструментом современного RL — глубоко изучите его механизм обрезки (clipping).

### Инструменты
- PyTorch
- Gymnasium
- Stable-Baselines3 (для справки)
- MuJoCo или Box2D (для непрерывного управления)

### Ресурсы
- 📖 Саттон и Барто - Глава 13
- 📝 [OpenAI Spinning Up - Policy Gradients](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
- 📄 [Статья о PPO (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- 📄 [Статья о GAE](https://arxiv.org/abs/1506.02438)
- 🎥 [Deep RL Bootcamp Питера Аббила](https://sites.google.com/view/deep-rl-bootcamp/lectures)

### Результат
Реализовать PPO и решить BipedalWalker-v3 (награда > 300).

---

## Этап 5: Продвинутые темы (6-8 недель) {#phase-5-advanced-topics-6-8-weeks}

Выберите 2-3 области в зависимости от ваших интересов.
- [RL на основе моделей](#model-based-rl)
- [Мультиагентный RL](#multi-agent-rl)
- [Офлайн/Пакетный RL](#offlinebatch-rl)
- [Исследование](#exploration)
- [Иерархический RL](#hierarchical-rl)
- [RLHF (RL на основе обратной связи от человека)](#rlhf-rl-from-human-feedback)

### RL на основе моделей {#model-based-rl}
Изучение динамики среды для планирования или генерации синтетического опыта.

**Концепции:**
- Архитектура Dyna
- Модели мира (World models)
- Прогнозное модельное управление (MPC)
- MuZero, Dreamer

**Практическая работа:**
- [ ] [Реализовать Dyna-Q](advanced_topics/model_based_rl/dyna_q_explained.md)
- [ ] [Обучить модель мира в простой среде](advanced_topics/model_based_rl/world_model_explained.md)
- [ ] [Использовать обученную модель для планирования](advanced_topics/model_based_rl/model_based_planning_explained.md)

**Ресурсы:**
- 📖 Саттон и Барто - Глава 8
- 📄 [Статья о моделях мира](https://arxiv.org/abs/1803.10122)
- 📄 [Статья о MuZero](https://arxiv.org/abs/1911.08265)
- 📄 [Статья о DreamerV3](https://arxiv.org/abs/2301.04104)

---

### Мультиагентный RL {#multi-agent-rl}
Несколько агентов, обучающихся одновременно в общих средах.

**Концепции:**
- Независимые обучающиеся (Independent learners)
- Централизованное обучение, децентрализованное исполнение (CTDE)
- Самообучение (Self-play)
- Эмерджентная коммуникация

**Практическая работа:**
- [ ] [Обучить агентов в простых матричных играх](advanced_topics/multi_agent_rl/matrix_games_explained.md)
- [ ] [Реализовать самообучение для настольной игры](advanced_topics/multi_agent_rl/self_play_tic_tac_toe_explained.md)
- [ ] [Исследовать среды PettingZoo](advanced_topics/multi_agent_rl/pettingzoo_explore_explained.md)

**Ресурсы:**
- 📄 [Обзор мультиагентного RL](https://arxiv.org/abs/1911.10635)
- 📄 [OpenAI Five](https://arxiv.org/abs/1912.06680)
- 💻 [Библиотека PettingZoo](https://pettingzoo.farama.org/)

---

### Офлайн/Пакетный RL {#offlinebatch-rl}
Обучение на фиксированных наборах данных без взаимодействия со средой.

**Концепции:**
- Проблема сдвига распределения (Distribution shift)
- Conservative Q-Learning (CQL)
- Implicit Q-Learning (IQL)
- Decision Transformer

**Практическая работа:**
- [ ] [Обучение на наборах данных D4RL](advanced_topics/offline_rl/d4rl_dataset_explained.md)
- [ ] [Реализовать CQL](advanced_topics/offline_rl/cql_explained.md)
- [ ] [Сравнить с поведенческим клонированием (behavioral cloning)](advanced_topics/offline_rl/behavioral_cloning_explained.md)

**Ресурсы:**
- 📄 [Туториал по офлайн-RL](https://arxiv.org/abs/2005.01643)
- 📄 [Статья о CQL](https://arxiv.org/abs/2006.04779)
- 📄 [Decision Transformer](https://arxiv.org/abs/2106.01345)
- 💻 [Бенчмарк D4RL](https://github.com/Farama-Foundation/D4RL)

---

### Исследование {#exploration}
Решение проблем с редким вознаграждением и сложным исследованием.

**Концепции:**
- Внутренняя мотивация
- Исследование на основе любопытства (ICM)
- Дистилляция случайных сетей (RND)
- Исследование на основе подсчета (Count-based exploration)
- Go-Explore

**Практическая работа:**
- [ ] [Реализовать бонус любопытства](advanced_topics/exploration/curiosity_bonus_explained.md)
- [ ] [Обучить на Montezuma's Revenge](advanced_topics/exploration/montezuma_revenge_explained.md)
- [ ] [Сравнить стратегии исследования](advanced_topics/exploration/compare_exploration_explained.md)

**Ресурсы:**
- 📄 [Статья об ICM](https://arxiv.org/abs/1705.05363)
- 📄 [Статья о RND](https://arxiv.org/abs/1810.12894)
- 📄 [Go-Explore](https://arxiv.org/abs/1901.10995)
- 📄 [Deep Exploration via Bootstrapped DQN](https://arxiv.org/abs/1602.04621) — источник задачи DeepSea

---

### Иерархический RL {#hierarchical-rl}
Обучение на нескольких уровнях временной абстракции.

**Концепции:**
- Фреймворк опций (Options framework)
- Политики с условием на цель (Goal-conditioned policies)
- Феодальные сети (Feudal networks)
- HIRO, HAM

**Практическая работа:**
- [ ] [Реализовать архитектуру option-critic](advanced_topics/hierarchical_rl/option_critic_explained.md)
- [ ] [Обучить политику с условием на цель](advanced_topics/hierarchical_rl/goal_conditioned_policy_explained.md)
- [ ] [Тестирование на задачах с длинным горизонтом](advanced_topics/hierarchical_rl/long_horizon_tasks_explained.md)

**Ресурсы:**
- 📖 Саттон и Барто - Глава 12 (Опции)
- 📄 [Статья об Option-Critic](https://arxiv.org/abs/1609.05140)
- 📄 [Статья о HIRO](https://arxiv.org/abs/1805.08296)

---

### RLHF (RL на основе обратной связи от человека) {#rlhf-rl-from-human-feedback}
Согласование моделей с человеческими предпочтениями.

**Концепции:**
- Моделирование вознаграждения на основе сравнений
- Оптимизация политики с ограничением KL
- Конституционный ИИ (Constitutional AI)
- DPO (Direct Preference Optimization)

**Практическая работа:**
- [ ] [Обучить модель вознаграждения на основе данных о предпочтениях](advanced_topics/rlhf/reward_modeling_explained.md)
- [ ] [Дообучить небольшую языковую модель с помощью PPO](advanced_topics/rlhf/ppo_fine_tuning_explained.md)
- [ ] [Реализовать DPO](advanced_topics/rlhf/dpo_implementation_explained.md)

**Ресурсы:**
- 📄 [Статья об InstructGPT](https://arxiv.org/abs/2203.02155)
- 📄 [Конституционный ИИ](https://arxiv.org/abs/2212.08073)
- 📄 [Статья о DPO](https://arxiv.org/abs/2305.18290)
- 💻 [Библиотека TRL](https://github.com/huggingface/trl)

---

## Этап 6: Исследовательский уровень (Постоянно) {#phase-6-research-level-ongoing}

### Цель
Вносить оригинальный вклад в область.

### Активности
- Регулярно читать статьи с NeurIPS, ICML, ICLR
- Воспроизводить ключевые результаты из недавних статей
- Выявлять открытые проблемы и ограничения
- Проводить строгие эксперименты с надлежащей оценкой
- Учитывать эффективность выборки, обобщение, безопасность

### Ключевые статьи для изучения
- [ ] [AlphaGo (Silver et al., 2016)](https://www.nature.com/articles/nature16961)
- [ ] [AlphaZero (Silver et al., 2018)](https://www.science.org/doi/10.1126/science.aar6404)
- [ ] [MuZero (Schrittwieser et al., 2020)](https://arxiv.org/abs/1911.08265)
- [ ] [Decision Transformer (Chen et al., 2021)](https://arxiv.org/abs/2106.01345)
- [ ] [InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)
- [ ] [DreamerV3 (Hafner et al., 2023)](https://arxiv.org/abs/2301.04104)

### Где публиковаться
- NeurIPS, ICML, ICLR (ведущие площадки)
- AAAI, IJCAI
- CoRL (фокус на робототехнике)
- Статьи для воркшопов для ранних работ

---

## Резюме инструментов

| Этап | Рекомендуемые инструменты |
|-------|-------------------|
| 1-2 | NumPy, Matplotlib |
| 3 | PyTorch, Gymnasium, ALE-Py, W&B |
| 4 | Stable-Baselines3, MuJoCo/Box2D |
| 5-6 | Собственный код, Ray/RLlib, JAX (для скорости) |

---

## Советы для успеха

1. **Реализуйте с нуля в первую очередь** — используйте библиотеки только после того, как поймете алгоритм
2. **Отлаживайте в простых средах** — всегда сначала CartPole, потом Atari
3. **Логируйте всё** — награды, потери, градиенты, длину эпизодов
4. **Визуализируйте обучение** — стройте кривые обучения, рендерите эпизоды
5. **Прочитайте книгу Саттона и Барто** — это библия RL
6. **Поймите математику** — как минимум теорему о градиенте политики и уравнения Беллмана
7. **Будьте терпеливы** — RL печально известен своей нестабильностью; неудачные запуски — это нормально
8. **Используйте сиды (seeds)** — воспроизводимость имеет значение; усредняйте по нескольким сидам
9. **Вступайте в сообщества** — r/reinforcementlearning, RL Discord, Twitter/X

---

## Распространенные ошибки

- ❌ Пропуск основ и немедленный переход к глубокому RL
- ❌ Отсутствие нормализации наблюдений/наград
- ❌ Использование слишком больших/маленьких скоростей обучения (learning rates)
- ❌ Забывание установить режим оценки (evaluation mode) во время тестирования
- ❌ Использование недостаточного количества сидов для экспериментов
- ❌ Реализация из статей без проверки эталонного кода
- ❌ Сдача после одного неудачного сеанса обучения

---

## Глоссарий

| Термин | Определение |
|------|------------|
| **MDP** | Марковский процесс принятия решений - формальная основа для RL |
| **Политика (π)** | Отображение состояний в действия |
| **Функция ценности (V)** | Ожидаемый доход из состояния |
| **Q-функция** | Ожидаемый доход от пары состояние-действие |
| **TD-ошибка** | Разница между предсказанным и бутстрапированным значением |
| **GAE** | Обобщенная оценка преимущества |
| **PPO** | Проксимальная оптимизация политики |
| **RLHF** | Обучение с подкреплением на основе обратной связи от человека |

---

## Лицензия

Это руководство предоставляется в образовательных целях. Не стесняйтесь делиться и адаптировать.

---
