# Aprendizaje por Refuerzo (Reinforcement Learning): De Principiante a Avanzado

Una guía completa para comprender y construir sistemas de aprendizaje por refuerzo, desde los conceptos fundamentales hasta la comprensión a nivel de investigación.

---

## Descripción General

| Fase | Enfoque | Duración |
|-------|-------|----------|
| 1 | [Fundamentos](#phase-1-foundations-2-4-weeks) | 2-4 semanas |
| 2 | [Métodos Tabulares](#phase-2-tabular-methods-3-4-weeks) | 3-4 semanas |
| 3 | [Aproximación de Funciones](#phase-3-function-approximation-3-4-weeks) | 3-4 semanas |
| 4 | [Métodos de Gradiente de Política](#phase-4-policy-gradient-methods-4-5-weeks) | 4-5 semanas |
| 5 | [Temas Avanzados](#phase-5-advanced-topics-6-8-weeks) | 6-8 semanas |
| 6 | [Nivel de Investigación](#phase-6-research-level-ongoing) | Continuo |

**Tiempo total para la competencia:** ~6 meses

---

## Fase 1: Fundamentos (2-4 semanas) {#phase-1-foundations-2-4-weeks}

### Objetivo
Comprender los conceptos básicos sin matemáticas profundas.

### Conceptos a Aprender
- Agente, entorno, estado, acción, recompensa
- Procesos de Decisión de Markov (MDP)
- Retorno y factor de descuento (γ)
- Política vs. función de valor
- Dilema exploración vs. explotación

### Trabajo Práctico
- [ ] [Implementar un problema simple de bandido multibrazo desde cero](foundations/multi_armed_bandit_explained.md)
- [ ] [Resolver Frozen Lake con una política aleatoria, observar los resultados](foundations/frozen_lake_explained.md)
- [ ] [Visualizar funciones de valor de estado en una cuadrícula simple](foundations/state_value_visualization_explained.md)

### Herramientas
- Python
- NumPy
- Matplotlib (para visualización)

### Recursos
- 📖 [Sutton & Barto](http://incompleteideas.net/book/the-book.html) - Capítulos 1-3
- 🎥 [Lecciones de David Silver](https://www.youtube.com/watch?v=2pWv7GOvuf0) - Lecciones 1-2
- 📝 [OpenAI Spinning Up - Introducción](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)

### Hito
Deberías ser capaz de explicar la formulación del problema de RL y por qué los MDP son el marco estándar.

---

## Fase 2: Métodos Tabulares (3-4 semanas) {#phase-2-tabular-methods-3-4-weeks}

### Objetivo
Dominar los algoritmos clásicos de RL donde los espacios de estado/acción son lo suficientemente pequeños como para almacenarlos en tablas.

### Conceptos a Aprender
- Programación Dinámica
  - Evaluación de Políticas
  - Iteración de Políticas
  - Iteración de Valores
- Métodos de Monte Carlo
  - MC de primera visita vs. cada visita
  - Control MC con inicios exploratorios
- Aprendizaje por Diferencia Temporal (TD)
  - Predicción TD(0)
  - SARSA (control TD dentro de la política/on-policy)
  - Q-learning (control TD fuera de la política/off-policy)
- Trazas de Elegibilidad y TD(λ)

### Trabajo Práctico
- [ ] [Implementar iteración de políticas para GridWorld](tabular_methods/policy_iteration_gridworld_explained.md)
- [ ] [Construir un agente de Q-learning para Frozen Lake](tabular_methods/q_learning_frozen_lake_explained.md)
- [ ] [Implementar SARSA para Cliff Walking](tabular_methods/sarsa_cliff_walking_explained.md)
- [ ] [Comparar el comportamiento de SARSA vs. Q-learning (caminos seguros vs. óptimos)](tabular_methods/sarsa_vs_qlearning_explained.md)
- [ ] [Implementar el control de Monte Carlo para Blackjack](tabular_methods/monte_carlo_blackjack_explained.md)

### Idea Clave
Comprender los diagramas de respaldo (backup diagrams): aclaran cómo cada algoritmo actualiza los valores.

### Herramientas
- NumPy
- Gymnasium (para entornos)

### Recursos
- 📖 Sutton & Barto - Capítulos 4-7
- 🎥 Lecciones de David Silver 3-5
- 💻 [Repositorio de RL de Denny Britz](https://github.com/dennybritz/reinforcement-learning)

### Hito
Implementar Q-learning desde cero y resolver Frozen Lake con una tasa de éxito >70%.

---

## Fase 3: Aproximación de Funciones (3-4 semanas) {#phase-3-function-approximation-3-4-weeks}

### Objetivo
Escalar el RL más allá de los espacios de estado pequeños utilizando aproximadores de funciones.

### Conceptos a Aprender
- Por qué fallan los métodos tabulares a gran escala
- Aproximación de funciones lineales
- Redes neuronales como aproximadores de funciones
- Redes Q Profundas (DQN)
  - Búfer de repetición (Experience replay)
  - Redes objetivo (Target networks)
  - Recorte de recompensas (Reward clipping)
- Mejoras de DQN
  - Double DQN
  - Dueling DQN
  - Búfer de repetición priorizado (Prioritized Experience Replay)
  - Rainbow DQN

### Trabajo Práctico
- [ ] [Resolver CartPole con Q-learning lineal](function_approximation/linear_q_cartpole_explained.md)
- [ ] [Implementar DQN desde cero para CartPole](function_approximation/dqn_cartpole_explained.md)
- [ ] [Añadir búfer de repetición, observar la mejora de la estabilidad](function_approximation/dqn_experience_replay_explained.md)
- [ ] [Añadir red objetivo, comparar curvas de aprendizaje](function_approximation/dqn_target_network_explained.md)
- [ ] [Entrenar DQN en Atari Pong (usar ALE-Py)](function_approximation/dqn_atari_pong_explained.md)
- [ ] [Implementar Double DQN, comparar con DQN vainilla](function_approximation/double_dqn_cartpole_explained.md)

### Idea Clave
La "tríada mortal" (aproximación de funciones + bootstrapping + fuera de la política) causa inestabilidad. Las innovaciones de DQN abordan esto.

### Herramientas
- PyTorch o TensorFlow
- Gymnasium
- ALE-Py (para Atari)
- Weights & Biases (para seguimiento de experimentos)

### Recursos
- 📖 Sutton & Barto - Capítulos 9-11
- 📄 [Artículo de DQN (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)
- 📄 [Artículo de Rainbow](https://arxiv.org/abs/1710.02298)
- 💻 [Implementaciones de CleanRL](https://github.com/vwxyzjn/cleanrl)

### Hito
Entrenar un agente DQN que logre una recompensa positiva en Atari Pong.

---

## Fase 4: Métodos de Gradiente de Política (4-5 semanas) {#phase-4-policy-gradient-methods-4-5-weeks}

### Objetivo
Aprender a optimizar las políticas directamente sin calcular funciones de valor.

### Conceptos a Aprender
- Teorema del Gradiente de Política
- Algoritmo REINFORCE
- Técnicas de reducción de varianza
  - Líneas de base (Baselines)
  - Recompensa por alcanzar (Reward-to-go)
- Métodos Actor-Critic
  - A2C (Advantage Actor-Critic)
  - A3C (Variante asíncrona)
- Estimación de Ventaja Generalizada (GAE)
- Métodos de Región de Confianza
  - TRPO (Trust Region Policy Optimization)
  - PPO (Proximal Policy Optimization)

### Trabajo Práctico
- [ ] [Implementar REINFORCE para CartPole](policy_gradients/reinforce_cartpole_explained.md)
- [ ] [Añadir línea de base, medir la reducción de la varianza](policy_gradients/reinforce_baseline_explained.md)
- [ ] [Construir A2C para LunarLander](policy_gradients/a2c_lunarlander_explained.md)
- [ ] [Implementar PPO desde cero](policy_gradients/ppo_scratch_explained.md)
- [ ] [Entrenar PPO en control continuo (BipedalWalker o MuJoCo)](policy_gradients/ppo_continuous_explained.md)
- [ ] [Comparar la sensibilidad de los hiperparámetros de PPO](policy_gradients/ppo_hyperparams_explained.md)

### Idea Clave
PPO es la herramienta de trabajo del RL moderno; comprende profundamente su mecanismo de recorte (clipping).

### Herramientas
- PyTorch
- Gymnasium
- Stable-Baselines3 (como referencia)
- MuJoCo o Box2D (para control continuo)

### Recursos
- 📖 Sutton & Barto - Capítulo 13
- 📝 [OpenAI Spinning Up - Gradientes de Política](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
- 📄 [Artículo de PPO (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- 📄 [Artículo de GAE](https://arxiv.org/abs/1506.02438)
- 🎥 [Deep RL Bootcamp de Pieter Abbeel](https://sites.google.com/view/deep-rl-bootcamp/lectures)

### Hito
Implementar PPO y resolver BipedalWalker-v3 (recompensa > 300).

---

## Fase 5: Temas Avanzados (6-8 semanas) {#phase-5-advanced-topics-6-8-weeks}

Elige 2-3 áreas basadas en tus intereses.
- [RL Basado en Modelos](#model-based-rl)
- [RL Multiagente](#multi-agent-rl)
- [RL Offline/Por Lotes](#offlinebatch-rl)
- [Exploración](#exploration)
- [RL Jerárquico](#hierarchical-rl)
- [RLHF (RL a partir de retroalimentación humana)](#rlhf-rl-from-human-feedback)

### RL Basado en Modelos {#model-based-rl}
Aprender la dinámica del entorno para planificar o generar experiencia sintética.

**Conceptos:**
- Arquitectura Dyna
- Modelos de mundo (World models)
- Control Predictivo por Modelo (MPC)
- MuZero, Dreamer

**Trabajo Práctico:**
- [ ] [Implementar Dyna-Q](advanced_topics/model_based_rl/dyna_q_explained.md)
- [ ] [Entrenar un modelo de mundo en un entorno simple](advanced_topics/model_based_rl/world_model_explained.md)
- [ ] [Usar un modelo aprendido para la planificación](advanced_topics/model_based_rl/model_based_planning_explained.md)

**Recursos:**
- 📖 Sutton & Barto - Capítulo 8
- 📄 [Artículo de World Models](https://arxiv.org/abs/1803.10122)
- 📄 [Artículo de MuZero](https://arxiv.org/abs/1911.08265)
- 📄 [Artículo de DreamerV3](https://arxiv.org/abs/2301.04104)

---

### RL Multiagente {#multi-agent-rl}
Múltiples agentes aprendiendo simultáneamente en entornos compartidos.

**Conceptos:**
- Aprendices independientes
- Entrenamiento centralizado, ejecución descentralizada (CTDE)
- Auto-juego (Self-play)
- Comunicación emergente

**Trabajo Práctico:**
- [ ] [Entrenar agentes en juegos matriciales simples](advanced_topics/multi_agent_rl/matrix_games_explained.md)
- [ ] [Implementar auto-juego para un juego de mesa](advanced_topics/multi_agent_rl/self_play_tic_tac_toe_explained.md)
- [ ] [Explorar entornos PettingZoo](advanced_topics/multi_agent_rl/pettingzoo_explore_explained.md)

**Recursos:**
- 📄 [Encuesta sobre RL Multiagente](https://arxiv.org/abs/1911.10635)
- 📄 [OpenAI Five](https://arxiv.org/abs/1912.06680)
- 💻 [Biblioteca PettingZoo](https://pettingzoo.farama.org/)

---

### RL Offline/Por Lotes {#offlinebatch-rl}
Aprender de conjuntos de datos fijos sin interacción con el entorno.

**Conceptos:**
- Problema del desplazamiento de la distribución
- Q-Learning Conservador (CQL)
- Q-Learning Implícito (IQL)
- Decision Transformer

**Trabajo Práctico:**
- [ ] [Entrenar con conjuntos de datos benchmark D4RL](advanced_topics/offline_rl/d4rl_dataset_explained.md)
- [ ] [Implementar CQL](advanced_topics/offline_rl/cql_explained.md)
- [ ] [Comparar con la clonación de comportamiento](advanced_topics/offline_rl/behavioral_cloning_explained.md)

**Recursos:**
- 📄 [Tutorial de RL Offline](https://arxiv.org/abs/2005.01643)
- 📄 [Artículo de CQL](https://arxiv.org/abs/2006.04779)
- 📄 [Decision Transformer](https://arxiv.org/abs/2106.01345)
- 💻 [Benchmark D4RL](https://github.com/Farama-Foundation/D4RL)

---

### Exploración {#exploration}
Abordar la recompensa escasa y los problemas de exploración difícil.

**Conceptos:**
- Motivación intrínseca
- Exploración impulsada por la curiosidad (ICM)
- Destilación de Redes Aleatorias (RND)
- Exploración basada en el recuento
- Go-Explore

**Trabajo Práctico:**
- [ ] [Implementar bono de curiosidad](advanced_topics/exploration/curiosity_bonus_explained.md)
- [ ] [Entrenar en Montezuma's Revenge](advanced_topics/exploration/montezuma_revenge_explained.md)
- [ ] [Comparar estrategias de exploración](advanced_topics/exploration/compare_exploration_explained.md)

**Recursos:**
- 📄 [Artículo de ICM](https://arxiv.org/abs/1705.05363)
- 📄 [Artículo de RND](https://arxiv.org/abs/1810.12894)
- 📄 [Go-Explore](https://arxiv.org/abs/1901.10995)
- 📄 [Deep Exploration via Bootstrapped DQN](https://arxiv.org/abs/1602.04621) — fuente de la tarea DeepSea

---

### RL Jerárquico {#hierarchical-rl}
Aprender a múltiples niveles de abstracción temporal.

**Conceptos:**
- Marco de opciones (Options framework)
- Políticas condicionadas a objetivos
- Redes feudales
- HIRO, HAM

**Trabajo Práctico:**
- [ ] [Implementar la arquitectura option-critic](advanced_topics/hierarchical_rl/option_critic_explained.md)
- [ ] [Entrenar una política condicionada a objetivos](advanced_topics/hierarchical_rl/goal_conditioned_policy_explained.md)
- [ ] [Probar en tareas de horizonte largo](advanced_topics/hierarchical_rl/long_horizon_tasks_explained.md)

**Recursos:**
- 📖 Sutton & Barto - Capítulo 12 (Opciones)
- 📄 [Artículo de Option-Critic](https://arxiv.org/abs/1609.05140)
- 📄 [Artículo de HIRO](https://arxiv.org/abs/1805.08296)

---

### RLHF (RL a partir de retroalimentación humana) {#rlhf-rl-from-human-feedback}
Alinear modelos con las preferencias humanas.

**Conceptos:**
- Modelado de recompensas a partir de comparaciones
- Optimización de políticas con restricciones KL
- IA Constitucional
- DPO (Direct Preference Optimization)

**Trabajo Práctico:**
- [ ] [Entrenar un modelo de recompensa a partir de datos de preferencias](advanced_topics/rlhf/reward_modeling_explained.md)
- [ ] [Ajustar un LM pequeño con PPO](advanced_topics/rlhf/ppo_fine_tuning_explained.md)
- [ ] [Implementar DPO](advanced_topics/rlhf/dpo_implementation_explained.md)

**Recursos:**
- 📄 [Artículo de InstructGPT](https://arxiv.org/abs/2203.02155)
- 📄 [IA Constitucional](https://arxiv.org/abs/2212.08073)
- 📄 [Artículo de DPO](https://arxiv.org/abs/2305.18290)
- 💻 [Biblioteca TRL](https://github.com/huggingface/trl)

---

## Fase 6: Nivel de Investigación (Continuo) {#phase-6-research-level-ongoing}

### Objetivo
Contribuir con trabajo original al campo.

### Actividades
- Leer regularmente artículos de NeurIPS, ICML, ICLR.
- Reproducir resultados clave de artículos recientes.
- Identificar problemas abiertos y limitaciones.
- Realizar experimentos rigurosos con una evaluación adecuada.
- Considerar la eficiencia de las muestras, la generalización y la seguridad.

### Artículos Históricos para Estudiar
- [ ] [AlphaGo (Silver et al., 2016)](https://www.nature.com/articles/nature16961)
- [ ] [AlphaZero (Silver et al., 2018)](https://www.science.org/doi/10.1126/science.aar6404)
- [ ] [MuZero (Schrittwieser et al., 2020)](https://arxiv.org/abs/1911.08265)
- [ ] [Decision Transformer (Chen et al., 2021)](https://arxiv.org/abs/2106.01345)
- [ ] [InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)
- [ ] [DreamerV3 (Hafner et al., 2023)](https://arxiv.org/abs/2301.04104)

### Dónde publicar
- NeurIPS, ICML, ICLR (sedes principales)
- AAAI, IJCAI
- CoRL (enfoque en robótica)
- Artículos de taller (workshop) para trabajos tempranos

---

## Resumen de Herramientas

| Fase | Herramientas Recomendadas |
|-------|-------------------|
| 1-2 | NumPy, Matplotlib |
| 3 | PyTorch, Gymnasium, ALE-Py, W&B |
| 4 | Stable-Baselines3, MuJoCo/Box2D |
| 5-6 | Código personalizado, Ray/RLlib, JAX (para velocidad) |

---

## Consejos para el Éxito

1. **Implementar desde cero primero**: usa bibliotecas solo después de entender el algoritmo.
2. **Depurar con entornos simples**: CartPole antes que Atari, siempre.
3. **Registrar todo**: recompensas, pérdidas, gradientes, duración de los episodios.
4. **Visualizar el aprendizaje**: graficar curvas de aprendizaje, renderizar episodios.
5. **Leer el libro de Sutton & Barto**: es la biblia del RL.
6. **Entender las matemáticas**: al menos el teorema del gradiente de política y las ecuaciones de Bellman.
7. **Tener paciencia**: el RL es notoriamente inestable; las ejecuciones fallidas son normales.
8. **Usar semillas**: la reproducibilidad importa; promedia sobre múltiples semillas.
9. **Unirse a comunidades**: r/reinforcementlearning, RL Discord, Twitter/X.

---

## Errores Comunes que se Deben Evitar

- ❌ Saltarse los fundamentos para saltar directamente al RL profundo.
- ❌ No normalizar las observaciones/recompensas.
- ❌ Usar tasas de aprendizaje demasiado grandes o pequeñas.
- ❌ Olvidar configurar el modo de evaluación durante las pruebas.
- ❌ No usar suficientes semillas para los experimentos.
- ❌ Implementar a partir de artículos sin consultar el código de referencia.
- ❌ Rendirse tras una ejecución de entrenamiento fallida.

---

## Glosario

| Término | Definición |
|------|------------|
| **MDP** | Proceso de Decisión de Markov - marco formal para el RL |
| **Política (π)** | Mapeo de estados a acciones |
| **Función de Valor (V)** | Retorno esperado desde un estado |
| **Función Q** | Retorno esperado desde un par estado-acción |
| **Error TD** | Diferencia entre el valor predicho y el valor con bootstrap |
| **GAE** | Estimación de Ventaja Generalizada (Generalized Advantage Estimation) |
| **PPO** | Optimización de Política Próxima (Proximal Policy Optimization) |
| **RLHF** | Aprendizaje por Refuerzo a partir de Retroalimentación Humana (Reinforcement Learning from Human Feedback) |

---

## Licencia

Esta guía se proporciona con fines educativos. Siéntete libre de compartirla y adaptarla.

---
