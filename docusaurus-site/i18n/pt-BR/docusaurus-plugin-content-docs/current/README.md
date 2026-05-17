# Aprendizado por Reforço: Do Iniciante ao Avançado

Um guia abrangente para entender e construir sistemas de aprendizado por reforço, desde conceitos fundamentais até o entendimento em nível de pesquisa.

---

## Visão Geral

| Fase | Foco | Duração |
|-------|-------|----------|
| 1 | [Fundamentos](#phase-1-foundations-2-4-weeks) | 2-4 semanas |
| 2 | [Métodos Tabulares](#phase-2-tabular-methods-3-4-weeks) | 3-4 semanas |
| 3 | [Aproximação de Funções](#phase-3-function-approximation-3-4-weeks) | 3-4 semanas |
| 4 | [Métodos de Gradiente de Política](#phase-4-policy-gradient-methods-4-5-weeks) | 4-5 semanas |
| 5 | [Tópicos Avançados](#phase-5-advanced-topics-6-8-weeks) | 6-8 semanas |
| 6 | [Nível de Pesquisa](#phase-6-research-level-ongoing) | Contínuo |

**Tempo Total para Proficiência:** ~6 meses

---

## Fase 1: Fundamentos (2-4 semanas) {#phase-1-foundations-2-4-weeks}

### Objetivo
Entender os conceitos fundamentais sem matemática profunda.

### Conceitos a Aprender
- Agente, ambiente, estado, ação, recompensa
- Processos de Decisão de Markov (MDP)
- Retorno e fator de desconto (γ)
- Política vs. função de valor
- Tradeoff entre exploração e explotação (exploration vs exploitation)

### Trabalho Prático
- [ ] [Implementar um problema simples de multi-armed bandit do zero](foundations/multi_armed_bandit_explained.md)
- [ ] [Resolver o Frozen Lake com política aleatória, observar resultados](foundations/frozen_lake_explained.md)
- [ ] [Visualizar funções de valor de estado em uma grade simples](foundations/state_value_visualization_explained.md)

### Ferramentas
- Python
- NumPy
- Matplotlib (para visualização)

### Recursos
- 📖 [Sutton & Barto](http://incompleteideas.net/book/the-book.html) - Capítulos 1-3
- 🎥 [David Silver Lectures](https://www.youtube.com/watch?v=2pWv7GOvuf0) - Aulas 1-2
- 📝 [OpenAI Spinning Up - Introdução](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)

### Marco (Milestone)
Você deve ser capaz de explicar a formulação do problema de RL e por que os MDPs são o framework padrão.

---

## Fase 2: Métodos Tabulares (3-4 semanas) {#phase-2-tabular-methods-3-4-weeks}

### Objetivo
Dominar algoritmos clássicos de RL onde os espaços de estado/ação são pequenos o suficiente para serem armazenados em tabelas.

### Conceitos a Aprender
- Programação Dinâmica
  - Avaliação de Política
  - Iteração de Política
  - Iteração de Valor
- Métodos de Monte Carlo
  - MC de primeira visita vs. todas as visitas
  - Controle MC com inícios exploratórios
- Aprendizado por Diferença Temporal (Temporal Difference Learning)
  - Predição TD(0)
  - SARSA (controle TD on-policy)
  - Q-learning (controle TD off-policy)
- Traços de Elegibilidade e TD(λ)

### Trabalho Prático
- [ ] [Implementar iteração de política para GridWorld](tabular_methods/policy_iteration_gridworld_explained.md)
- [ ] [Construir agente de Q-learning para Frozen Lake](tabular_methods/q_learning_frozen_lake_explained.md)
- [ ] [Implementar SARSA para Cliff Walking](tabular_methods/sarsa_cliff_walking_explained.md)
- [ ] [Comparar comportamento do SARSA vs Q-learning (caminhos seguros vs ideais)](tabular_methods/sarsa_vs_qlearning_explained.md)
- [ ] [Implementar controle de Monte Carlo para Blackjack](tabular_methods/monte_carlo_blackjack_explained.md)

### Percepção Chave
Entenda os diagramas de backup — eles esclarecem como cada algoritmo atualiza os valores.

### Ferramentas
- NumPy
- Gymnasium (para ambientes)

### Recursos
- 📖 Sutton & Barto - Capítulos 4-7
- 🎥 David Silver Lectures 3-5
- 💻 [Repositório de RL do Denny Britz](https://github.com/dennybritz/reinforcement-learning)

### Marco (Milestone)
Implementar Q-learning do zero e resolver o Frozen Lake com >70% de taxa de sucesso.

---

## Fase 3: Aproximação de Funções (3-4 semanas) {#phase-3-function-approximation-3-4-weeks}

### Objetivo
Escalar RL para além de pequenos espaços de estado usando aproximadores de funções.

### Conceitos a Aprender
- Por que os métodos tabulares falham em escala
- Aproximação de função linear
- Redes neurais como aproximadores de funções
- Deep Q-Networks (DQN)
  - Experience replay
  - Redes de alvo (Target networks)
  - Recorte de recompensa (Reward clipping)
- Melhorias no DQN
  - Double DQN
  - Dueling DQN
  - Prioritized Experience Replay
  - Rainbow DQN

### Trabalho Prático
- [ ] [Resolver CartPole com Q-learning linear](function_approximation/linear_q_cartpole_explained.md)
- [ ] [Implementar DQN do zero para CartPole](function_approximation/dqn_cartpole_explained.md)
- [ ] [Adicionar experience replay, observar melhoria na estabilidade](function_approximation/dqn_experience_replay_explained.md)
- [ ] [Adicionar rede de alvo, comparar curvas de aprendizado](function_approximation/dqn_target_network_explained.md)
- [ ] [Treinar DQN no Atari Pong (use ALE-Py)](function_approximation/dqn_atari_pong_explained.md)
- [ ] [Implementar Double DQN, comparar com DQN básico](function_approximation/double_dqn_cartpole_explained.md)

### Percepção Chave
A "tríade mortal" (aproximação de função + bootstrapping + off-policy) causa instabilidade. As inovações do DQN abordam isso.

### Ferramentas
- PyTorch ou TensorFlow
- Gymnasium
- ALE-Py (para Atari)
- Weights & Biases (para acompanhamento de experimentos)

### Recursos
- 📖 Sutton & Barto - Capítulos 9-11
- 📄 [Artigo do DQN (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)
- 📄 [Artigo do Rainbow](https://arxiv.org/abs/1710.02298)
- 💻 [Implementações CleanRL](https://github.com/vwxyzjn/cleanrl)

### Marco (Milestone)
Treinar um agente DQN que alcance recompensa positiva no Atari Pong.

---

## Fase 4: Métodos de Gradiente de Política (4-5 semanas) {#phase-4-policy-gradient-methods-4-5-weeks}

### Objetivo
Aprender a otimizar políticas diretamente sem computar funções de valor.

### Conceitos a Aprender
- Teorema do Gradiente de Política
- Algoritmo REINFORCE
- Técnicas de redução de variância
  - Baselines
  - Reward-to-go
- Métodos Actor-Critic
  - A2C (Advantage Actor-Critic)
  - A3C (Variante assíncrona)
- Generalized Advantage Estimation (GAE)
- Métodos de Região de Confiança (Trust Region)
  - TRPO (Trust Region Policy Optimization)
  - PPO (Proximal Policy Optimization)

### Trabalho Prático
- [ ] [Implementar REINFORCE para CartPole](policy_gradients/reinforce_cartpole_explained.md)
- [ ] [Adicionar baseline, medir redução de variância](policy_gradients/reinforce_baseline_explained.md)
- [ ] [Construir A2C para LunarLander](policy_gradients/a2c_lunarlander_explained.md)
- [ ] [Implementar PPO do zero](policy_gradients/ppo_scratch_explained.md)
- [ ] [Treinar PPO em controle contínuo (BipedalWalker ou MuJoCo)](policy_gradients/ppo_continuous_explained.md)
- [ ] [Comparar sensibilidade de hiperparâmetros do PPO](policy_gradients/ppo_hyperparams_explained.md)

### Percepção Chave
O PPO é a ferramenta de trabalho do RL moderno — entenda seu mecanismo de corte profundamente.

### Ferramentas
- PyTorch
- Gymnasium
- Stable-Baselines3 (para referência)
- MuJoCo ou Box2D (para controle contínuo)

### Recursos
- 📖 Sutton & Barto - Capítulo 13
- 📝 [OpenAI Spinning Up - Gradientes de Política](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
- 📄 [Artigo do PPO (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- 📄 [Artigo do GAE](https://arxiv.org/abs/1506.02438)
- 🎥 [Bootcamp de Deep RL de Pieter Abbeel](https://sites.google.com/view/deep-rl-bootcamp/lectures)

### Marco (Milestone)
Implementar PPO e resolver o BipedalWalker-v3 (recompensa > 300).

---

## Fase 5: Tópicos Avançados (6-8 semanas) {#phase-5-advanced-topics-6-8-weeks}

Escolha 2-3 áreas com base em seus interesses.
- [RL Baseado em Modelo (Model-Based)](#model-based-rl)
- [RL Multi-Agente](#multi-agent-rl)
- [RL Offline/Batch](#offlinebatch-rl)
- [Exploração](#exploration)
- [RL Hierárquico](#hierarchical-rl)
- [RLHF (RL a partir de Feedback Humano)](#rlhf-rl-from-human-feedback)

### RL Baseado em Modelo (Model-Based RL) {#model-based-rl}
Aprender a dinâmica do ambiente para planejar ou gerar experiência sintética.

**Conceitos:**
- Arquitetura Dyna
- Modelos de mundo (World models)
- Controle Preditivo de Modelo (MPC)
- MuZero, Dreamer

**Trabalho Prático:**
- [ ] [Implementar Dyna-Q](advanced_topics/model_based_rl/dyna_q_explained.md)
- [ ] [Treinar um modelo de mundo em um ambiente simples](advanced_topics/model_based_rl/world_model_explained.md)
- [ ] [Usar modelo aprendido para planejamento](advanced_topics/model_based_rl/model_based_planning_explained.md)

**Recursos:**
- 📖 Sutton & Barto - Capítulo 8
- 📄 [Artigo World Models](https://arxiv.org/abs/1803.10122)
- 📄 [Artigo MuZero](https://arxiv.org/abs/1911.08265)
- 📄 [Artigo DreamerV3](https://arxiv.org/abs/2301.04104)

---

### RL Multi-Agente (Multi-Agent RL) {#multi-agent-rl}
Múltiplos agentes aprendendo simultaneamente em ambientes compartilhados.

**Conceitos:**
- Aprendizes independentes
- Treinamento centralizado, execução descentralizada (CTDE)
- Self-play (Auto-jogo)
- Comunicação emergente

**Trabalho Prático:**
- [ ] [Treinar agentes em jogos de matriz simples](advanced_topics/multi_agent_rl/matrix_games_explained.md)
- [ ] [Implementar self-play para um jogo de tabuleiro](advanced_topics/multi_agent_rl/self_play_tic_tac_toe_explained.md)
- [ ] [Explorar ambientes PettingZoo](advanced_topics/multi_agent_rl/pettingzoo_explore_explained.md)

**Recursos:**
- 📄 [Pesquisa sobre RL Multi-Agente](https://arxiv.org/abs/1911.10635)
- 📄 [OpenAI Five](https://arxiv.org/abs/1912.06680)
- 💻 [Biblioteca PettingZoo](https://pettingzoo.farama.org/)

---

### RL Offline/Batch RL {#offlinebatch-rl}
Aprender a partir de conjuntos de dados fixos sem interação com o ambiente.

**Conceitos:**
- Problema do desvio de distribuição (distribution shift)
- Conservative Q-Learning (CQL)
- Implicit Q-Learning (IQL)
- Decision Transformer

**Trabalho Prático:**
- [ ] [Treinar em conjuntos de dados de benchmark D4RL](advanced_topics/offline_rl/d4rl_dataset_explained.md)
- [ ] [Implementar CQL](advanced_topics/offline_rl/cql_explained.md)
- [ ] [Comparar com clonagem comportamental (behavioral cloning)](advanced_topics/offline_rl/behavioral_cloning_explained.md)

**Recursos:**
- 📄 [Tutorial de RL Offline](https://arxiv.org/abs/2005.01643)
- 📄 [Artigo do CQL](https://arxiv.org/abs/2006.04779)
- 📄 [Decision Transformer](https://arxiv.org/abs/2106.01345)
- 💻 [Benchmark D4RL](https://github.com/Farama-Foundation/D4RL)

---

### Exploração (Exploration) {#exploration}
Abordar recompensas esparsas e problemas de exploração difícil.

**Conceitos:**
- Motivação intrínseca
- Exploração impulsionada pela curiosidade (ICM)
- Random Network Distillation (RND)
- Exploração baseada em contagem (count-based)
- Go-Explore

**Trabalho Prático:**
- [ ] [Implementar bônus de curiosidade](advanced_topics/exploration/curiosity_bonus_explained.md)
- [ ] [Treinar no Montezuma's Revenge](advanced_topics/exploration/montezuma_revenge_explained.md)
- [ ] [Comparar estratégias de exploração](advanced_topics/exploration/compare_exploration_explained.md)

**Recursos:**
- 📄 [Artigo do ICM](https://arxiv.org/abs/1705.05363)
- 📄 [Artigo do RND](https://arxiv.org/abs/1810.12894)
- 📄 [Go-Explore](https://arxiv.org/abs/1901.10995)
- 📄 [Exploração Profunda via DQN com Bootstrap](https://arxiv.org/abs/1602.04621) — fonte da tarefa DeepSea

---

### RL Hierárquico (Hierarchical RL) {#hierarchical-rl}
Aprender em múltiplos níveis de abstração temporal.

**Conceitos:**
- Framework de Opções (Options framework)
- Políticas condicionadas por objetivos (goal-conditioned)
- Redes Feudais (Feudal networks)
- HIRO, HAM

**Trabalho Prático:**
- [ ] [Implementar arquitetura option-critic](advanced_topics/hierarchical_rl/option_critic_explained.md)
- [ ] [Treinar política condicionada por objetivo](advanced_topics/hierarchical_rl/goal_conditioned_policy_explained.md)
- [ ] [Testar em tarefas de longo horizonte](advanced_topics/hierarchical_rl/long_horizon_tasks_explained.md)

**Recursos:**
- 📖 Sutton & Barto - Capítulo 12 (Opções)
- 📄 [Artigo Option-Critic](https://arxiv.org/abs/1609.05140)
- 📄 [Artigo HIRO](https://arxiv.org/abs/1805.08296)

---

### RLHF (RL a partir de Feedback Humano) {#rlhf-rl-from-human-feedback}
Alinhar modelos com as preferências humanas.

**Conceitos:**
- Modelagem de recompensa a partir de comparações
- Otimização de política com restrição KL
- IA Constitucional
- DPO (Direct Preference Optimization)

**Trabalho Prático:**
- [ ] [Treinar um modelo de recompensa a partir de dados de preferência](advanced_topics/rlhf/reward_modeling_explained.md)
- [ ] [Ajustar (fine-tune) um pequeno LM com PPO](advanced_topics/rlhf/ppo_fine_tuning_explained.md)
- [ ] [Implementar DPO](advanced_topics/rlhf/dpo_implementation_explained.md)

**Recursos:**
- 📄 [Artigo InstructGPT](https://arxiv.org/abs/2203.02155)
- 📄 [IA Constitucional](https://arxiv.org/abs/2212.08073)
- 📄 [Artigo do DPO](https://arxiv.org/abs/2305.18290)
- 💻 [Biblioteca TRL](https://github.com/huggingface/trl)

---

## Fase 6: Nível de Pesquisa (Contínuo) {#phase-6-research-level-ongoing}

### Objetivo
Contribuir com trabalhos originais para a área.

### Atividades
- Ler artigos do NeurIPS, ICML, ICLR regularmente
- Reproduzir resultados chave de artigos recentes
- Identificar problemas abertos e limitações
- Executar experimentos rigorosos com avaliação adequada
- Considerar eficiência de amostragem, generalização, segurança

### Artigos Históricos para Estudar
- [ ] [AlphaGo (Silver et al., 2016)](https://www.nature.com/articles/nature16961)
- [ ] [AlphaZero (Silver et al., 2018)](https://www.science.org/doi/10.1126/science.aar6404)
- [ ] [MuZero (Schrittwieser et al., 2020)](https://arxiv.org/abs/1911.08265)
- [ ] [Decision Transformer (Chen et al., 2021)](https://arxiv.org/abs/2106.01345)
- [ ] [InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)
- [ ] [DreamerV3 (Hafner et al., 2023)](https://arxiv.org/abs/2301.04104)

### Onde Publicar
- NeurIPS, ICML, ICLR (principais locais)
- AAAI, IJCAI
- CoRL (foco em robótica)
- Artigos de workshop para trabalhos iniciais

---

## Resumo de Ferramentas

| Fase | Ferramentas Recomendadas |
|-------|-------------------|
| 1-2 | NumPy, Matplotlib |
| 3 | PyTorch, Gymnasium, ALE-Py, W&B |
| 4 | Stable-Baselines3, MuJoCo/Box2D |
| 5-6 | Código personalizado, Ray/RLlib, JAX (para velocidade) |

---

## Dicas para o Sucesso

1. **Implemente do zero primeiro** — Use bibliotecas apenas depois de entender o algoritmo
2. **Depure com ambientes simples** — CartPole antes de Atari, sempre
3. **Registre tudo** — Recompensas, perdas, gradientes, comprimentos de episódios
4. **Visualize o aprendizado** — Plote curvas de aprendizado, renderize episódios
5. **Leia o livro do Sutton & Barto** — É a bíblia do RL
6. **Entenda a matemática** — Pelo menos o teorema do gradiente de política e as equações de Bellman
7. **Seja paciente** — RL é notoriamente instável; falhas são normais
8. **Use sementes (seeds)** — Reprodutibilidade importa; tire a média sobre várias sementes
9. **Junte-se a comunidades** — r/reinforcementlearning, RL Discord, Twitter/X

---

## Armadilhas Comuns a Evitar

- ❌ Pular os fundamentos para mergulhar direto em deep RL
- ❌ Não normalizar observações/recompensas
- ❌ Usar taxas de aprendizado muito grandes/pequenas
- ❌ Esquecer de definir o modo de avaliação durante os testes
- ❌ Não usar sementes suficientes para os experimentos
- ❌ Implementar a partir de artigos sem verificar o código de referência
- ❌ Desistir após uma falha no treinamento

---

## Glossário

| Termo | Definição |
|------|------------|
| **MDP** | Processo de Decisão de Markov - framework formal para RL |
| **Política (π)** | Mapeamento de estados para ações |
| **Função de Valor (V)** | Retorno esperado a partir de um estado |
| **Função-Q** | Retorno esperado a partir de um par estado-ação |
| **Erro TD** | Diferença entre o valor previsto e o valor por bootstrap |
| **GAE** | Generalized Advantage Estimation |
| **PPO** | Proximal Policy Optimization |
| **RLHF** | Aprendizado por Reforço a partir de Feedback Humano |

---

## Licença

Este guia é fornecido para fins educacionais. Sinta-se à vontade para compartilhar e adaptar.

---
