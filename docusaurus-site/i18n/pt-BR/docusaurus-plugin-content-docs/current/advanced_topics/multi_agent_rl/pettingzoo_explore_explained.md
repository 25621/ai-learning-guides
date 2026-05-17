# Explorando os Ambientes PettingZoo 🦓

## O que é PettingZoo?

Se você já fez RL de agente único, provavelmente usou o **Gymnasium** (o sucessor do OpenAI Gym). Todo ambiente parece igual: `env.reset()`, `env.step(action) → obs, reward, done, info` — uma nova *observação* do mundo, um sinal de *recompensa* escalar, uma flag *done* dizendo "fim de jogo" e um dicionário *info* para extras de depuração. Essa uniformidade é o que faz as bibliotecas de RL funcionarem.

**PettingZoo** é exatamente a mesma ideia, mas para *múltiplos agentes*. É um "zoológico" de ambientes multiagente — todos por trás de uma API bem definida:
- **Problemas clássicos de brinquedo**: ambientes simples como Pedra-Papel-Tesoura para testar algoritmos básicos.
- **Mundos de grade cooperativos**: agentes navegando em uma grade para alcançar um objetivo compartilhado.
- **Atari multiplayer**: jogos competitivos clássicos como Pong.
- **MPE (Multi-Particle Environment)**: ambientes físicos de espaço contínuo para coordenação e competição complexas.

Se você conseguir escrever um código que funcione em um ambiente PettingZoo, poderá conectá-lo a qualquer outro com quase nenhuma alteração.

---

## Os Dois Estilos de API

As configurações multiagente são mais complexas do que as de agente único porque dois agentes podem agir ao mesmo tempo, ou por turnos, ou mesmo em ordens arbitrárias. O PettingZoo resolve isso com duas APIs paralelas:

### 1) AEC (Agent-Environment-Cycle)

Um agente age por vez. O ambiente percorre os agentes em alguma ordem, e cada um recebe:
- uma **observação** — o que eles veem *agora mesmo*,
- uma **recompensa** — o retorno obtido pela ação *conjunta* na última rodada completa (ou seja, o que aconteceu como resultado da ação de *todos* os agentes, não apenas a sua; em um jogo de xadrez, por exemplo, sua recompensa reflete o estado do tabuleiro após o último movimento do seu oponente, não apenas o seu),
- uma **flag de terminação** — `True` quando o episódio termina *naturalmente* (ex: xeque-mate, alguém vence),
- uma **flag de truncamento** — `True` quando o episódio é *encerrado prematuramente* por um limite de tempo antes de um final natural ser alcançado.

Isso é natural para **jogos por turnos** como xadrez, Go, pôquer.

```python
env.reset()
for agent in env.agent_iter():
    obs, reward, term, trunc, info = env.last()
    if term or trunc:
        env.step(None)
        continue
    action = minha_politica(obs, agent)
    env.step(action)
```

### 2) Parallel (Paralelo)

Todos os agentes observam e agem simultaneamente a cada passo. `step()` recebe um *dicionário* de ações e retorna dicionários de observações e recompensas.

Isso é natural para **jogos em tempo real** como MPE (Multi-Particle Environments, onde todos os agentes-ponto se movem simultaneamente) ou mundos de grade multiagente.

```python
obs, info = env.reset()
while env.agents:
    actions = {a: minha_politica(obs[a]) for a in env.agents}
    obs, rewards, terms, truncs, info = env.step(actions)
```

Os dois estilos são **isomórficos** — estruturalmente equivalentes e interconversíveis: qualquer ambiente AEC pode ser automaticamente envolvido (wrapped) para parecer um Paralelo, e vice-versa. O PettingZoo fornece os wrappers de conversão para que você só precise escrever código para um estilo.

---

## Analogia da Vida Real

- **AEC = uma noite de jogos de tabuleiro.** "Vez da Alice. Agora do Bob. Agora da Carol. De volta para a Alice." Quem move em seguida vê o estado mais recente do tabuleiro.
- **Parallel = um videogame multiplayer.** Todos os quatro jogadores estão pressionando botões simultaneamente; o jogo atualiza o mundo 60 vezes por segundo.
- **Por que APIs uniformes importam.** Imagine se cada videogame multiplayer precisasse de seu próprio joystick. PettingZoo é o "joystick universal" do MARL.

---

## O que Nosso Código Faz

Construímos um ambiente **estilo PettingZoo** do zero: o **Jogo de Coordenação Iterado**. Dois agentes escolhem repetidamente o canal `0` ou `1`:

- Mesma escolha → ambos recebem +1
- Escolha diferente → ambos recebem -1

A **observação** que cada agente recebe é a *ação conjunta* anterior — o que ambos os agentes escolheram na rodada passada, empacotado em um único número inteiro. Concretamente: a última ação de cada agente é uma de `{início, 0, 1}` (3 estados), então o par é codificado como `3 × estado_agente_1 + estado_agente_2`, resultando em 9 inteiros possíveis (0 – 8). O inteiro 0 é o estado de "início" — sinaliza que nenhuma ação foi tomada ainda (o início de um episódio). Um episódio dura 25 passos, então o retorno total máximo é +25 por agente e o mínimo é -25. **O jogo aleatório pontua ≈ 0** porque a cada passo dois agentes aleatórios independentes escolhem 0 ou 1 com igual probabilidade: eles coincidem 50% das vezes (+1) e divergem 50% das vezes (-1), resultando em uma recompensa esperada por passo de 0,5 × (+1) + 0,5 × (-1) = **0**. Somado ao longo de 25 passos, o retorno esperado do episódio também é 0.

Em seguida:

1. **Demonstramos a interface AEC** com uma execução aleatória — isso confirma o loop AEC básico: `agent_iter()` fornece o agente cuja vez é a atual, `last()` lê a observação atual e a recompensa acumulada desse agente, e `step()` entrega a ação escolhida de volta ao ambiente.
2. **Treinamos dois Q-learners independentes através da interface Parallel**. Cada agente mantém sua própria tabela Q indexada pela **observação da ação conjunta** (o único inteiro que codifica o que *ambos* os agentes fizeram na rodada passada), para que ele possa aprender "quando nós dois escolhemos 0 da última vez, eu devo escolher 0 novamente".
3. **Tentamos importar a biblioteca `pettingzoo` real** e executar um de seus ambientes integrados (Pedra-Papel-Tesoura) com uma política aleatória. Se o PettingZoo não estiver instalado, pulamos esta etapa com uma mensagem amigável.

### O que você deve ver

| Estágio | Esperado |
|-------|----------|
| Execução aleatória (AEC) | Retorno médio do episódio perto de **0** — agentes aleatórios escolhem canais de forma independente, coincidindo e divergindo em medidas aproximadamente iguais. |
| Q-learners independentes (Parallel) — primeiros 100 eps | Cerca de **0** — ainda predominantemente aleatório enquanto os agentes exploram. |
| Q-learners independentes — últimos 100 eps | Fortemente positivo, **+20 a +25** — **a coordenação emergiu**: ambos os agentes aprenderam a escolher confiavelmente o mesmo canal em cada rodada. |

O gráfico `outputs/pettingzoo_coordination.png` mostra os retornos individuais dos episódios (cinza) e uma curva de **Média** móvel (azul). A média suaviza os episódios ruidosos para que você possa ver a tendência: os agentes passam de um jogo aleatório descoordenado perto de ~0 para uma **coordenação** estável perto de ~+25. A linha verde tracejada marca o teto de coordenação perfeita.

Se o `pettingzoo` estiver instalado, o script também executa `pettingzoo.classic.rps_v2` para provar que o script funciona com a biblioteca real exatamente da mesma forma que funciona em nosso ambiente artesanal. Para habilitar essa seção:

```bash
source ../../venv/bin/activate
pip install "pettingzoo[classic]"
```

---

## Por que Construir um Ambiente Personalizado Primeiro?

Porque **a API é a lição.** (Entender como estruturar a interação entre múltiplos agentes e o ambiente é mais importante do que as regras específicas do jogo.) O RL multiagente tem muitos sabores (por turnos, tempo real, cooperativo, competitivo, misto), e todos se encaixam no padrão AEC / Parallel. Uma vez que você tenha implementado esses dois loops, cada ambiente do PettingZoo é apenas uma questão de conectar um construtor de ambiente diferente — o código de treinamento permanece o mesmo.

Isso é exatamente como o Gymnasium mudou o RL de agente único: transformando o ambiente em uma caixa preta atrás de uma interface uniforme.

---

## Onde o Q-learning Independente Ajuda e Prejudica

Jogos de coordenação são *generosos* — os agentes compartilham o sinal de recompensa, então seus interesses estão alinhados. Aprendizes independentes podem resolver isso alegremente porque qualquer melhoria de um agente ajuda o outro.

Em jogos **adversários** (Pedra-Papel-Tesoura), o Q-learning independente oscila para sempre (conforme um agente se adapta, o outro muda sua estratégia para contra-atacar, levando a uma perseguição infinita).
Em jogos **parcialmente observáveis**, ele não consegue aprender de forma alguma porque a "observação" é apenas uma peça do estado (un agente pode ser penalizado por uma boa ação apenas porque não conseguiu ver o que o outro agente estava fazendo). O PettingZoo inclui ambos os tipos de ambiente para que você possa ver esses modos de falha por si mesmo.

---

## Palavras-Chave para Lembrar

| Palavra | Significado |
|------|---------|
| **PettingZoo** | O Gymnasium do RL multiagente — uma biblioteca de ambientes MARL padronizados |
| **AEC** | Agent-Environment-Cycle: um agente age por passo (por turnos) |
| **Parallel API** | Todos os agentes agem simultaneamente a cada passo |
| **MPE** | Multi-Particle Environment, um banco de testes cooperativo/competitivo popular incluído no PettingZoo (geralmente envolvendo pontos se movendo em tarefas baseadas em física). |
| **CTDE** | Centralised Training, Decentralised Execution — treinar com uma visão global (acesso a todos os estados), implantar apenas com observações locais (cada agente age em sua própria visão limitada). |
| **Independent Q-learning** | Cada agente executa um Q-learning comum (o algoritmo padrão, não modificado), ignorando que outros aprendizes existem. |

---

## Resumo em Uma Frase

> **O PettingZoo dá a cada ambiente multiagente a mesma forma — para que o código que você escreve hoje ainda funcione amanhã em um jogo totalmente diferente.**

Uma vez que os dois estilos de API sejam naturais para você, você poderá avançar para MADDPG (crítico centralizado para agentes de controle contínuo), QMIX (mistura de valores para equipes cooperativas), MAPPO (PPO multiagente) ou qualquer outro algoritmo MARL moderno — o lado do ambiente do seu código nunca precisará mudar.
