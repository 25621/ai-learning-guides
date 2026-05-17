# Double DQN: Corrigindo o Problema do Excesso de Confiança 🤔

## O Problema: O DQN Acha que é Melhor do que Realmente é

Imagine que lhe perguntem: "Qual é o melhor restaurante da cidade?"

Você pode dizer: "O Pizza Palace é incrível — é com certeza 10/10!". Mas você só esteve lá duas vezes. Você não sabe se ele é *realmente* 10/10. Você pode estar superestimando porque teve a sorte de comer uma pizza muito boa nessas duas visitas.

Este mesmo problema acontece com o DQN: o agente **superestima os valores Q**.

---

## Por que o DQN Superestima?

Quando o DQN calcula o alvo (target), ele faz:
> alvo = recompensa + γ × **max** Q(próximo_estado)

O `max` é o problema! Quando você escolhe o máximo de várias estimativas ruidosas, quase sempre escolhe aquela com o maior erro aleatório (viés para cima).

**Exemplo da vida real:** Você pede para 5 amigos adivinharem a altura de um prédio. Os palpites são: 40m, 38m, 45m (palpite sortudo!), 39m, 41m. A altura real é 40m. Se você usar o `max(palpites)` = 45m, você está bem longe! O máximo de palpites ruidosos é quase sempre uma superestimativa.

Ao longo de milhares de atualizações, o DQN continua treinando em direção a esses alvos superinflados, aprendendo que as coisas são melhores do que realmente são. Isso pode retardar o aprendizado ou fazer com que o agente tome decisões ruins por excesso de confiança.

---

## A Solução: Double DQN

O **Double DQN** (Hasselt et al., 2016) divide o `max` em duas etapas:

**Passo 1 — Qual ação?** Use a **rede online** (online network) para escolher a melhor ação:
> melhor_ação = argmax Q_online(próximo_estado)

**Passo 2 — Qual o seu valor?** Use a **rede alvo** (target network) para avaliar essa ação:
> alvo = recompensa + γ × Q_alvo(próximo_estado, melhor_ação)

```
Vanilla DQN:   alvo = r + γ × max_a Q_alvo(s', a)
                                 ↑ a mesma rede escolhe E avalia → enviesado

Double DQN:    melhor_a = argmax_a Q_online(s', a)   ← online escolhe
               alvo = r + γ × Q_alvo(s', melhor_a)   ← alvo avalia
                                 ↑ redes diferentes → menos enviesado
```

**Exemplo da vida real:** Em uma entrevista de emprego, você não deixa o candidato dar a nota do seu próprio teste de desempenho (esse é o problema do vanilla DQN!). Em vez disso, o candidato *indica* seu melhor trabalho, e um examinador *separado* o avalia. Duas pessoas diferentes = avaliação mais justa!

---

## Por que a Separação Ajuda?

As duas redes (online e alvo) têm pesos diferentes porque a rede alvo é atualizada com menos frequência. Elas têm "opiniões" diferentes sobre qual ação é a melhor.

Quando elas discordam:
- A Online diz: "A Ação A parece ótima!"
- A Alvo diz: "Na verdade, a Ação A é apenas razoável — vale cerca de 7, não 10"

Ao usar a estimativa de VALOR da rede alvo para a ação ESCOLHIDA pela rede online, obtemos um número mais honesto e menos inflado.

---

## Diferença no Código: Apenas uma Linha!

A única mudança no código do vanilla DQN para o double DQN está no cálculo do alvo:

```python
# Vanilla DQN:
q_next = target_net(s_next).max(dim=1).values

# Double DQN:
best_actions = online_net(s_next).argmax(dim=1, keepdim=True)   # escolhe com a online
q_next = target_net(s_next).gather(1, best_actions)              # avalia com a alvo
```

Apenas duas linhas mudam — mas o impacto na estabilidade e na precisão é significativo!

---

## O que a Comparação Mostra

Ao executar `double_dqn_cartpole.py`, você verá dois gráficos:

**Gráfico 1: Curvas de Aprendizado**
- Tanto o vanilla quanto o double DQN devem resolver o CartPole.
- O Double DQN geralmente converge mais rápido e de forma mais estável.
- O CartPole é simples o suficiente para que a diferença seja modesta; ela é muito mais dramática no Atari.

**Gráfico 2: Estimativas de Valor Q**
- Vanilla DQN: os valores Q sobem ao longo do tempo (superestimativa).
- Double DQN: os valores Q permanecem mais modestos e precisos.

O gráfico de superestimativa do valor Q é a visão principal — ele mostra o vanilla DQN aprendendo valores inflados que acabam prejudicando o desempenho.

---

## A Família de Melhorias do DQN

O Double DQN é apenas uma das muitas melhorias do vanilla DQN. O artigo "Rainbow" (2017) combinou 6 melhorias:

1. **Double DQN** (corrige a superestimativa) ← este script!
2. **Prioritized Replay** (aprende mais com experiências surpreendentes)
3. **Dueling Networks** (separa "quão bom é este estado?" de "qual é a melhor ação?")
4. **Multi-step returns** (olha mais para o futuro)
5. **Distributional RL** (aprende a distribuição completa dos retornos, não apenas a média)
6. **NoisyNets** (exploração aprendida em vez de [ε-greedy](../foundations/multi_armed_bandit_explained.md#the-epsilon-greedy-strategy))

O Rainbow combinou TODAS elas e alcançou o melhor desempenho no Atari de sua época!

---

## Vocabulário Chave

| Palavra | Significado |
|---------|-------------|
| **Superestimativa** | Os valores Q são maiores que os valores reais (excessivamente otimistas) |
| **Double DQN** | Usa a rede online para seleção de ações e a rede alvo para avaliação |
| **Desacoplamento** | Separar duas tarefas que eram feitas pela mesma rede |
| **Viés (Bias)** | Um erro sistemático em uma direção (sempre muito alto ou sempre muito baixo) |
| **Rainbow** | Uma variante do DQN que combina 6 melhorias para desempenho máximo |

---

## Resumo da Jornada da Fase 3

Você completou agora toda a progressão da Fase 3:

| Algoritmo | O que adiciona | Por que ajuda |
|-----------|----------------|----------------|
| Linear Q | Rede neural → fórmula simples | Lida com estados contínuos |
| Naive DQN | Rede neural completa | Aprende padrões complexos |
| + Replay buffer | Amostragem aleatória da memória | Quebra correlações |
| + Target network | Cópia congelada para os alvos | Estabiliza o "alvo" |
| Atari DQN | CNN + empilhamento de frames | Aprende a partir de pixels! |
| Double DQN | Separação entre escolher/avaliar | Reduz a superestimativa |

Cada etapa resolveu um problema específico. É assim que a pesquisa real funciona — uma melhoria cuidadosa de cada vez!
