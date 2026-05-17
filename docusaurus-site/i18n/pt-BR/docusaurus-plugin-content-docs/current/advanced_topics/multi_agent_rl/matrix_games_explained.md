# Jogos Matriciais: O Mundo Multiagente Mais Simples 🎲

## O Que É um Jogo Matricial?

Imagine que você e um amigo escolhem, cada um, um sinal de mão — **pedra, papel ou tesoura** — *ao mesmo tempo*. Vocês não veem a escolha um do outro. O vencedor é decidido por uma pequena tabela:

|          | Pedra | Papel | Tesoura |
|----------|:-----:|:-----:|:-------:|
| Pedra    |  0,0  | -1,+1 | +1,-1   |
| Papel    | +1,-1 |  0,0  | -1,+1   |
| Tesoura  | -1,+1 | +1,-1 |  0,0    |

Essa tabela é o *mundo inteiro* do jogo. Sem movimento, sem tempo, sem mapa. Apenas uma decisão única. Chamamos isso de **jogo matricial** (matrix game) porque a matriz de pagamentos (payoff matrix) é o ambiente completo.

Os jogos matriciais são o lugar mais limpo para estudar o **RL multiagente**, porque a única coisa que pode mudar durante o treinamento é a *política* de cada jogador — a probabilidade de escolher cada ação.

---

## Por Que É "Multiagente"

No RL de agente único, o ambiente é fixo: o vento sempre sopra na mesma direção, o chão nunca se move. O agente melhora e eventualmente vence.

Em um jogo matricial, seu "ambiente" é *outro agente que aprende*. Conforme ele fica mais inteligente, o que conta como uma boa jogada para você *muda*. Isso é chamado de **não estacionariedade**, e é o problema central do RL multiagente.

> Se você continuar jogando Pedra, seu oponente acabará começando a jogar sempre Papel. Então você muda para Tesoura. Então ele muda para Pedra. Então você muda para Papel... e assim por diante. A "melhor jogada" nunca fica parada.

A solução clássica são as **estratégias mistas**: não escolha nenhuma ação de forma determinística — use a aleatoriedade de uma forma que o oponente não consiga explorar.

---

## Os Três Jogos Que Jogamos

### 1) Pedra-Papel-Tesoura (soma zero)
- O ganho de um jogador é a perda do outro.
- O **equilíbrio de Nash** é: cada jogador escolhe cada ação com probabilidade de ⅓. Qualquer desvio é explorável.
- Esperamos que nossos dois aprendizes Q oscilem em torno de ⅓-⅓-⅓ — nunca perfeitamente estáveis, porque toda vez que um se desvia, o outro reage.

### 2) Dilema do Prisioneiro (soma geral)
Dois suspeitos são interrogados separadamente:

|            | Cooperar | Trair |
|------------|:--------:|:-----:|
| Cooperar   |   3, 3   | 0, 5  |
| Trair      |   5, 0   | 1, 1  |

- "Trair" vence "Cooperar" não importa o que o outro faça — é uma **estratégia dominante**.
- Ambos os jogadores são racionais → ambos traem → ambos recebem 1, embora (Cooperar, Cooperar) rendesse 3 para cada um. A melhor resposta egoísta destrói o bem-estar do grupo.
- Esperamos que o Q-learning converja claramente para (Trair, Trair).

### 3) Caça ao Cervo (coordenação)
Dois caçadores podem, juntos, abater um cervo (grande prêmio), ou cada um se contentar com uma lebre (prêmio pequeno, mas seguro):

|       | Cervo | Lebre |
|-------|:-----:|:-----:|
| Cervo | 4, 4  | 0, 3  |
| Lebre | 3, 0  | 2, 2  |

- (Cervo, Cervo) é **dominante em pagamento** — melhor para ambos.
- (Lebre, Lebre) é **dominante em risco** — seguro se você não confiar no seu parceiro.
- O resultado depende das condições iniciais: aprendizes Q independentes frequentemente terminam no equilíbrio *pior* (Lebre, Lebre) porque lebres são mais seguras de aprender.

---

## Exemplos da Vida Real

- **Precificação em um duopólio.** Duas cafeterias na mesma rua escolhem um preço a cada manhã. O formato da matriz de pagamentos decide se elas terminam em um preço "cooperativo" alto (bom para elas, ruim para os clientes) ou em um preço baixo agressivo.
- **Protocolos de rede.** Roteadores e remetentes escolhem estratégias de tempo; o resultado do congestionamento da rede é determinado pelo pagamento tipo jogo matricial de conseguir passar vs. recuar.
- **Lances em um leilão.** Cada licitante escolhe um lance sem conhecer os outros; os ganhos dependem de todo o vetor. O equilíbrio de Nash é uma *estratégia de lances*, não um único número.

---

## O Que Nosso Código Faz

Para cada jogo, nós:
1. Criamos dois aprendizes Q sem estado (Q é apenas um número por ação — não existem estados em um jogo de um único turno).
2. Rodamos 20.000 passos. Em cada passo: ambos os agentes escolhem uma ação ε-greedy simultaneamente, recebem uma recompensa e atualizam seus valores Q.
3. Rastreamos a **frequência empírica de ações** de cada agente em uma janela móvel de 500 passos. Em vez de apenas olhar para probabilidades abstratas, contamos quais ações eles realmente escolheram recentemente (ex: "nas últimas 500 rodadas, eles jogaram Pedra 40% das vezes"). Isso nos dá uma imagem prática e em tempo real da mudança de estratégia.
4. Plotamos as frequências ao longo do tempo, salvamos em `outputs/<jogo>.png` e imprimimos os valores Q finais.

### O que você deve ver

| Jogo | Resultado esperado no gráfico |
|------|-------------------------------|
| **Pedra-Papel-Tesoura** | Ambos os jogadores flutuam perto de ⅓-⅓-⅓, mas com oscilações visíveis. As curvas se perseguem — comportamento cíclico clássico. |
| **Dilema do Prisioneiro** | A frequência de "Trair" de ambos os jogadores sobe para ~1,0 rapidamente. "Cooperar" é esmagado. |
| **Caça ao Cervo** | A maioria das sementes (seeds) aleatórias termina em (Lebre, Lebre). Algumas sementes sortudas alcançam (Cervo, Cervo) — tente mudar a semente no script e veja a mudança. |

---

## Onde o Aprendizado Independente Falha

Nossos agentes são *independentes* — eles só veem sua própria recompensa, nunca a ação ou os valores Q do oponente. Esta é a linha de base mais simples e tem limites:

- **Não pode garantir a convergência** em jogos de soma geral.
- Pode ficar preso em **maus equilíbrios** (Caça ao Cervo).
- **Não pode modelar o oponente**.

Algoritmos multiagente reais resolvem isso raciocinando explicitamente sobre o outro aprendiz. Aqui está o que cada um faz, em linguagem simples:

| Algoritmo | Ideia central | Analogia da vida real |
|-----------|---------------|-----------------------|
| **Fictitious play** | Mantenha uma contagem de quantas vezes seu oponente escolheu cada ação. Assuma que amanhã ele fará o que sempre fez — então escolha sua melhor resposta a essa crença. | Observar os hábitos de um oponente ao longo de muitos jogos de xadrez e ajustar sua abertura de acordo. |
| **CFR (Counterfactual Regret Minimisation)** | Após cada rodada, pergunte: *"Quanto eu me arrependi de não ter escolhido cada uma das outras ações?"* Desloque gradualmente a probabilidade para as ações que você se arrepende de ter ignorado. Usado no pôquer porque lida com jogos de **informação imperfeita**. | Após uma mão de pôquer, revê-la e pensar: *"Eu deveria ter apostado mais — farei isso da próxima vez".* |
| **LOLA (Learning with Opponent-Learning Awareness)** | Seu passo de gradiente leva em conta o fato de que o oponente *também* está dando um passo de gradiente. Você otimiza sua própria atualização antecipando a próxima atualização do oponente. | Negociar um acordo pensando: *"Se eu oferecer X, eles contra-atacarão com Y, então eu devo começar com Z".* |
| **MADDPG (Multi-Agent Deep Deterministic Policy Gradient)** | O *crítico* (estimador de valor) de cada agente é treinado com a **visão global**: ele vê as observações e ações de todos. O *ator* (a política) ainda usa apenas informações locais. | Um treinador de basquete que observa a quadra inteira (crítico centralizado), mas ensina cada jogador a reagir apenas ao que ele pode ver (ator descentralizado). |

Mas o Q-learning independente é o primeiro passo correto. Você vê o problema da não estacionariedade na sua frente, e as correções fazem sentido depois.

---

## Palavras-Chave para Lembrar

| Palavra | Significado |
|------|---------|
| **Matriz de pagamentos** | A tabela que define um jogo multiagente de um único turno |
| **Equilíbrio de Nash** | Um perfil de políticas onde nenhum agente pode melhorar mudando sozinho |
| **Estratégia mista** | Uma política que usa aleatoriedade sobre múltiplas ações |
| **Não estacionariedade** | O ambiente (= outros agentes) muda constantemente conforme aprende |
| **Aprendiz independente** | Um agente que ignora a existência de outros aprendizes |
| **Soma zero** | O ganho de um agente é exatamente a perda do outro |
| **Soma geral** | Ambos os agentes podem ganhar, ambos podem perder, ou qualquer coisa entre isso |

---

## Resumo de Uma Frase

> **Em jogos matriciais, o "ambiente" é outro aprendiz — então a melhor jogada está em constante movimento.**

Esta é a ideia fundamental por trás de cada algoritmo multiagente que você encontrará mais tarde, do self-play ao MADDPG e ao MARL com comunicação.
