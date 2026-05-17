# Treinamento no Montezuma's Revenge 🏛️🔑

## Por que este jogo é famoso (nos círculos de RL)

Em 2015, o DQN da DeepMind aprendeu a jogar dezenas de jogos de Atari em nível sobre-humano a partir de pixels brutos. Foi notícia em todo o mundo. Mas enterrado na tabela de resultados estava um jogo onde o DQN obteve **0** — o mesmo que não fazer nada: **Montezuma's Revenge**.

Por quê? Veja o que o jogo pede de você logo na primeira sala:

1. Descer uma escada.
2. Atravessar uma saliência.
3. Pular sobre uma caveira rolando (se errar o tempo → você morre).
4. Subir outra escada.
5. Pegar a chave.

Isso são aproximadamente **100 pressionamentos de botão precisos**, e o jogo não te dá **nem um único ponto** até que a chave esteja na mão. O sinal de recompensa é um **zero** plano e sem características durante toda a sequência de abertura.

Um agente de RL normal aprende ajustando-se em direção às recompensas que realmente recebe. Se a recompensa for zero em todos os lugares que ele pode alcançar, não há *nada a aprender* — é como tentar encontrar o fundo de um vale perfeitamente plano tateando a direção da descida. Então, o DQN apenas se debateu na plataforma inicial para sempre. Montezuma tornou-se *o* benchmark para **exploração difícil** (hard exploration): o jogo que você só consegue vencer se explorar de forma *inteligente*, não aleatória.

O avanço veio em 2018 com o **Random Network Distillation (RND)** — e o truque foi exatamente o assunto do item de trabalho 1: adicionar um **bônus de curiosidade intrínseca** para que o agente recompense a *si mesmo* por alcançar novas telas, e de repente ele tem um sinal denso puxando-o mais para o fundo do nível. O RND obteve uma pontuação sobre-humana no Montezuma. (Mais tarde: Go-Explore, Agent57, …)

## Exemplos da vida real de recompensa esparsa "estilo Montezuma"

- **Um cadeado de combinação / uma caça ao tesouro com pistas enigmáticas.** Sem crédito parcial. Você está no zero até que de repente chega ao prêmio.
- **Conseguir que um artigo seja aceito, ou que uma startup seja lucrativa.** Meses sem recompensa externa, e então (talvez) uma grande.
- **Uma rota de speedrun de videogame.** Dezenas de comandos com precisão de frame seguidos, sem feedback até que o truque funcione ou não.
- **Escape rooms.** A sala não te diz quase nada até que você tenha encadeado várias descobertas.

Em todos esses casos, "apenas tentar coisas aleatórias" é inútil. Você precisa explorar *sistematicamente* — e um sinal interno de "opa, isso é novo, continue" é o que mantém você sistemático.

## Por que não treinamos realmente no Montezuma de pixels aqui

Fazer a coisa *real* adequadamente significa:

- uma rede convolucional para ver a tela RGB de 210×160,
- empilhamento de frames (para que o agente saiba para qual lado a caveira está se movendo),
- um módulo RND (mais duas redes: um "alvo" aleatório fixo e um "preditor" treinado),
- e **dezenas de milhões de frames de ambiente** — muitas horas de GPU.

Isso é um projeto de pesquisa, não um script de ensino. Portanto, `montezuma_revenge.py` faz duas coisas honestas em vez disso:

### 1. Ele "toca" o jogo real (se o `ale-py` estiver instalado)

Ele carrega `ALE/MontezumaRevenge-v5` via Gymnasium, executa um **agente aleatório uniforme por 2000 passos** e reporta a recompensa total do jogo. O número que ele imprime é quase sempre **0.0** — a frase abstrata "recompensa esparsa" transformada em um fato concreto e executável. Se o pacote Atari não estiver instalado, ele imprime o comando `pip install` de uma linha e prossegue.

### 2. Ele treina um agente tabular em um *modelo em escala*: `MiniMontezumaEnv`

Este é um pequeno mundo de grade com o mesmo *esqueleto* da primeira sala do Montezuma:

```
###############
#S....#.......#
#.....#.......#
#.....#...T...#     S = início (start)
#.....D.......#     K = chave (key)      D = porta (só passável carregando a chave)
#..K..#.......#     T = tesouro (o ÚNICO ladrilho que dá recompensa)
###############
```

Para vencer você deve: caminhar até a **chave** (~6 movimentos), pegá-la; caminhar até a **porta** (~4 movimentos) — que agora se abre; atravessar e chegar ao **tesouro** (~5 movimentos). Cerca de **15 movimentos perfeitos**, com **zero feedback até o tesouro**. A flag `has_key` faz parte do estado do agente, então, uma vez que você pega a chave, há toda uma segunda sala de *novos* estados para descobrir — exatamente como novas telas se abrindo no jogo real.

Em seguida, treinamos um **Q-learner tabular** comum duas vezes:

| Agente | Resultado no MiniMontezuma |
|-------|--------------------------|
| **sem curiosidade (epsilon-greedy)** | O retorno permanece em **0** por todos os 1.500 episódios. Ele nunca chega à chave. (Parece familiar? Esse é o DQN no jogo real.) |
| **com um bônus de curiosidade por erro de predição** | Alcança o tesouro em cerca de 20–25 episódios e então aprende a **rota ótima de 15 passos**. (Essa é a ideia do RND, reduzida para caber em uma tabela Q.) |

A figura mostra as duas curvas de aprendizado lado a lado, além da rota real que o agente curioso aprendeu, desenhada na grade (início → chave → porta → tesouro). O script também imprime essa rota como frames ASCII.

## A Lição

> **"Recompensa esparsa" não é uma peculiaridade de um jogo estranho de Atari — é o padrão em qualquer mundo onde o sucesso exige uma sequência longa e específica de ações.** Um agente apenas de recompensa (DQN vanilla) literalmente não consegue começar: não há gradiente para seguir. Um bônus de curiosidade fabrica um — um sinal denso e autogerado de "isso é novo, continue" — e esse sinal é o que carrega o agente através do deserto de zeros até a primeira recompensa real. Tudo o que vem depois (RND, Go-Explore, Agent57) é uma versão em rede neural em escala ampliada da mesma jogada.

## Palavras-Chave para Lembrar

| Palavra | Significado |
|------|---------|
| **Exploração difícil (Hard exploration)** | Problemas onde você só tem sucesso explorando de forma inteligente; a exploração aleatória falha |
| **Recompensa esparsa (Sparse reward)** | A recompensa é zero em quase todos os lugares; você a obtém apenas após uma longa sequência correta |
| **Montezuma's Revenge** | O jogo de Atari no qual os agentes de deep RL clássicos (DQN, A3C) pontuaram 0 — o benchmark canônico de exploração difícil |
| **RND (Random Network Distillation)** | O método de 2018 que venceu o Montezuma usando um bônus de curiosidade por erro de predição |
| **Go-Explore** | "Lembrar estados promissores, retornar a eles e então explorar a partir dali" — outro vencedor do Montezuma |
| **Modelo em escala (Scale model)** | Um ambiente pequeno e barato que mantém a *estrutura* de um problema difícil para que você possa estudá-lo rapidamente |

## Resumo em Uma Frase

> **Montezuma's Revenge é o jogo que ensinou ao RL que "recompensas que você nunca recebe não podem te ensinar nada" — e a solução, naquela época e agora, é um bônus de curiosidade que permite ao agente recompensar a si mesmo por explorar até encontrar o prêmio real.**
