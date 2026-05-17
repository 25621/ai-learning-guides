# Self-Play: Ensinando um Agente deixando-o Jogar contra Si Mesmo ♟️

## O que é Self-Play?

Imagine uma criança que quer ficar muito boa no xadrez, mas não tem ninguém para jogar.
Então, ela joga contra si mesma. Mão esquerda contra mão direita. A cada jogo, *ambos* os lados tentam
vencer. A cada jogo, *ambos* os lados aprendem o que funcionou.

Isso é o **self-play**: um único agente atua como ambos os jogadores, e cada movimento
torna-se uma lição para quem for jogar em seguida. Sem professor, sem oponente especialista.
Apenas um aprendiz que é também seu próprio degrau.

O self-play parece um truque — certamente você precisa de um oponente real? — mas é
o motor por trás dos marcos mais famosos de RL da última década:
**AlphaGo Zero**, **AlphaZero**, **MuZero**, **OpenAI Five**. Todos eles usam
self-play. O motivo é simples: à medida que você melhora, seu oponente melhora na
mesma medida. O desafio sempre acompanha sua habilidade.

---

## Por que Funciona

Três coisas tornam o self-play especial:

1. **Oponentes infinitos.** Você nunca fica sem jogos. O oponente está sempre
   presente e é gratuito.
2. **Currículo que cresce com você.** Um iniciante só pode jogar com outros
   iniciantes. À medida que você melhora, sua sombra também melhora — automaticamente.
3. **Simetria.** Em um jogo de soma zero (a vitória de um jogador é a perda do outro),
   um conjunto de valores Q descreve ambos os lados; você apenas inverte o sinal quando
   é a vez do outro jogador. Assim, uma *única* Tabela Q pode ensinar a si mesma.

O jogo da velha (tic-tac-toe) é o campo de testes perfeito: pequeno o suficiente para caber em um dicionário, mas
complexo o suficiente para que escolher movimentos aleatórios quase sempre leve a uma derrota contra um jogador estratégico.

---

## Uma Analogia da Vida Real

- **Praticar tênis contra a parede.** Você não pode perder para uma parede, mas
  pode praticar seus saques. O self-play é fazer isso em ambas as pontas — você é a parede
  *e* o jogador, e você alterna entre as funções.
- **Um clube de debate que argumenta ambos os lados.** Melhores debatedores surgem ao
  defender sempre o ponto de vista oposto ao que acreditam pessoalmente. Cada argumento treina tanto o ataque quanto a defesa.
- **AlphaGo Zero.** Ele aprendeu com zero jogos humanos. Começando com movimentos aleatórios,
  jogou milhões de partidas contra si mesmo; em poucos dias, era melhor do que qualquer
  programa de Go anterior, incluindo aquele que venceu Lee Sedol.

---

## O que nosso Código Faz

Nós aprendemos uma Tabela Q para o *jogador da vez*:

```
Q[(tabuleiro, jogador_da_vez)][ação] = retorno esperado para esse jogador
```

O loop de treinamento é:

1. Começa com um tabuleiro vazio. `jogador = X`.
2. Ambos os jogadores agem com o **mesmo agente**, usando ε-greedy.
3. Após cada jogo, percorre-se de trás para frente cada trio (tabuleiro, jogador, ação)
   no histórico e aplica-se a atualização de Q-learning.
4. O sinal da recompensa inverte a cada turno: se o X vence, cada movimento que o X fez recebe
   +1 (ou faz o bootstrap do valor de um estado vencedor futuro); cada movimento que o O fez recebe -1.
5. Nós diminuímos lentamente (decay) nossa taxa de exploração (ε) de 0,2 → 0,02, para que o agente se comprometa com sua melhor jogada no final do treinamento em vez de tentar movimentos aleatórios.

A cada 2.500 episódios, avaliamos o agente contra um **oponente aleatório**
(congelamos o processo de aprendizado para que nenhuma nova atualização seja feita na Tabela Q durante a avaliação, e ambos os lados jogam de forma gananciosa/greedy). O agente deve vencer ou empatar
~100% desses jogos após self-play suficiente.

### O que você deve ver

Após 50.000 episódios de self-play:

| Confronto | Resultado esperado |
|----------|-----------------|
| Agente treinado vs Oponente aleatório (1000 jogos) | **~95-99% de vitórias ou empates**, virtualmente 0% de derrotas |
| Agente treinado vs Ele mesmo (200 jogos greedy) | **Todos os 200 empatados**. Jogo da velha é um jogo que sempre termina em empate se ambos os jogadores jogarem perfeitamente. O fato de o self-play empatar todos os jogos é um sinal de convergência. |

O gráfico `outputs/self_play_tic_tac_toe.png` mostra as frações de vitória/empate/derrota do
agente versus um oponente aleatório ao longo do tempo:
- A taxa de vitória começa em ~60% (quando ambos os jogadores jogam aleatoriamente, o primeiro jogador tem uma vantagem inerente porque consegue colocar mais marcadores no tabuleiro, levando a uma taxa de vitória base de cerca de 60% para o jogador X).
- Sobe para >90%.
- A taxa de derrota cai para quase 0%.

O script também imprime um exemplo de jogo jogada a jogada no final para que você possa ver o agente jogar.

---

## Cuidado com estas Sutilezas

- **Inversões de sinal importam.** Um erro comum: esquecer que "o oponente
  maximizando o valor dele" significa *minimizar o nosso* no alvo do bootstrap.
  A atualização em nosso código usa `target = recompensa - gamma * max(Q[proximo, oponente])`.
- **A simetria não é explorada aqui.** Uma implementação real faria a canonicalização
  dos tabuleiros (significando que rotacionariam ou refletiriam qualquer estado do tabuleiro para uma 'forma normal' padrão e única para que o agente reconheça situações de tabuleiro idênticas) para compartilhar valores Q entre as 8 simetrias. Nós pulamos isso — o espaço de estados é pequeno o suficiente para usar força bruta.
- **A Tabela Q cresce.** Após 50 mil jogos de self-play, você verá alguns milhares
  de chaves de estado-jogador. Isso é aceitável aqui; para xadrez ou Go, você precisaria de
  uma rede neural em vez disso, e é por isso que o **AlphaZero substitui a tabela por uma CNN + MCTS**.

---

## Onde o Self-Play Falha

- **Jogos de soma não zero.** "Ambos os lados felizes" é incompatível com
  o jogo simétrico; você não pode simplesmente inverter um sinal.
- **Papéis assimétricos.** Se "atacante" e "defensor" têm espaços de ação
  diferentes, você precisa de duas redes separadas.
- **Ciclos de estratégia.** O self-play puro pode ficar preso em ciclos tipo
  pedra-papel-tesoura. O AlphaStar corrigiu isso mantendo um grande *conjunto* (pool ou "liga") de versões passadas salvas do agente e escolhendo oponentes desse conjunto aleatoriamente, para que o agente aprenda a vencer muitos estilos de jogo diferentes em vez de apenas o atual.
- **Reward hacking.** O self-play torna ambos os lados mais inteligentes, mas apenas no
  jogo *conforme você o definiu*. Se o seu sistema de recompensa tiver brechas não intencionais (como recompensar um jogador apenas por sobreviver mais tempo em vez de vencer), ambos os lados explorarão mutuamente a brecha, levando a comportamentos bizarros e inúteis em vez de dominar o jogo real.

---

## Palavras-Chave para Lembrar

| Palavra | Significado |
|------|---------|
| **Self-play**      | O mesmo agente joga em ambos os lados de um jogo |
| **Soma zero**       | O ganho de um jogador = a perda do outro |
| **Simetria**       | Uma Tabela Q pode servir a ambos os lados se você inverter os sinais |
| **Population play**| Self-play com *muitas* versões passadas de si mesmo como oponentes (AlphaStar) |
| **Currículo**     | Uma progressão natural de dificuldade — o self-play a consegue de graça |
| **MCTS**           | Busca em Árvore Monte-Carlo — o algoritmo de planejamento que o AlphaZero combina com o self-play |

---

## Resumo em uma Frase

> **O self-play transforma a melhoria em sua própria escada: toda vez que você
> fica melhor, seu oponente também fica — automaticamente.**

Esta ideia, escalada com **redes neurais** (funções matemáticas inspiradas no cérebro que aprendem padrões a partir de dados) e busca em árvore, venceu os melhores humanos no Go, xadrez, shogi, Dota 2 e StarCraft.
