# Comparando Estratégias de Exploração 🔦

## O Problema em uma Frase

Um agente de RL tem que fazer duas coisas que puxam em direções opostas:

- **Explotar (Exploit)**: fazer a coisa que funcionou melhor até agora (aproveitamento).
- **Explorar (Explore)**: tentar algo novo, caso seja ainda melhor.

Se inclinar demais para a explotação e você se contentará com uma rotina medíocre para sempre. Se inclinar demais para a exploração e você nunca colherá os frutos. *Como* você explora — e não apenas *se* explora — é o que separa um agente que resolve o Montezuma's Revenge de um que marca zero.

Este script coloca **cinco** estratégias de exploração frente a frente nas mesmas duas tarefas difíceis, para que você possa ver suas personalidades.

## Analogia da Vida Real: Escolhendo onde Almoçar

Você acabou de se mudar para uma cidade nova com 200 restaurantes.

- **ε-greedy** = "Vá ao meu favorito atual, mas uma vez a cada dez dias, jogue um dado e escolha um restaurante *totalmente aleatório*." Você experimentará de tudo, mas *sem rumo* — e acabará voltando a lugares que já odiava.
- **Inicialização otimista** = "Assuma que *cada* restaurante que não experimentei é o melhor da cidade até que se prove o contrário." Você passará metodicamente pelos 200, riscando cada um à medida que a realidade o decepciona — e encontrará os genuinamente fantásticos rapidamente.
- **UCB (Upper Confidence Bound)** = "Prefira o meu favorito, mas dê um *bônus* aos lugares que mal experimentei — quanto menos eu souber sobre ele, maior o bônus." Isso é inteligente sobre *qual* lugar desconhecido tentar hoje, mas cada decisão é local: escolhe a melhor opção *agora mesmo* sem planejar uma rota por bairros inteiros inexplorados. Ele não pensará "devo atravessar a cidade para o lado leste, porque há vinte lugares não testados agrupados lá" — cada restaurante é avaliado isoladamente, passo a passo.
- **Bônus de recompensa baseado em contagem** = como o UCB, mas você também *aproveita a novidade em si* — uma refeição em um lugar novo é intrinsecamente satisfatória, e essa satisfação molda seu plano de longo prazo de em quais bairros se aventurar.
- **Bônus de recompensa por erro de previsão** = "Eu sinto um entusiasmo com uma refeição que me *surpreendeu* — algo que eu não poderia ter previsto." Um lugar novo que acaba sendo exatamente como o esperado? Sem graça. Um que é radicalmente diferente do seu modelo mental? Fascinante, e você atualiza seu plano para buscar mais lugares assim.

## As Cinco Estratégias (todas em `compare_exploration.py`)

### 1. ε-greedy — o padrão, e é *dithering* (hesitação), não exploração

Age de forma gananciosa (greedy), mas com probabilidade ε toma uma ação uniformemente aleatória. É o benchmark padrão no DQN e similares. Sua falha fatal em tarefas difíceis: **cada passo é um lançamento de moeda independente.** Para tropeçar em uma cadeia de `N` movimentos corretos, você precisa que a moeda caia do lado certo `N` vezes seguidas — isso é exponencialmente improvável. ε-greedy é *trepidação*, não *exploração*.

### 2. Inicialização otimista — "inocente até que se prove entediante"

Começa *cada* valor Q no maior retorno que seja possível, `R_max / (1 − γ)`. Agora, uma ação que o agente nunca tentou parece a melhor coisa do mundo, então a política **greedy** é forçada a experimentá-la; somente após visitá-la é que o valor cai em direção à verdade. O otimismo sobre regiões *não* testadas se **propaga automaticamente através da função de valor** (via o bootstrap do Q-learning), então o agente é puxado, passo a passo, em direção às partes do mundo que ainda não viu. Quase de graça, sem contabilidade extra — e, como você verá, o explorador *profundo* mais forte em um pequeno mundo tabular.

### 3. Seleção de ações estilo UCB — bônus na *escolha*, não na *recompensa*

Escolha `argmax_a [ Q(s,a) + c·√(ln t / N(s,a)) ]`: prefira ações de alto valor, mas infle aquelas que você raramente tentou. Famoso em bandidos multibraço (multi-armed bandits). O detalhe: o bônus vive apenas na **regra de seleção de ações**, nunca na recompensa — portanto, *não* flui através da função de valor. O UCB é ótimo para "garantir que eu tentei cada ação *neste* estado", mas fraco para "planejar uma rota em direção a uma região inexplorada distante".

### 4. Bônus de **recompensa** baseado em contagem — curiosidade, a versão clássica

Adiciona `1/√(N(s,a))` à **recompensa** (com um peso `beta` que decai). Como está na recompensa, o Q-learning *propaga* esse valor: estados que levam a regiões novas tornam-se valiosos. Esta é a ideia de MBIE-EB / "bônus de exploração" clássico.

### 5. Bônus de **recompensa** por erro de previsão — curiosidade, a versão ICM/RND

Adiciona `−log P(s'|s,a)` de um pequeno modelo preditivo treinado à recompensa (novamente com `beta` decrescente). O sinal de novidade mais nítido dos cinco: em um mundo determinístico, a surpresa de uma transição cai para ~0 no momento em que você a vê uma vez, em vez de desaparecer lentamente como `1/√N`. É o primo tabular do ICM / RND.

## As Duas Tarefas de Teste

- **Tarefa A — MiniMontezuma**: um mundo de grade (gridworld) chave→porta→tesouro, recompensa apenas no tesouro (~15 movimentos perfeitos de distância). Testa: "você consegue sobreviver a uma longa cadeia de recompensa esparsa?"
- **Tarefa B — DeepSea(N)**: a corrente de exploração profunda de livro-texto, executada em comprimentos `N = 5, 8, 11, 14`. A recompensa se esconde atrás de `N` movimentos corretos, cada um com um pequeno custo imediato — de modo que um agente míope aprende a evitar o custo e nunca encontra o prêmio. Testa: "sua estratégia continua funcionando à medida que a corrente fica mais *longa*?"

## O Que Realmente Acontece (execute e veja)

**Tarefa A — MiniMontezuma:**

| Estratégia | Primeiro tesouro | Taxa de resolução final |
|------------|------------------|-------------------------|
| ε-greedy | nunca | 0.00 |
| inicialização otimista | ~episódio 1 | 1.00 |
| seleção de ação UCB | ~episódio 3 | ~0.95 |
| bônus recompensa contagem | ~episódio 82 | ~0.41 |
| bônus recompensa previsão | ~episódio 23 | 1.00 |

**Tarefa B — DeepSea, fração de sementes (seeds) que encontraram a recompensa:**

| Estratégia | N=5 | N=8 | N=11 | N=14 |
|------------|----:|----:|-----:|-----:|
| ε-greedy | 0 | 0 | 0 | 0 |
| inicialização otimista | 1.0 | 1.0 | 1.0 | 1.0 |
| seleção de ação UCB | 1.0 | 1.0 | 0.0 | 0.0 |
| bônus recompensa contagem | 1.0 | 1.0 | ~0.1 | 0.0 |
| bônus recompensa previsão | ~0.9 | ~0.8 | ~0.9 | ~0.2 |

*(Os números oscilam um pouco com as sementes aleatórias, mas o padrão é sólido).*

## As Lições

1. **ε-greedy não é exploração.** Ele nunca resolve *nenhuma* das duas tarefas difíceis. A hesitação aleatória simplesmente não consegue encadear sequências longas e corretas. (Ainda assim, é o padrão em muitos códigos — porque em tarefas *fáceis* é bom o suficiente e extremamente simples).

2. **Exploração real significa ser otimista sobre o desconhecido — de uma forma ou de outra.** Seja inserindo o otimismo nos *valores iniciais* (estratégia 2), na *escolha da ação* (estratégia 3) ou em uma *recompensa autogerada* (estratégias 4–5), o ponto em comum é: *fazer o inexplorado parecer atraente*, e então deixar que o aprendizado o leve até lá.

3. **Em uma grade de recompensa esparsa, todas as quatro estratégias "reais" funcionam — e o bônus por erro de previsão chega mais rápido**, porque produz o sinal de "isso é novo" mais nítido.

4. **Em uma corrente *profunda*, onde o otimismo tem que viajar um longo caminho, o campeão simples é a inicialização otimista.** Ela propaga o otimismo através da função de valor gratuitamente. O UCB desmorona primeiro (seu bônus nunca entra na função de valor, então ele não pode *planejar* profundamente). Os bônus de recompensa se saem melhor — eles *se* propagam — mas o Q-learning tabular simples é lento para empurrar esse otimismo por toda a corrente antes que o bônus decaia.

5. **Esse último ponto é exatamente o motivo pelo qual escalar a exploração profunda para pixels precisou de poder extra** — DQN bootstrapped, RND com uma rede neural real (para que o otimismo *generalize* entre estados semelhantes em vez de se propagar uma célula por vez), Go-Explore (literalmente lembrar e retornar a estados promissores). Os exemplos tabulares aqui mostram os *princípios*; os sistemas reais são esses mesmos princípios mais uma rede neural que generaliza.

## Palavras-Chave para Lembrar

| Palavra | Significado |
|---------|-------------|
| **Exploration–exploitation trade-off** | O dilema exploração–explotação: tentar coisas novas vs. aproveitar o que já sabe |
| **Dithering** | "Exploração" adicionando ruído aleatório às ações (ε-greedy, ruído Gaussiano) — fraco em tarefas difíceis |
| **Otimismo diante da incerteza** | O princípio guarda-chuva: tratar o desconhecido como se fosse ótimo até que se verifique o contrário |
| **Inicialização otimista** | Implementar esse princípio começando todos os valores no retorno máximo possível |
| **UCB** | Upper Confidence Bound: escolher `argmax (valor + bônus que diminui com a contagem de visitas)` |
| **Exploração profunda** | Exploração que requer uma sequência *coerente* e longa de ações "incomuns", não apenas uma |
| **Annealing de `beta`** | Reduzir o peso da curiosidade ao longo do tempo para que o agente pare de explorar e passe a explotar |

## Resumo de Uma Frase

> **ε-greedy é apenas ruído; toda estratégia de exploração real funciona fazendo o inexplorado parecer atraente — via valores otimistas, um bônus de escolha de ação ou uma recompensa de novidade autogerada — e a escolha certa depende de se sua recompensa é meramente *esparsa* (como encontrar um prêmio escondido em um campo plano) ou genuinamente *profunda* (como um segredo de cofre que requer uma sequência longa e precisa de escolhas específicas).**
