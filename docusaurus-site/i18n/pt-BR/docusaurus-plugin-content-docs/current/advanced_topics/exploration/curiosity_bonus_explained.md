# Bônus de Curiosidade (Motivação Intrínseca) 🧭

## O Que É?

Imagine uma criança pequena deixada em uma sala nova. Ninguém a paga, ninguém a aplaude — no entanto, ela vai direto para o armário que não abriu, para o botão que não apertou, para o brinquedo barulhento no canto. Ela está funcionando com uma **recompensa interna**: *"Isso parece novo. Vá conferir."*

Um **bônus de curiosidade** dá a um agente de aprendizado por reforço esse mesmo impulso interno. A recompensa real do ambiente (a recompensa "extrínseca" — pontos, dinheiro, ganhar o jogo) permanece exatamente como é. Nós apenas adicionamos uma segunda recompensa, autogerada por visitar coisas que o agente considera *novas* ou *surpreendentes*, e treinamos sobre a soma:

```
recompensa que o agente aprende = recompensa real + beta * bônus de curiosidade
```

`beta` é um parâmetro que começa alto (seja curioso!) e diminui ao longo do tempo (pare de enrolar, vá colher o que você aprendeu).

## Por que se preocupar? O Problema da Recompensa Esparsa

Agentes de RL normais aprendem com as recompensas que realmente recebem. Isso funciona muito bem quando as recompensas estão em todos os lugares ("+1 a cada passo que você ficar de pé" no CartPole). O sistema desmorona quando a recompensa é **esparsa** (sparse) — zero, zero, zero, ... , zero e, finalmente, um +1 após uma sequência longa e muito específica de ações corretas.

Exemplos reais de recompensa esparsa:

- **Montezuma's Revenge** (o jogo do Atari): seu primeiro ponto chega apenas após ~100 movimentos precisos — descer uma escada, desviar de uma caveira, subir, pegar uma chave. Até lá, a pontuação é um zero absoluto.
- **Um cadeado de combinação.** 9.999 códigos errados não te dão nada; um te dá o prêmio.
- **Descoberta de medicamentos / experimentos científicos.** Milhares de tentativas fracassadas e, depois, uma que funciona.
- **Escrever uma prova longa ou um programa.** Nenhum crédito parcial até que tudo esteja correto.

Um agente que só se baseia em recompensas nessas situações é como um estudante que se recusa a estudar a menos que seja pago por cada resposta correta na prova final — ele nunca começa. A curiosidade é o bônus que diz *"explorar é sua própria recompensa"*, então o agente continua bisbilhotando até tropeçar no prêmio real.

## Dois Tipos de Curiosidade (ambos implementados em `curiosity_bonus.py`)

### 1. Novidade baseada em contagem: "Eu mal estive aqui"

O sinal de novidade mais simples possível. Mantenha uma contagem `N(s, a)` de quantas vezes você tomou a ação `a` no estado `s`, e dê a si mesmo um bônus que diminui conforme essa contagem cresce:

```
bônus de curiosidade = 1 / sqrt( N(s, a) + 1 )
```

A primeira vez que você tenta algo: bônus = 1,0. Após 100 tentativas: bônus = 0,1. Após 10.000 tentativas: 0,01. O agente é recompensado por ir aonde não esteve, e o atrativo naturalmente desaparece de terrenos já conhecidos.

**Analogia da vida real:** um turista com uma lista de "lugares que não visitei". Um bairro novinho em folha? Prioridade máxima. Aquele café onde você já foi cinquenta vezes? Não é mais emocionante.

Esta é a ideia mais antiga do gênero (MBIE-EB, UCB). Sua fraqueza: em um mundo enorme ou contínuo, você nunca visita o *exato* mesmo estado duas vezes, então a contagem bruta é sempre 1 — e é por isso que o próximo tipo existe.

### 2. Novidade por erro de previsão: "Eu não esperava por *essa*"

Esta é a ideia por trás do famoso **ICM** (Intrinsic Curiosity Module, Pathak et al. 2017) e seu primo **RND** (Random Network Distillation, Burda et al. 2018). Em vez de contar, o agente mantém um pequeno **modelo que tenta prever o que acontece a seguir** — "se estou aqui e faço isso, onde vou parar?" — e se recompensa pelo **quão errado o modelo estava**:

```
bônus de curiosidade = surpresa = -log P( o estado que realmente alcancei | onde eu estava, o que eu fiz )
```

- Uma situação que o modelo nunca viu → ele prevê mal → grande surpresa → grande bônus → "vá explorar lá!"
- Uma situação que o modelo já viu cem vezes → ele prevê perfeitamente → zero surpresa → zero bônus → "já estive lá, já entendi, próximo."

**Analogia da vida real:** uma criança aprendendo como o mundo funciona brincando. Derrubar um copo da mesa pela *primeira* vez é fascinante (ele quebrou!). Na centésima vez, você já sabia que ele quebraria — não é mais interessante. Curiosidade = a lacuna entre o que você esperava e o que aconteceu.

Em nosso código tabular, o "modelo" é apenas uma tabela de contagens de transição, e o "quão errado estava" é a surpresa `-log P`. O ICM/RND real usa redes neurais para que a mesma ideia funcione em pixels brutos — mas o princípio é idêntico.

> **Por que duas versões?** A baseada em contagem é extremamente simples e um ótimo benchmark. A de erro de previsão escala para mundos grandes que nunca se repetem e fornece um sinal mais *nítido*: em um ambiente determinístico, uma vez que você vê uma transição, a surpresa cai instantaneamente para ~0, enquanto um bônus de contagem desaparece lentamente como `1/sqrt(N)`. Em nossos experimentos, o agente de erro de previsão resolve o MiniMontezuma em algumas dezenas de episódios; o agente de contagem também chega lá, mas de forma mais lenta e menos confiável.

## O Que Nosso Código Faz

`curiosity_bonus.py` treina um **Q-learner tabular** simples no `MiniMontezumaEnv` — um pequeno mundo de grade de duas salas onde você deve caminhar até uma **chave**, pegá-la (agora a **porta** se abre), passar por ela e alcançar o **tesouro**. A recompensa (+1) aparece *apenas* no tesouro, após cerca de 15 movimentos perfeitos. O script executa três agentes e os plota:

| Agente | O que faz no MiniMontezuma |
|--------|----------------------------|
| **epsilon-greedy (sem curiosidade)** | Vagueia perto do início, *nunca* chega à chave, a pontuação permanece 0 para sempre. |
| **bônus baseado em contagem** | Encontra a chave de forma confiável; completa toda a cadeia até o tesouro em cerca de ~40% dos episódios. Funciona, mas é um pouco instável. |
| **bônus por erro de previsão** | Alcança a chave *e* o tesouro pela primeira vez em cerca de 20–25 episódios; conforme `beta` diminui, converge para resolver o problema em cada episódio. |

A figura mostra:
- uma curva de aprendizado: *P(alcançar o tesouro)* ao longo do treinamento,
- uma segunda curva para o marco intermediário *P(pegar a chave)*,
- e **mapas de calor de visitação de estados** da grade — o agente sem curiosidade é uma mancha densa perto do início; os agentes curiosos inundam *ambas* as salas.

## O Mecanismo em Uma Imagem

```
            sem curiosidade                     com bônus de curiosidade
   recompensa: 0 0 0 0 0 0 0 0 ... 0 (+1?)     0 0 0 0 0 0 0 0 ... 0 (+1!)
               └──── nada para aprender ───┘    └ + 0.4 0.3 0.9 0.2 ... ┘ (autogerada,
                                                  densa, aponta "para a novidade")
   resultado:  caminhada aleatória, nunca       varre o mundo sistematicamente,
                encontra o +1                   tropeça no +1, depois o bônus some
```

O bônus de curiosidade transforma o *"eu não vi isso"* em recompensa, de modo que o agente **deliberadamente avança para território inexplorado** em vez de se debater aleatoriamente. E como o bônus diminui à medida que as coisas se tornam familiares (e `beta` decai), uma vez que o agente encontra a recompensa real, ele naturalmente para de perder tempo e começa a explotar (aproveitar o conhecimento).

## Algumas Ressalvas Honestas

- **O problema da "TV barulhenta" (noisy-TV).** Um agente de erro de previsão pode ser hipnotizado por uma fonte de aleatoriedade pura (uma TV mostrando estática, dados rolando) — ele *nunca* consegue prevê-la, então a surpresa nunca desaparece. O truque real do ICM é prever em um *espaço de características aprendido* que ignora coisas que o agente não pode controlar; o RND contorna isso de forma diferente. Nosso mundo de grade determinístico não tem TV barulhenta, então não enfrentamos isso.
- **A curiosidade é um meio, não um fim.** É por isso que o `beta` diminui. Um agente que permanece maximamente curioso para sempre nunca se aquieta para realmente *vencer*.
- **Escalar a exploração profunda ainda é difícil.** Um bônus na recompensa ajuda, mas o Q-learning tabular simples é lento para propagar o otimismo resultante em uma longa cadeia (veja `compare_exploration.py`). Resolver o Montezuma em pixels exigiu mais poder de fogo — RND com rede neural, bootstrapped DQN, Go-Explore.

## Palavras-Chave para Lembrar

| Palavra | Significado |
|---------|-------------|
| **Recompensa intrínseca** | Uma recompensa que o agente gera para si mesmo, separada da recompensa do ambiente |
| **Recompensa extrínseca** | A recompensa real do ambiente (pontos, ganhar/perder) |
| **Recompensa esparsa** | A recompensa é zero em quase todos os lugares; você só a obtém após uma longa sequência correta |
| **Novidade / surpresa** | O quão novo ou inesperado um estado (ou transição) é — o que a curiosidade recompensa |
| **Bônus de contagem** | Novidade ≈ `1/sqrt(contagem de visitas)` — o clássico bônus de exploração |
| **ICM** | Intrinsic Curiosity Module: novidade = erro de previsão de um modelo (em um espaço de características aprendido) |
| **`beta`** | O peso do bônus de curiosidade; geralmente reduzido para 0 para que o agente eventualmente explore |

## Resumo de Uma Frase

> **Um bônus de curiosidade é uma recompensa autogerada pela novidade — ele fabrica um sinal denso de "vá explorar ali" que arrasta o agente por mundos de recompensa esparsa que de outra forma ele nunca resolveria, e depois desaparece educadamente quando tudo se torna familiar.**
