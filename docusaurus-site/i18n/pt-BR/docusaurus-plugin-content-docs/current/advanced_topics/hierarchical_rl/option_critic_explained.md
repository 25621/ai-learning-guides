# Arquitetura Option-Critic

## A Grande Ideia: Trabalhar por Capítulos, Não Palavra por Palavra

Imagine que você está escrevendo um romance. Você não planeja cada palavra antes de começar. Em vez disso, você pensa em **capítulos**: "O Capítulo 1 apresenta o herói. O Capítulo 2 é a missão. O Capítulo 3 é o confronto final". Dentro de cada capítulo, você define os detalhes à medida que avança.

É exatamente assim que a arquitetura Option-Critic pensa sobre decisões.

---

## O que é um Agente "Plano" (Flat)?

Um agente de RL normal (como os das Fases 3 e 4 do currículo) decide uma ação de cada vez, a cada passo. É como um GPS que recalcula toda a rota do zero cada vez que você se move um metro. Funciona, mas é exaustivo e lento para aprender.

---

## O que é uma "Opção" (Option)?

Uma **opção** é uma **habilidade nomeada** — uma minipolítica que o agente pode executar por vários passos seguidos antes de devolver o controle.

Pense nisso como um gerente delegando para especialistas:

| Quem | O que eles fazem |
|-----|-------------|
| **Gerente (metapolítica)** | Decide *qual* especialista enviar para um trabalho |
| **Especialista A** | Especialista em navegar na sala superior esquerda |
| **Especialista B** | Especialista em atravessar portas |
| **Especialista C** | Especialista em avançar em direção ao objetivo |
| **Especialista D** | Generalista de apoio |

O gerente escolhe um especialista. O especialista trabalha até decidir que terminou (isso é chamado de **terminação**). Então o gerente escolhe novamente.

---

## As Três Partes Móveis

Cada opção hat três componentes — pense neles como a **descrição do trabalho** do especialista:

1. **Iniciação**: Quando este especialista pode ser chamado? *(ex: "Especialista A só ativa perto da sala superior esquerda.")*
2. **Política intra-opção**: O que o especialista faz enquanto está trabalhando? *(ex: "Caminhar em direção ao canto superior esquerdo.")*
3. **Terminação**: Quando o especialista devolve o controle? *(ex: "Parar quando chegar a uma porta.")*

A beleza do Option-Critic é que todos os três são **aprendidos automaticamente** — você não cria os especialistas manualmente. O algoritmo descobre sozinho que é útil ter uma opção para cada sala, ou uma para correr para o objetivo.

---

## Um Dia na Vida de um Agente Option-Critic

1. O agente entra em uma nova sala (estado).
2. O **Gerente** olha para a sala e escolhe uma opção — digamos, a Opção 2.
3. O **especialista da Opção 2** assume o controle: caminha em direção à porta, passo a passo.
4. Em algum momento, a Opção 2 diz "terminei aqui" (terminação).
5. O **Gerente** acorda, escolhe uma nova opção para a nova situação.
6. Repete.

Compare isso com o agente plano: o agente plano se angustia com cada passo individual. O Option-Critic delega trechos inteiros de comportamento, permitindo que cada especialista se torne bom em sua tarefa específica.

---

## Por que Isso Ajuda?

Em um labirinto, o agente precisa alcançar um objetivo que pode estar a 30–50 passos de distância. Com o aprendizado plano, cada passo no caminho é igualmente "invisível" até que a recompensa finalmente chegue ao fim — esse sinal tem que viajar de volta através de dezenas de passos.

Com as opções, o caminho se divide em **subtarefas**. Cada subtarefa recebe seu próprio sinal de minirrecompensa (chegar à porta, entrar na próxima sala). O aprendizado se propaga através de segmentos mais curtos. **O agente aprende mais rápido em problemas que exigem muitos passos.**

Esta é a ideia central de todo o RL Hierárquico — e o Option-Critic é uma de suas implementações mais limpas.

---

## O que Nosso Código Faz

O script `option_critic.py` coloca um agente Option-Critic em um **mundo de grade 7x7** com um objetivo fixo. O agente começa em qualquer lugar da grade e deve navegar até a célula do objetivo.

O agente tem quatro opções e deve aprender simultaneamente:

- Uma política para cada opção (para onde caminhar)
- Quando terminar cada opção (condição de terminação)
- Uma metapolítica para escolher entre as opções

A recompensa usa **shaping baseado em potencial** (potential-based shaping) — o agente recebe um pequeno bônus a cada passo que se aproxima do objetivo, além de +1 por alcançá-lo. Este feedback denso torna o aprendizado estável o suficiente para ver as opções funcionando em 2.500 episódios.

Nenhum humano jamais diz o que cada opção deve fazer. O algoritmo descobre em quais áreas da grade cada opção se especializa.

---

## O que os Gráficos Mostram

![Curvas de Aprendizado Option-Critic](outputs/option_critic.png)

**Esquerda — Retorno com Shaping (Shaped Return):** Um retorno mais alto significa que o agente está alcançando o objetivo de forma mais confiável *e* fazendo caminhos mais curtos (o shaping dá um bônus por passo mais próximo). A curva subindo e depois se estabilizando mostra as opções aprendendo a se coordenar.

**Direita — Passos para o Objetivo (Steps to Goal):** Menos passos significam que o agente encontrou um caminho mais eficiente. A tendência de queda mostra as opções amadurecendo em habilidades coerentes que guiam o agente mais diretamente para o objetivo.

As curvas suavizadas mostram a tendência geral em janelas de 100 episódios — algum ruído é normal em RL, especialmente quando vários componentes (opções, terminação, metapolítica) estão aprendendo simultaneamente.

---

## Resumo em Uma Frase

> **Option-Critic ensina um agente a trabalhar com habilidades em vez de passos individuais — um gerente escolhe qual especialista executa, cada especialista faz seu trabalho e todo o sistema aprende junto a partir do mesmo sinal de recompensa.**
