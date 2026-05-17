# Controle de Monte Carlo para Blackjack 🃏

## O Que É?

Você já jogou algum jogo de cartas onde precisava decidir: **"Pego mais uma carta ou estou satisfeito com o que tenho?"**

O **Blackjack** (também chamado de "21") é exatamente isso! Você quer que suas cartas somem o mais próximo possível de 21, sem ultrapassar. Se você passar de 21, você "estoura" (bust) e perde!

O **Controle de Monte Carlo** é como um robô aprende a jogar Blackjack — jogando *milhares de partidas completas* e lembrando o que funcionou e o que não funcionou.

---

## A Grande Ideia: Aprender com Histórias Completas

O termo "Monte Carlo" vem do famoso cassino em Mônaco. Na matemática, significa: **usar experimentos aleatórios para aprender algo**.

Veja como funciona:

1. **Jogue uma partida inteira** (um episódio completo) usando qualquer estratégia que você tenha.
2. **Veja o que aconteceu**: Você ganhou? Perdeu? Empatou?
3. **Trabalhe de trás para frente**: Pedir outra carta (hit) quando estava com 17 foi uma boa ideia? E com 14?
4. **Atualize sua memória**: Lembre-se se cada decisão levou à vitória ou à derrota.

Faça isso por **500.000 jogos** e você ficará muito bom!

**Exemplo da vida real:** Imagine aprender a cozinhar preparando 500.000 refeições. Cada vez, você se lembra exatamente do que fez — e se a comida ficou boa. Depois de tentativas suficientes, você sabe: "Adicionar sal demais neste passo sempre deixava o prato ruim." O Monte Carlo funciona da mesma maneira!

---

## Diferença Principal em Relação ao SARSA e Q-Learning

O SARSA e o Q-Learning atualizam seus conhecimentos **após cada passo individual** (mesmo no meio do episódio). O Monte Carlo espera até que o **episódio inteiro termine** e, então, olha para tudo o que aconteceu.

| Método | Atualiza quando? | Precisa do episódio completo? |
|--------|------------------|------------------------------|
| **TD (SARSA, Q-Learning)** | Após cada passo | Não |
| **Monte Carlo** | Após cada episódio completo | Sim |

Isso torna o Monte Carlo mais simples de entender, mas ele não consegue aprender até que cada episódio termine.

---

## O Estado no Blackjack

O robô observa 3 coisas a cada turno:
1. **Meu total de cartas** (12 a 21)
2. **Qual carta o crupiê está mostrando?** (Ás a 10)
3. **Eu tenho um Ás utilizável?** (Um Ás pode valer 1 ou 11)

A partir dessas 3 informações, ele decide: **Pedir carta (Hit) ou Parar (Stick)**?

---

## O Que Nosso Código Encontrou

Após **500.000 jogos** de Blackjack:

| Resultado | Porcentagem |
|-----------|-------------|
| **Vitórias** | **43,1%** |
| **Empates** | 8,9% |
| **Derrotas** | 48,0% |

Isso está próximo da "estratégia básica" matematicamente ideal (cerca de 42-43% de vitórias)! O robô aprendeu quando pedir carta e quando parar — apenas jogando e lembrando.

A política aprendida mostra:
- **Pedir carta** (Hit) quando seu total é baixo (é improvável que você estoure)
- **Parar** (Stick) quando seu total é alto (você pode estourar se pegar outra carta)
- Ter um **Ás utilizável** permite que você seja mais agressivo (ele pode mudar de 11 para 1 se necessário)

---

## Exemplos da Vida Real

- **Previsão do tempo**: Simulações de Monte Carlo executam milhares de cenários "e se" para prever o tempo de amanhã.
- **Modelagem do mercado de ações**: Analistas simulam milhares de futuros possíveis para estimar o risco.
- **Aprender a jogar xadrez**: Um jogador analisa partidas inteiras (não apenas lances isolados) para entender qual estratégia levou à vitória.

---

## Palavras-Chave para Lembrar

- **Episódio**: Um jogo completo do início ao fim
- **Retorno (G)**: Recompensa total coletada de um ponto do jogo até o fim
- **MC de cada visita (Every-visit MC)**: Atualiza a pontuação de um estado toda vez que você o visita em um episódio
- **Sem bootstrapping**: O Monte Carlo não usa estimativas de valores futuros — ele espera pelo resultado real!
- **Política ε-soft** (ε = epsilon): Geralmente faz a melhor ação conhecida, mas às vezes explora aleatoriamente

A grande ideia: **O Monte Carlo aprende jogando muitas partidas completas. É como aprender com a experiência — você se lembra de tudo o que aconteceu e descobre o que levou à vitória!**
