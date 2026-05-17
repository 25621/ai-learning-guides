# Q-Learning Linear para CartPole 🎪

## O Que é CartPole?

Imagine uma vassoura equilibrada verticalmente no seu dedo. Se você mover o dedo para a esquerda ou para a direita apenas um pouco, poderá evitar que a vassoura caia. Isso é o **CartPole**!

Um pequeno robô senta em um carrinho (uma caixa sobre rodas) e tem uma haste (pole) apontando para cima. O robô só pode empurrar o carrinho para a **esquerda** ou para a **direita**. Ele tem que aprender a manter essa haste equilibrada o maior tempo possível — exatamente como você equilibrando uma vassoura!

O robô pode ver 4 coisas sobre o mundo:
1. Onde o carrinho está
2. Quão rápido o carrinho está se movendo
3. O quanto a haste está inclinada
4. Quão rápido a haste está inclinada

---

## O Grande Problema: Estados Demais!

Lembra do Q-learning da Fase 2? Ele usava uma grande tabela para lembrar quão boa é cada ação em cada situação (estado). Isso funcionou muito bem para o Frozen Lake — havia apenas 16 quadrados no gelo.

Mas o CartPole é diferente! O carrinho pode estar em **qualquer posição**, movendo-se em **qualquer velocidade**, com a haste em **qualquer ângulo**. Existem basicamente **infinitos estados possíveis**! Não podemos fazer uma tabela com infinitas linhas. Precisaríamos de um caderno do tamanho do universo!

**Exemplo da vida real:** Imagine que você está aprendendo a andar de bicicleta. Você não pode memorizar cada oscilação possível — existem muitas! Em vez disso, você aprende uma **regra**: "quando eu me inclinar para a esquerda, empurro para a direita; quando me inclinar para a direita, empurro para a esquerda". Uma regra simples funciona para TODAS as oscilações.

---

## A Solução: Uma Fórmula Mágica

A **aproximação de funções lineares** substitui a tabela gigante por uma **fórmula minúscula**:

> **Pontuação(situação, ação) = w₁ × posição_do_carrinho + w₂ × velocidade_do_carrinho + w₃ × ângulo_da_haste + w₄ × velocidade_da_haste**

- Os números `w` são chamados de **pesos** (weights) — eles são como botões que você pode girar.
- Aprendemos **pesos diferentes para cada ação** (empurrar para a esquerda e empurrar para a direita).
- A fórmula fornece uma pontuação para o quão boa é cada ação no momento.

**Exemplo da vida real:** Pense em uma receita simples: "1 xícara de farinha + 2 ovos + ½ xícara de manteiga". Os pesos (1, 2, ½) dizem o quanto cada ingrediente importa. Estamos aprendendo a receita para tomar boas decisões!

---

## Como Ele Aprende?

O robô tenta coisas, recebe feedback e ajusta os pesos:

1. **O robô empurra o carrinho** (escolhe a ação com a maior pontuação).
2. **A física acontece** (a haste inclina um pouco, o carrinho se move).
3. **O robô recebe uma recompensa** (+1 para cada passo que a haste permanece de pé, 0 se ela cair).
4. **O robô pergunta:** "O resultado real foi melhor ou pior do que eu previ?".
5. **O robô ajusta os pesos** para ficar mais perto da realidade na próxima vez.

Esta é a **Atualização TD de Semi-Gradiente** (Semi-Gradient TD Update) — um nome sofisticado para "dar um empurrãozinho na receita com base na surpresa".

> **Novo peso = Peso antigo + Taxa de aprendizado × (O que realmente aconteceu − O que eu previ) × Característica**

---

## O Que Nosso Código Encontrou

Ao executar `linear_q_cartpole.py`, o robô:

- Começa terrível (a haste cai em 10–30 passos).
- Aprende gradualmente bons pesos ao longo de 3.000 tentativas.
- Eventualmente mantém a haste equilibrada por 100–400+ passos!

O gráfico mostra a **curva de aprendizado** — como a pontuação melhora ao longo do tempo. Será irregular (o aprendizado nunca é suave!), mas a tendência deve ser de subida.

---

## Por Que Isso é Legal (e Limitado!)

**Legal:** Uma fórmula minúscula com apenas 8 números (4 pesos × 2 ações) pode equilibrar uma haste! Nenhuma tabela gigante é necessária.

**Limitado:** A fórmula é simples demais para tarefas complexas. Ela assume que números maiores sempre significam efeitos maiores (o que nem sempre é verdade). Para jogos mais difíceis como Atari, precisamos de **redes neurais** — que é o que a DQN faz!

---

## Vocabulário Chave

| Palavra | Significado |
|------|---------|
| **Característica (Feature)** | Uma coisa mensurável sobre o mundo (ex: ângulo da haste) |
| **Peso (Weight)** | O quanto uma característica afeta a decisão |
| **Linear** | A fórmula é apenas multiplicação e adição (sem curvas complicadas) |
| **Semi-gradiente** | Atualizar os pesos seguindo a direção de menos erro |
| **Aproximação de funções** | Usar uma fórmula em vez de uma tabela |

---

## O Que Vem a Seguir?

A aproximação linear é como usar uma régua reta para desenhar uma curva — funciona bem para formas simples, mas não para as complexas. Para jogos de Atari com milhões de situações possíveis, precisamos de **Deep Q-Networks (DQN)** — redes neurais que podem aprender padrões muito mais complexos. Isso está no próximo arquivo!
