# O Problema do Bandido Multibraço (Multi-Armed Bandit) 🎰

## O que é?

Imagine que você está em uma festa de aniversário e há **10 potes de doces diferentes**. Cada pote tem doces dentro, mas alguns potes têm doces *deliciosos* e outros têm doces *não tão bons*. Você não sabe qual pote é o melhor — você tem que experimentá-los!

Cada vez que você coloca a mão em um pote, você ganha um doce. Sua tarefa é:

> **Conseguir o máximo de doces deliciosos possível!**

Esse é o problema do Bandido Multibraço! Em vez de potes de doces, os cientistas os chamam de "braços" (como os braços de um caça-níquel). Cada braço te dá um prêmio, mas os prêmios são diferentes a cada vez.

---

## A Grande Pergunta: Devo tentar novos potes ou continuar com o meu favorito?

Esta é a parte mais difícil! Digamos que você experimentou o Pote nº 3 e ele era muito bom. Agora você tem uma escolha:

- **Aproveitar (Exploit)**: Continuar escolhendo o Pote nº 3 porque você já sabe que ele é bom.
- **Explorar (Explore)**: Tentar um novo pote — talvez o Pote nº 7 seja ainda *melhor*!

Se você sempre escolher apenas o primeiro pote de que gostou, poderá perder o pote super delicioso. Mas se você *sempre* tentar novos potes, nunca usará o que já aprendeu!

**Exemplo da vida real:** Pense no seu restaurante favorito. Você sempre pede nuggets de frango (exploit!), mas talvez a pizza seja ainda melhor. Se você nunca tentar nada novo, nunca saberá!

---

## A Estratégia Epsilon-Greedy {#the-epsilon-greedy-strategy}

Uma maneira inteligente de resolver isso é chamada de **epsilon-greedy** (epsilon é apenas a letra grega ε):

1. **Na maioria das vezes (digamos 90%)**: Escolha o pote que você *acha* que é o melhor.
2. **Às vezes (digamos 10%)**: Escolha um pote *aleatório* para explorar!

As viagens de exploração de 10% ajudam você a descobrir potes melhores. As viagens de aproveitamento (exploit) de 90% permitem que você use o que já aprendeu.

---

## O Que Nosso Código Encontrou

Testamos 10 braços (potes de doces) com 200 crianças diferentes, 1000 escolhas cada:

| Estratégia | % de Vezes Escolhendo o Melhor Pote |
|----------|----------------------------------|
| **Nunca explorar (ε=0)** | 14,5% — ficou preso cedo, nunca encontrou o melhor! |
| **Explorar 1% das vezes (ε=0,01)** | 37,6% — encontrou o melhor pote lentamente |
| **Explorar 10% das vezes (ε=0,10)** | **74,2%** — aprendeu rápido, escolheu o melhor na maioria das vezes! |

**Lição**: Um pouco de exploração faz muita diferença!

---

## Exemplos da Vida Real

- **Recomendações da Netflix**: A Netflix deve te mostrar um filme que você provavelmente gostará (exploit) ou sugerir algo novo (explore)?
- **Médico escolhendo um tratamento**: Usar o tratamento que geralmente funciona (exploit) ou tentar um novo que pode ser ainda melhor (explore)?
- **Uma abelha encontrando flores**: Ela deve continuar visitando as flores que sabe que têm néctar ou voar para um novo campo?

---

## Palavras-Chave para Lembrar

- **Braço (Arm)**: Uma das opções (como um pote de doces)
- **Recompensa (Reward)**: O que você recebe quando escolhe um braço (como um doce)
- **Aproveitar (Exploit)**: Usar o que você já sabe que é bom
- **Explorar (Explore)**: Tentar algo novo para aprender mais
- **Epsilon (ε)**: A chance de você explorar em vez de aproveitar (exploit)

A grande ideia: **Você deve equilibrar a experimentação de coisas novas com o uso do que você já sabe!**
