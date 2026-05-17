# Deep Q-Network (DQN) do Zero 🧠

## O Problema com o Linear

Lembra da nossa fórmula linear de antes?

> Pontuação = w₁ × posição_do_carrinho + w₂ × velocidade_do_carrinho + w₃ × ângulo_da_haste + w₄ × velocidade_da_haste

Isso funciona bem para o CartPole, mas e quanto a um videogame onde você vê milhares de pixels? Você não pode escrever uma receita simples para isso!

Precisamos de algo que possa analisar situações complicadas e descobrir a melhor ação. Esse algo é uma **rede neural**.

---

## O que é uma Rede Neural?

Pense no seu cérebro. Milhões de pequenas células chamadas neurônios conversam entre si. Quando você toca em algo quente, os neurônios enviam sinais: "QUENTE! → Tire a mão AGORA!" Cada neurônio passa a informação adiante e, juntos, eles tomam uma decisão inteligente.

Uma **rede neural em um computador** funciona da mesma maneira:

```
Camada de Entrada   Camada Oculta 1   Camada Oculta 2   Camada de Saída
[pos carrinho]  →   [128 neurônios]  →  [128 neurônios]  →  [pontuação ESQUERDA]
[vel carrinho]  →   [  ...       ]     [  ...       ]     [pontuação DIREITA]
[ângulo haste]  →
[vel haste]     →
```

Cada seta tem um **peso** (quão forte é essa conexão). Existem milhares desses pesos — e a rede aprende TODOS eles!

**Exemplo da vida real:** Um chef de cozinha em um restaurante prova sua comida e ajusta centenas de ingredientes de uma vez. Cada papila gustativa é como um neurônio e, juntas, elas dizem ao chef "adicione mais sal" ou "menos pimenta". Treinar a rede é como o chef aprendendo ao longo de milhares de refeições.

---

## DQN = Deep Q-Network

A **DQN** (Deep Q-Network) foi inventada pela DeepMind em 2013. Eles pegaram a antiga fórmula do Q-learning e trocaram a tabela Q por uma rede neural!

Em vez de:
> Tabela-Q[estado][ação] = pontuação

Temos:
> Rede-Q(estado) → [pontuação_para_esquerda, pontuação_para_direita]

A rede recebe o estado como entrada e gera os valores Q para TODAS as ações de uma vez. Isso é muito mais eficiente do que calculá-los separadamente!

---

## Este Script: A Versão "Ingênua" (Naive)

Este script mostra a DQN **sem** nenhum truque especial. Ele apenas:
1. Vê o estado
2. Pergunta à rede "quão bom é ir para a esquerda? quão bom é ir para a direita?"
3. Realiza a ação com a pontuação mais alta
4. Recebe uma recompensa, atualiza a rede

**Isso é intencionalmente instável!** Pense nisso como um estudante que esquece imediatamente suas lições anteriores toda vez que aprende algo novo. A rede se atualiza após cada passo, o que causa o caos.

**Exemplo da vida real:** Imagine aprender a cozinhar mudando sua receita inteira após cada mordida. Você pode ir de "muito salgado" para "sem sal nenhum" e depois para "muito salgado de novo", nunca chegando à quantidade certa. É isso que acontece aqui!

---

## O Que Você Verá

Quando você executar `dqn_cartpole.py`:
- As pontuações podem variar muito (aprendizado instável)
- Às vezes o agente fica muito bom e depois esquece tudo
- O gráfico de perda (loss) mostra oscilações bruscas

**Isso é esperado!** Isso mostra POR QUE precisamos de melhorias — experience replay e redes alvo (target networks). Eles vêm a seguir!

---

## O Truque ε-Greedy 🎲

O robô nem sempre escolhe a melhor ação. Às vezes ele escolhe aleatoriamente!

Por quê? Porque se ele sempre escolher o que parece melhor, pode nunca descobrir opções melhores.

> Com probabilidade ε (epsilon): escolha uma ação ALEATÓRIA (explorar!)
> Com probabilidade 1-ε: escolha a MELHOR ação conhecida (aproveitar!)

Começamos com ε = 1.0 (100% aleatório) e diminuímos lentamente para ε = 0.01 (1% aleatório). Dessa forma, o robô explora muito no início e depois foca no que aprendeu.

**Exemplo da vida real:** Ao visitar uma cidade nova, você pode experimentar restaurantes aleatórios no início (explorar). Depois de um tempo, você volta aos seus favoritos (aproveitar). Mas, ocasionalmente, você ainda experimenta algo novo, caso haja uma joia escondida!

---

## Vocabulário Chave

| Palavra | Significado |
|------|---------|
| **Rede Neural** | Camadas de neurônios matemáticos conectados que aprendem com os dados |
| **Deep (Profunda)** | Mais de uma camada oculta (daí o termo "deep learning") |
| **DQN** | Deep Q-Network — usa rede neural em vez de tabela Q |
| **ε-Greedy** | Estratégia: explorar aleatoriamente às vezes, aproveitar o melhor conhecimento em outras |
| **Instabilidade** | A rede continua "esquecendo" porque as atualizações interferem umas nas outras |

---

## O Que Está Faltando (e Por Que Isso Importa)

Esta DQN ingênua tem dois grandes problemas:

1. **Atualizações correlacionadas**: Cada experiência vem em ordem (passo 1, passo 2, passo 3...). Se o passo 5 foi ruim, TODAS as atualizações próximas ficam confusas juntas.
   
2. **Alvo móvel**: Após cada atualização, a rede muda. Mas a próxima atualização usa a MESMA rede para calcular qual deveria ser o alvo. É como atirar em um alvo móvel!

Esses problemas são resolvidos pelo **Experience Replay** e pelas **Redes Alvo (Target Networks)** nos próximos scripts. Juntos, eles transformam a DQN de uma iniciante instável em uma campeã de jogos!
