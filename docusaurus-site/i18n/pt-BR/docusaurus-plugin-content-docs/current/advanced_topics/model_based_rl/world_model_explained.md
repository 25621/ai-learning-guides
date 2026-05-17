# Treinando um Modelo de Mundo: Ensinando o Agente a Sonhar 🌍

## O que é um "Modelo de Mundo"?

Um **modelo de mundo** (world model) é a *cópia interna do universo* do agente. Forneça a ele um
estado e uma ação, e ele preverá o que acontecerá a seguir:

```
(estado, ação)  ──►  Rede Neural  ──►  (proximo_estado, recompensa)
```

Não é o mundo real — é um **simulador que o agente construiu para si mesmo** ao
observar a realidade e aprender a imitá-la.

Uma vez treinado, o modelo permite que o agente faça perguntas do tipo "e se" sem tomar
qualquer ação real:

> *"Se eu empurrar para a esquerda agora e depois para a direita duas vezes, onde vou parar? O poste
> vai cair?"*

O agente pode ponderar cem planos dentro de seu modelo no tempo que levaria para
fazer um único movimento real. Esse é o ponto principal.

---

## Uma Analogia da Vida Real

Pense em como *você* resolve um quebra-cabeça. Você não move fisicamente cada peça
em cada encaixe. Você **imagina** o que acontece se a peça A for colocada aqui. Se essa
simulação mental parecer errada, você a rejeita antes de levantar um dedo.

Seu cérebro tem um modelo de mundo aprendido — construído a partir de anos vendo como os objetos
se comportam — que permite simular resultados antes de se comprometer.

Outros exemplos:

- **Um jogador de xadrez** imagina jogadas vários turnos à frente.
- **Um motorista** pensando: "Se eu frear agora, o carro de trás terá espaço suficiente."
- **Uma criança** empilhando blocos: "Se eu colocar o grande em cima, a torre vai
  balançar." (Eles aprenderam este modelo ao derrubar torres anteriormente.)

Em todos os casos, **um modelo mental + imaginação = melhores decisões com menos risco**.

---

## Como o Agente Constrói seu Modelo?

Ele apenas **observa**. Especificamente:

1. **Coleta dados.** Deixa qualquer política (mesmo aleatória) interagir com o
   ambiente real por um tempo. Salva cada transição:
   ```
   (estado, ação, recompensa, proximo_estado)
   ```
2. **Treina uma rede neural** para prever `proximo_estado` e `recompensa` a partir de
   `(estado, ação)`. Isso é aprendizado supervisionado: cada transição salva é um
   exemplo rotulado onde a entrada é "o que o agente viu e fez" e o rótulo é
   "o que realmente aconteceu depois."
3. **Valida.** Reserva 10% dos dados e verifica as previsões do modelo
   em relação às reais. Um erro baixo significa que o modelo capturou a
   **dinâmica** do ambiente: como os estados mudam após as ações.

O truque que usamos: em vez de prever o `proximo_estado` diretamente, prevemos o
**delta** `proximo_estado − estado`. A maior parte da física é incremental ("o carrinho se moveu
um pouquinho"), e alvos pequenos são mais amigáveis para as redes neurais.

---

## Nossa Configuração

| Escolha | Valor | Por quê |
|--------|-------|-----|
| Ambiente | `CartPole-v1` | Estado 4-D, 2 ações — fácil de modelar |
| Dados | 20.000 transições de uma política aleatória | Ampla cobertura do espaço de estados |
| Rede | MLP, 2 × 128 ReLU ocultas | MLP = Perceptron de Múltiplas Camadas (rede neural "vanilla" padrão). Duas camadas ocultas de 128 neurônios usando ativações ReLU. Capacidade suficiente, rápida de treinar. |
| Perda (Loss) | MSE em `(delta_estado, recompensa)` | MSE = Erro Quadrático Médio (média dos erros de previsão ao quadrado). Perda de regressão padrão. |
| Otimizador | Adam, lr = 1e-3, 30 épocas | Adam = otimizador adaptativo (ajusta as taxas de aprendizado por parâmetro automaticamente). Pronto para uso, não precisa de ajuste especial. |

O treinamento completo termina em poucos segundos em CPU.

---

## Como é um Resultado "Bom"?

Dois diagnósticos importam:

### 1. Precisão de passo único (MSE de validação)

Isso é "quão bem o modelo prevê UM passo no futuro?". Após 30
épocas, você deve ver o MSE de validação na faixa de **1e-4 a 1e-3**. Isso é
minúsculo — ângulos do poste e posições do carrinho são precisos até algumas casas decimais.

### 2. **Erro acumulado** em rollouts de k-passos

Este é o teste *real*. Pegue um estado, passe-o pelo modelo, depois pegue a
sua previsão e passe-a de volta pelo modelo — por `k` passos seguidos.
O erro cresce porque cada passo adiciona um pouco de ruído sobre a
previsão anterior.

```
Passo  1:  Erro L2 ≈ 0,01   (quase perfeito)
Passo  5:  Erro L2 ≈ 0,05
Passo 10:  Erro L2 ≈ 0,15
Passo 20:  Erro L2 ≈ 0,40   (deriva visível)
```

*(Erro L2 = distância euclidiana entre o próximo estado previsto e o real —
pense nisso como "o quão longe está o palpite do modelo no espaço de estados 4-D?")*

**Por que isso importa.** Se planejarmos 15 passos à frente com o modelo, o estado
*exato* no passo 15 estará errado — mas se a classificação relativa de "bons planos
vs. maus planos" for preservada, o planejamento ainda funciona. (Isso é o que
o `model_based_planning.py` explora.)

O gráfico em `outputs/world_model.png` mostra ambos os diagnósticos lado a lado: a
curva de perda de treinamento desce bem em uma escala logarítmica, e a curva de erro de
rollout sobe de forma constante.

---

## Por que prever o *Delta*?

Compare duas maneiras de formular o mesmo problema para a rede:

| Alvo | Magnitude típica | Fácil ou difícil? |
|--------|------------------:|--------------|
| `proximo_estado`        | 0–2,4 (pos carrinho) | A rede deve reproduzir a posição **e** a pequena mudança |
| `proximo_estado - estado`| ~0,02            | A rede apenas aprende a pequena mudança |

Prever o delta também significa: se a rede produzir zeros (como uma rede iniciante
não treinada costuma fazer), a previsão é simplesmente "nada se moveu" — um padrão
sensato e seguro para um único passo de tempo. Em contraste, prever o `proximo_estado`
absoluto diretamente produziria inicialmente valores de lixo completamente aleatórios,
tornando o início do treinamento altamente instável.

---

## O que Ganhamos com isso

Um modelo de mundo treinado é a base para:

- **Planejamento** — busca sobre sequências de ações imaginadas (veja
  `model_based_planning.py`).
- **Aumentação tipo Dyna** — treinar uma rede Q em dados imaginados para
  multiplicar a eficiência de amostragem.
- **Curiosidade / exploração** — visitar estados que o modelo não consegue prever bem.
- **Artigos Dreamer / World-Models** — treinar uma *política* inteiramente dentro do
  modelo com zero interação no mundo real além da coleta inicial de dados.

---

## Limites e Cuidados

- **Deriva fora da distribuição.** O modelo conhece apenas a parte do mundo que
  ele viu. Planeje de forma agressiva demais e você acabará em regiões que o modelo nunca
  visitou — as previsões lá são pura fantasia.
- **Erro acumulado.** O planejamento sobre longos **horizontes** (muitos passos no futuro) não é confiável devido ao acúmulo de erros, como mostra o gráfico.
  Sistemas modernos abordam isso usando **ensembles probabilísticos** (treinar vários modelos e verificar se eles concordam, como no PETS ou Dreamer) para que o planejador
  saiba exatamente o *quão incerto* o modelo está em cada passo e possa evitar caminhos arriscados e desconhecidos.
- **Ambientes estocásticos.** Um regressor determinístico padrão prevê apenas o resultado *médio*
  e perde completamente a dispersão de resultados possíveis. Ambientes complexos do mundo real exigem modelos
  probabilísticos (como aqueles com saídas Gaussianas ou **modelos estocásticos latentes** — redes que
  codificam o estado do mundo como uma distribuição de probabilidade em um espaço comprimido,
  permitindo capturar a aleatoriedade genuína em vez de apenas tirar a média) para representar com precisão a incerteza e a aleatoriedade.

---

## Palavras-Chave

| Termo | Em Português Simples |
|------|---------------|
| **Modelo de mundo** | Uma rede neural que imita o ambiente |
| **Dinâmica (Dynamics)** | A função `(s, a) → s'` |
| **Modelo de recompensa** | A função `(s, a) → r` (geralmente incluída) |
| **Previsão de passo único**| O que o modelo produz a partir de um estado real |
| **Rollout** | Previsões repetidas de passo único, realimentando as saídas |
| **Erro acumulado** | Pequenos erros que crescem ao longo de um rollout |

---

## Resumo em uma Frase

> **Um modelo de mundo é uma pequena cópia neural do universo que o agente pode
> consultar — e dentro da qual pode sonhar — antes de arriscar uma ação real.**

A seguir: `model_based_planning.py` coloca este modelo para trabalhar na tomada de decisões real.
