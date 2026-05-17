# REINFORCE com Baseline: Cortando o Ruído

## O Problema com o REINFORCE Simples

Imagine que você é um aluno tentando decidir se sua resposta em um teste foi boa.

**Feedback ruim:** "Você tirou 7 pontos!"

7 é bom? Se o máximo for 10, sim! Se todo mundo tirou 9, não! Sem contexto, você não pode saber se deve mudar seu estilo de resposta.

Este é exatamente o problema com o REINFORCE: ele usa **retornos brutos** (G_t) para avaliar as ações. Uma pontuação de retorno total de 200 pontos pode ser incrível ou terrível, dependendo da situação.

---

## A Chegada do Baseline

Um **baseline** b(s) é um ponto de referência: "Que recompensa eu **espero** nesta situação?"

Em vez de perguntar "Esta ação foi boa?", perguntamos:

> **"Esta ação foi melhor do que eu normalmente esperaria?"**

```
Sinal antigo: atualização ∝ G_t
Sinal novo:   atualização ∝ (G_t - b(s_t))
```

**Exemplo da vida real:** Você tirou 85 em um teste de matemática.
- Se a média da turma for 60 → sua resposta foi **25 pontos acima da média** → ótimo!
- Se a média da turma for 90 → sua resposta foi **5 pontos abaixo da média** → precisa melhorar!

A **vantagem** (G_t - b(s)) é positiva quando você se saiu melhor do que o esperado e negativa quando se saiu pior. Este é um sinal de aprendizado muito mais limpo!

---

## O que é o Baseline?

O baseline natural é a **função de valor V(s)**:

> V(s) = "Recompensa total esperada se eu estiver no estado s e seguir minha política atual"

Aprendemos isso com uma **Rede de Valor** separada (também chamada de rede de baseline ou crítico):

```
Estado  →  [128 neurônios]  →  [128 neurônios]  →  V(s)   (número único)
```

Para cada estado que o agente visita, V(s) prevê o retorno esperado. Se o retorno real G_t for maior que V(s), a ação foi melhor do que o esperado!

---

## Duas Redes Aprendendo Juntas

```
O episódio acontece
     ↓
Calcula os retornos reais G_t
     ↓
         ┌─────────────────────────────┐
         │ Vantagem = G_t - V(s_t)      │
         │  +: a ação foi melhor       │
         │  -: a ação foi pior        │
         └─────────────────────────────┘
              ↓                  ↓
    Atualiza a Rede de Política   Atualiza a Rede de Valor
    (torna ações boas mais/       (torna as previsões mais
     menos prováveis)              precisas na próxima vez)
```

**Exemplo da vida real:** Dois amigos vão juntos a um restaurante.

- Amigo 1 (Rede de Valor): "Prevejo que este prato será um 7/10"
- Amigo 2 (Rede de Política): Você prova o prato e dá nota 9/10
- Vantagem = 9 - 7 = +2 → "Isso foi melhor do que o esperado! Peça de novo!"

Na próxima visita, o Amigo 1 atualiza sua previsão para mais perto de 9/10.
O Amigo 2 terá mais probabilidade de pedir esse prato da próxima vez.

---

## Por que isso Reduz a Variância?

**Prova matemática (intuição):**

Sem baseline: `gradiente ∝ ∇log π(a|s) × G_t`

Os valores de G_t variam muito de episódio para episódio:
```
Episódio 1: G = [45, 44, 43, ..., 1]   (jogo médio)
Episódio 2: G = [500, 499, ..., 1]     (jogo ótimo!)
Episódio 3: G = [12, 11, ..., 1]       (jogo terrível)
```

As estimativas de gradiente saltam descontroladamente porque G_t é grande e ruidoso.

Com baseline: `gradiente ∝ ∇log π(a|s) × (G_t - V(s_t))`

A vantagem (G_t - V(s_t)) é muito menor e centrada perto de zero:
```
Episódio 1: vantagem ≈ [-2, +1, -3, ..., 0]   (pequena, centrada)
Episódio 2: vantagem ≈ [+10, +8, ..., +3]      (este jogo FOI ótimo)
Episódio 3: vantagem ≈ [-5, -6, ..., -2]       (este jogo FOI ruim)
```

**Exemplo da vida real:** Medir sua velocidade de corrida.
- Sem baseline: "Eu corri a 8 km/h" (sem sentido sem contexto)
- Com baseline: "Eu corri 2 km/h MAIS RÁPIDO que a minha média" (claramente bom!)

A vantagem é sempre uma comparação — é naturalmente menor e mais estável.

---

## Crucial: Sem Viés!

O baseline não muda O QUE o algoritmo aprende — apenas o QUÃO RÁPIDO e ESTÁVEL ele aprende.

**Por quê?** Porque a vantagem esperada é sempre 0 em valor esperado:

> E[G_t - V(s_t)] = E[G_t] - V(s_t) = V(s_t) - V(s_t) = 0

Qualquer b(s) que não dependa da ação funciona como um baseline válido!

**Exemplo da vida real:** Avaliar com base em uma curva não muda quem teve o melhor desempenho — apenas torna as pontuações mais fáceis de interpretar. O ranking permanece o mesmo; apenas a escala muda.

---

## Os Resultados

```
Sem baseline   — Média final 100-ep: 500,0, variância do grad: 599,3
Com baseline — Média final 100-ep: 491,4, variância do grad: 578,8
```

Ambos os métodos alcançam um desempenho quase perfeito no CartPole, mas observe:
1. A **variância do gradiente** é mensurável (o gráfico à direita mostra a variância durante o treinamento)
2. Com o baseline, o agente atinge alto desempenho de forma **mais confiável** — ocorrem menos quedas para recompensas baixas durante o treinamento

A redução da variância é mais dramática em ambientes mais difíceis (LunarLander, MuJoCo).

---

## Equações Principais

```
Valor do baseline: V(s) ← V(s) + α(G_t - V(s))   [minimizar MSE]
Gradiente da política: θ ← θ + α ∇log π(a_t|s_t) · (G_t - V(s_t))
Vantagem:          A_t = G_t - V(s_t)
```

---

## Principais Conclusões

| Conceito | Em Português Simples |
|---------|---------------|
| **Baseline b(s)** | Recompensa esperada no estado s — nosso ponto de referência |
| **Vantagem A_t** | "Esta ação foi melhor do que o esperado?" |
| **Rede de Valor** | Uma rede neural que aprende a prever retornos esperados |
| **Redução de variância** | Menos ruído nas estimativas de gradiente → aprendizado mais estável |
| **Não enviesado** | O baseline não altera a política-alvo em média; apenas torna o sinal de aprendizado menos ruidoso e mais estável |

---

## O que vem a seguir?

O baseline é, na verdade, o começo de algo muito mais poderoso: os métodos **Actor-Critic**.

Em vez de computar V(s) apenas no final de um episódio, o Actor-Critic atualiza V(s) a cada passo usando o aprendizado de **Diferença Temporal (Temporal Difference)**. Isso torna as atualizações muito mais rápidas e permite que o agente aprenda com episódios incompletos!

Veja `a2c_lunarlander.py` para a implementação completa de Actor-Critic.
