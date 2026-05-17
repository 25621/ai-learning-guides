# REINFORCE: Ensinando um Robô a Fazer Melhores Escolhas

## O que Estamos Tentando Fazer?

Imagine que você tem um robô jogando um videogame. A cada segundo, o robô deve escolher:
**"Devo apertar o botão ou não?"**

Em vez de memorizar cada situação em uma tabela (como no Q-learning), queremos que o robô aprenda uma **receita** — um conjunto de regras que diga diretamente: "Nesta situação, faça esta ação."

Essa receita é chamada de **política** (π, pi). No aprendizado por reforço, π significa "a regra para escolher ações."

---

## A Forma Antiga vs. A Forma Nova {#the-old-way-vs-the-new-way}

**Forma antiga (Q-learning / DQN):** Aprender quão BOA é cada ação (valores Q) e depois escolher a melhor.
> "Apertar para a ESQUERDA tem pontuação 7, apertar para a DIREITA tem pontuação 5 → aperte para a ESQUERDA!"

**Forma nova (Gradiente de Política):** Aprender diretamente qual ação ESCOLHER.
> "Quando o poste se inclinar para a direita, aperte para a DIREITA com 80% de chance e para a ESQUERDA com 20% de chance."
*(A palavra **Gradiente** refere-se ao "passo" matemático que damos para ajustar lentamente essas probabilidades na direção certa.)*

**Exemplo da vida real:** Aprender a andar de bicicleta.
- A forma antiga: calcular a *pontuação exata* para "inclinar para a esquerda 5 graus" vs "inclinar para a esquerda 7 graus."
- A forma nova: apenas praticar até que seu **corpo** aprenda — use o pé que parecer certo!

---

## Como o REINFORCE Funciona?

O REINFORCE observa o robô jogar uma partida completa do início ao fim (um **episódio**) e depois pergunta: "Quais ações levaram a uma boa pontuação? Vamos fazer mais dessas!"

### Passo a Passo

**1. Jogar um episódio**

O robô faz escolhas e coleta experiência:
```
Passo 1: Estado = [poste inclinando para a direita] → Ação = apertar DIREITA → Recompensa = +1
Passo 2: Estado = [poste quase equilibrado] → Ação = apertar DIREITA → Recompensa = +1
Passo 3: Estado = [poste inclinando para a esquerda] → Ação = apertar ESQUERDA → Recompensa = +1
...
Passo 47: Estado = [o poste caiu!] → Episódio encerrado
```

**2. Calcular os retornos**

Para cada passo, calcula-se G_t — a **recompensa total daquele ponto em diante**:
```
G_no_passo_47 = 1
G_no_passo_46 = 1 + 0,99 × 1 = 1,99
G_no_passo_45 = 1 + 0,99 × 1,99 = 2,97
...
G_no_passo_1  = 47 (aproximadamente — retorno maior porque foi desde o início)
```

O **fator de desconto** γ = 0,99 significa que as recompensas distantes no futuro contam um pouco menos.

**Exemplo da vida real:** Ganhar uma estrela dourada no primeiro dia de aula é mais emocionante do que saber que você *pode* ganhar uma no centésimo dia. As recompensas futuras são levemente "descontadas".

**3. Atualizar a política**

Para cada ação tomada:
> Se G_t foi ALTO (aquela ação levou a um ótimo resultado): **faça-a mais!**
> Se G_t foi BAIXO (aquela ação levou a um resultado ruim): **faça-a menos!**

A matemática: `loss = -log_prob(action) × G_t`

Pegar o gradiente e atualizar a política é como dizer ao robô:
*"Aquela ação que você tomou no passo 20? Você deve fazê-la 5% mais vezes na próxima vez!"*

---

## O que é uma Rede de Política?

Em vez de uma tabela, usamos uma **rede neural** para representar a política.

```
Observação        Rede de Política       Probabilidades de Ação
[pos. carrinho] → [128 neurônios] → →    [apertar ESQUERDA: 30%]
[vel. carrinho] → [128 neurônios]        [apertar DIREITA: 70%]
[ang. poste]    →
[vel. poste]    →
```

A rede gera **probabilidades** para cada ação. Então, fazemos uma amostragem:
> Jogue um dado → 1-30: aperte ESQUERDA, 31-100: aperte DIREITA

**Exemplo da vida real:** Um aplicativo de clima diz "70% de chance de chuva". Você não SABE que vai chover — você decide com base na probabilidade. O robô faz a mesma coisa!

---

## Normalização: Por que Subtraímos a Média

Antes de usar G_t para atualizar, nós normalizamos:
```
G_normalizado = (G - média(G)) / desvio_padrão(G)
```

**Por quê?** Imagine que todas as recompensas sejam positivas (como no CartPole — sempre +1 por passo). Sem normalização, TODA ação parece "boa" e o sinal de atualização fica confuso.

Após a normalização, alguns retornos tornam-se positivos (acima da média → aperte mais) e outros negativos (abaixo da média → aperte menos). O sinal torna-se muito mais limpo!

**Exemplo da vida real:** Seu professor avalia com base em uma curva. Se a média for 70 e você tirou 85, isso é ótimo! Mas se a média for 90 e você tirou 85, está abaixo da média. A pontuação bruta sozinha não conta a história toda.

---

## O Problema: Alta Variância

O REINFORCE tem uma grande fraqueza: **variância**. Os retornos G_t são muito ruidosos.

**Exemplo da vida real:** Imagine avaliar um chef provando apenas UMA refeição de cada restaurante. Às vezes o chef teve um dia ruim, às vezes os ingredientes não estavam bons. Uma refeição não é suficiente para saber com confiança se o restaurante é bom!

O REINFORCE espera por um episódio COMPLETO antes de atualizar. Um episódio pode ter muita sorte, outro muito azar. Os gradientes saltam para todos os lados.

É por isso que a curva de aprendizado (no gráfico) parece serrilhada:
- Algumas execuções chegam a 500 (incrível!)
- Algumas execuções caem para 50 (terrível!)

Apesar do ruído, o REINFORCE acaba aprendendo — mas exige paciência.

---

## Os Resultados

```
Episódio  100 | Média de recompensa (últimos 100):  43,1
Episódio  200 | Média de recompensa (últimos 100): 193,9
Episódio  500 | Média de recompensa (últimos 100): 408,4
Episódio 1000 | Média de recompensa (últimos 100): 500,0  ← Resolvido!
```

O robô aprende a equilibrar o poste pelos 500 passos máximos — RESOLVIDO!

Apesar dos seus problemas de variância, o REINFORCE no CartPole é eficaz porque:
1. Os episódios são curtos (assim, obtemos muitos por rodada de treinamento)
2. A política ideal é simples (basicamente apertar na direção em que o poste se inclina)

---

## Principais Conclusões

| Conceito | Em Português Simples |
|---------|---------------|
| **Política** | A receita do robô para escolher ações |
| **Episódio** | Uma partida completa do início ao fim |
| **Retorno G_t** | Recompensa futura total a partir deste momento |
| **Desconto γ** | Recompensas futuras valem um pouco menos que as imediatas |
| **Normalizar** | Subtrair a média para que o sinal fique mais limpo |
| **Variância** | O quanto as estimativas de gradiente saltam |

---

## O que vem a seguir?

A grande fraqueza do REINFORCE é a **variância**. No próximo script (`reinforce_baseline.py`), adicionamos um **baseline** que reduz drasticamente esse ruído — sem mudar o que o algoritmo aprende em média.

A ideia chave: em vez de perguntar "esta ação foi boa?", perguntamos "esta ação foi **melhor do que o esperado**?". Essa pequena mudança torna o aprendizado muito mais estável.
