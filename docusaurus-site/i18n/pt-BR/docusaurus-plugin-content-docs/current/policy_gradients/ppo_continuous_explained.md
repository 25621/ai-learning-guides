# PPO para Controle Contínuo: Fazendo o BipedalWalker Caminhar

## Ações Discretas vs. Contínuas

Até agora, todos os ambientes que resolvemos tinham ações **discretas**:
- CartPole: empurrar para a ESQUERDA ou para a DIREITA (2 escolhas)
- LunarLander: não acionar nada / esquerda / principal / direita (4 escolhas)

Mas os robôs do mundo real precisam de ações **contínuas**:
- Um robô humanoide: "com que força empurrar cada articulação" (qualquer valor de -1 a +1)
- Um carro: "exatamente quanto girar o volante" (qualquer ângulo de -30° a +30°)
- Um braço: "aplicar exatamente 2,3 Newtons nesta direção"

**Exemplo da vida real:** Digitar em um teclado = discreto (pressionar A, B, C...).
Escrever com um lápis = contínuo (mover a mão 2,3 cm para a direita, pressionar com 40g de força...).

---

## A Política Gaussiana para Ações Contínuas

Para ações contínuas, em vez de uma distribuição Categórica (escolher entre N categorias),
usamos uma **distribuição Normal (Gaussiana)**:

```
Ação ~ Normal(μ, σ)
```

Onde:
- **μ (mu, média)**: O centro da distribuição — o valor da ação que a rede "almeja"
- **σ (sigma, desvio padrão)**: A dispersão — quanta aleatoriedade / exploração adicionar

```
        Probabilidade
             │
        0,4 ─┤      ██████
             │    ████████████
        0,2 ─┤  ██████████████████
             │████████████████████████
             └──────────────────────── Valor da ação
           -1  -0,5   0   0,5   1
                      ↑
                   média μ
```

**Exemplo da vida real:** Um arqueiro habilidoso mira no centro do alvo (μ).
Suas flechas não atingem todas exatamente o mesmo ponto — há alguma dispersão (σ).
À medida que praticam, tornam-se mais precisos (σ diminui), mantendo-se centrados na mosca.

---

## Nossa Rede Actor-Critic Gaussiana

```
Estado (24 números) → [256 neurônios] → [256 neurônios] →
    ├── Actor: 4 valores de média (μ₁, μ₂, μ₃, μ₄)
    │          + 4 parâmetros log_std (compartilhados por todos os estados!)
    └── Critic: 1 valor (V(s))
```

O `log_std` (logaritmo do **desvio padrão** — uma medida de dispersão ou incerteza)
é um **parâmetro treinável** — não depende do estado!
Isso mantém o modelo simples, permitindo que a exploração mude durante o treinamento.

**Por que log_std em vez de std?** O desvio padrão deve ser positivo. O uso de `log_std` permite
que a rede produza qualquer número real (positivo ou negativo); então aplicamos
`exp(log_std)` — a função exponencial, que é a inversa do logaritmo — para
recuperar um std garantido como positivo. Isso evita que o std seja negativo ou zero.

---

## Computando a Log-Probabilidade para Ações Contínuas

Para ações discretas: `log_prob = log(P(ação=ESQUERDA))`

Para ações contínuas, a **distribuição Normal** descreve uma curva suave em forma de sino
ao redor da média. Um único valor exato tem probabilidade zero na matemática contínua, então usamos
a altura da curva naquele valor, chamada de **pdf** (função densidade de probabilidade):
```
log_prob = Σᵢ log[Normal(μᵢ, σᵢ).pdf(aᵢ)]
```

`log` significa logaritmo natural. Ele transforma valores de densidade minúsculos em números estáveis que são
mais fáceis de otimizar para redes neurais. Somamos todas as dimensões da ação (4 para o
BipedalWalker), porque a ação completa é um vetor de 4 números.

**Exemplo da vida real:** Qual é a probabilidade de fazer exatamente 5,732...°C amanhã?
Para o clima contínuo, você olharia para a curva da distribuição Normal e veria quão alta ela é
naquele ponto exato. Temperaturas mais prováveis (perto da média) têm probabilidade maior.

---

## BipedalWalker: Um Desafio de Caminhada

O BipedalWalker-v3 é um robô 2D que deve aprender a caminhar sem cair:

```
          O (cabeça)
         /│\
        / │ \
       /  │  \
      E   │   D   ← duas pernas, cada uma com uma articulação de joelho
     / \  │  / \
    ●   ● │ ●   ●  ← 4 motores (quadril/joelho para cada perna)
```

**Espaço de estados (24 números):**
- Chassi: ângulo, velocidade angular, velocidade horizontal, velocidade vertical (4 números)
- Articulações: 4 motores (2 quadris, 2 joelhos), cada um fornecendo ângulo e velocidade, além de 2 sensores de contato com o solo (um para cada perna) (10 números)
- 10 sensores de distância LIDAR (leituras de distância que veem o solo à frente) (10 números)

**Espaço de ações (4 valores contínuos, cada um em [-1, 1]):**
Os valores de ação controlam o **torque** (a força rotacional aplicada pelos motores) para exatamente 4 articulações (nenhuma ação é aplicada diretamente ao chassi):
- Torque do Quadril da Perna 1, Torque do Joelho da Perna 1, Torque do Quadril da Perna 2, Torque do Joelho da Perna 2

**Recompensas:**
- +300 por atingir o objetivo (lado direito)
- -100 por cair (tocar o chão com o corpo)
- Pequena recompensa por cada passo de progresso à frente
- Pequena penalidade para cada uso do motor (recompensa a eficiência)

**Resolvido quando:** Recompensa média > 300 ao longo de 100 episódios

---

## Diferença Principal em Relação ao PPO Discreto

Tudo é igual, EXCETO:

| | PPO Discreto | PPO Contínuo |
|---|---|---|
| **Política** | Categórica(logits) | Normal(μ, σ) |
| **Amostragem** | ação = amostra de {0,1,...,N} | ação = μ + σ × ruído |
| **log_prob** | log P(ação=k) | Σ log Normal(μᵢ, σᵢ).pdf(aᵢ) |
| **Clamp** | Não necessário | Restringir ações a [-1, 1] |

**Logits** são pontuações brutas, não normalizadas, para ações discretas. Uma política categórica as
converte em probabilidades com o **softmax** — uma função que pega qualquer conjunto de números e os
achata em uma distribuição de probabilidade válida (todos os valores positivos, somando 1).
Por exemplo, logits [2,0, 1,0, 0,5] tornam-se probabilidades [0,59, 0,24, 0,17]. O PPO contínuo **não** usa softmax para a ação em si,
porque a ação não é escolhida de um menu fixo. Em vez disso, a política produz a média
e o desvio padrão de uma distribuição Normal e, em seguida, amostra torques de valor real a partir dela.

**Clamp** significa forçar um valor dentro de um intervalo válido. O código usa `action.clamp(-1, 1)` para que o
ambiente nunca receba um comando de motor fora de seus limites permitidos.

**Clip** no PPO significa algo diferente: o PPO corta (clip) a razão de probabilidade dentro da perda (loss),
conforme explicado na [seção de corte do PPO](./ppo_scratch_explained.md#the-clipping-trick).
O "clamping" da ação protege a interface do ambiente; o "clipping" do PPO protege a atualização da política.

---

## Caminhando do Zero: O que o Agente Aprende

**Início do treinamento (recompensas negativas):** O robô se debate aleatoriamente e cai imediatamente.
Cada episódio termina em uma queda em segundos.

**Meio do treinamento:** O robô descobre que mover as pernas alternadamente cria progresso para frente.
Ele começa a dar passos pequenos e desajeitados — a recompensa torna-se menos negativa.

**Fim do treinamento:** Surge uma **marcha** (gait) suave e eficiente. Uma marcha é um padrão de movimento
repetido, como alternar passos esquerdos e direitos. O robô ajusta-se dinamicamente a terrenos irregulares, utilizando seus sensores LIDAR para adaptar seus passos em tempo real.

**Exemplo da vida real:** Um bebê aprendendo a caminhar:
1. Cai imediatamente (recompensa negativa)
2. Dá um passo, cai (ligeiramente menos negativa)
3. Dá alguns passos (pequena recompensa positiva)
4. Caminha pela sala (grande recompensa positiva!)

---

## Por que o BipedalWalker precisa de PPO (e não de REINFORCE)

- **Os episódios do BipedalWalker** podem ter até 1600 passos (muito mais longos que o CartPole!)
- **As recompensas são esparsas** — as recompensas de progresso para frente são minúsculas por passo
- **O REINFORCE precisaria** de milhares de episódios completos para obter um sinal útil

As atualizações de n-passos do PPO com [GAE (Generalized Advantage Estimation)](./ppo_scratch_explained.md#gae-smarter-advantage-estimates) permitem que o robô aprenda com episódios incompletos:
> "Mesmo que eu tenha caído após 50 passos, esses passos mostraram ALGUM progresso para frente.
> Deixe-me usar uma estimativa de retorno de 50 passos em vez de esperar pela conclusão do episódio."

---

## Resultados

Após 500 atualizações (≈ 1 milhão de passos no ambiente):
- O robô faz progresso visível, passando de movimentos aleatórios para algum movimento para frente
- Melhoria consistente na curva de aprendizado
- A convergência total para recompensa > 300 requer mais treinamento (5-10 milhões de passos)

A curva de aprendizado mostra a característica "curva em S" do controle contínuo:
1. Progresso inicial lento (estabilidade do aprendizado)
2. Melhoria rápida (descoberta da marcha)
3. Refinamento gradual (otimização da marcha)

---

## Principais Conclusões

| Conceito | Em Português Simples |
|---------|---------------|
| **Política Gaussiana** | Em vez de escolher de um menu, lança um dardo em um intervalo de valores |
| **μ (média)** | Onde a política "mira" |
| **σ (std)** | Quanta aleatoriedade / exploração a política usa |
| **log_std como parâmetro treinável** | Uma taxa de exploração global atualizada por otimização baseada em gradiente (gradiente *ascendente* na recompensa, ou equivalentemente gradiente *descendente* na perda do PPO) — como qualquer outro peso da rede |
| **Controle contínuo** | Controlar saídas de valores reais (torques, forças, ângulos) |

---

## O que vem a seguir?

O PPO tem muitos **hiperparâmetros** — configurações que você escolhe antes de o treinamento começar (em oposição aos
*parâmetros*, como os pesos da rede, que são aprendidos automaticamente). Exemplos incluem
`clip_eps`, taxa de aprendizado, número de épocas e tamanho do lote (batch size).

Quão sensível é o PPO a essas escolhas? O arquivo `ppo_hyperparams.py` executa experimentos
variando sistematicamente cada hiperparâmetro e mostra o efeito na velocidade e estabilidade do aprendizado.
