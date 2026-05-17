# Sensibilidade de Hiperparâmetros do PPO: O que é Mais Importante?

## Por que os Hiperparâmetros Importam

Imagine fazer um bolo de chocolate. A receita pede:
- 2 ovos
- 200g de farinha
- 1 colher de chá de fermento em pó
- 35 minutos a 180°C

Se você usar 10 ovos, o bolo explode. Se usar 0,1 colher de chá de fermento, ele não cresce.
Se assar a 300°C por 10 minutos, ele queima por fora e fica cru por dentro.

**Os hiperparâmetros no PPO são como os ingredientes e as configurações do forno.** A combinação certa funciona perfeitamente; configurações erradas podem impedir completamente o aprendizado.

Este script testa sistematicamente 3 hiperparâmetros principais, alterando apenas UM por vez, executando cada configuração com 3 sementes aleatórias (seeds) diferentes e comparando os resultados.

---

## Os Três Experimentos

### Experimento 1: Clip Epsilon (ε)

```
ε = 0,05   (muito conservador — apenas mudanças minúsculas na política são permitidas)
ε = 0,2    (padrão — equilíbrio entre segurança e velocidade)
ε = 0,4    (agressivo — permite grandes mudanças na política)
```

**O que o ε controla?**

ε é o tamanho da "janela de segurança" ao redor da política antiga:
```
a razão deve permanecer em [1 - ε,  1 + ε]
ε=0,05: razão em [0,95, 1,05]  ← mudanças minúsculas
ε=0,2:  razão em [0,80, 1,20]  ← padrão  
ε=0,4:  razão em [0,60, 1,40]  ← mudanças grandes
```

**Exemplo da vida real:** Pense no ε como "o quanto você tem permissão para girar o volante do carro em um único movimento".
- ε=0,05: Como dirigir no gelo — apenas ajustes minúsculos
- ε=0,2:  Direção normal — curvas razoáveis
- ε=0,4:  Piloto de corrida — direção agressiva, risco de **rodar** (perder o controle porque a mudança é muito drástica, como um carro derrapando para fora da estrada)

**Resultados esperados:**
- ε=0,05: Aprendizado lento, mas estável (muito cauteloso)
- ε=0,2:  Bom equilíbrio (o valor **"Cachinhos Dourados"** — nem muito pequeno, nem muito grande, na medida certa — nomeado após o conto de fadas onde a Cachinhos Dourados escolhe o mingau que não está nem muito quente nem muito frio)
- ε=0,4:  Pode aprender rápido, mas pode **ultrapassar o alvo e oscilar** (ultrapassar = passar da política ideal; oscilar = saltar para frente e para trás ao redor dela sem se estabilizar, como um pêndulo que balança demais em ambas as direções)

---

### Experimento 2: Taxa de Aprendizado (Learning Rate)

```
lr = 1e-4  (lento, mas estável)
lr = 3e-4  (padrão)
lr = 1e-3  (rápido, mas arriscado)
```

**O que a taxa de aprendizado controla?**

A taxa de aprendizado é como o "tamanho do passo" ao subir uma colina (cada passo = uma atualização nos pesos da rede neural, movendo-a ligeiramente na direção que melhora a recompensa):
- Muito pequena: Demora uma eternidade para chegar ao topo (converge lentamente)
- Muito grande: Você ultrapassa o pico e cai do outro lado (**diverge** — a recompensa de treinamento colapsa ou flutua descontroladamente em vez de melhorar de forma estável)
- Na medida certa: Progresso constante em direção ao cume

**Exemplo da vida real:** Afinar uma corda de violão.
- lr=1e-4: Giros minúsculos na **tarraxa** (o botão que você gira para apertar ou soltar uma corda) — demora uma eternidade, mas é preciso
- lr=3e-4: Afinação normal — encontra o tom certo em alguns giros
- lr=1e-3: Grandes **puxões** na tarraxa — pode **arrebentar** a corda (quebrá-la completamente, assim como atualizações excessivamente grandes podem quebrar o treinamento de forma irreversível)!

**Resultados esperados:**
- lr=1e-4: Eventualmente bom, mas muito lento
- lr=3e-4: Melhor desempenho geral
- lr=1e-3: Progresso inicial rápido, depois instabilidade

---

### Experimento 3: Épocas de Atualização (K)

```
K = 3   (conservador — poucas passagens por cada lote)
K = 10  (padrão)
K = 20  (agressivo — muitas passagens por cada lote)
```

**O que as épocas de atualização controlam?**

Após coletar um **rollout** (= jogar o jogo por um período de tempo para reunir nova experiência — como um aluno fazendo uma sessão de lição de casa antes de revisá-la), o PPO agrupa essa experiência em um **lote** (batch = o conjunto completo de tuplas de estado, ação e recompensa desse rollout). Em seguida, ele executa K **passagens** (passes = varreduras completas pelo lote, cada passagem atualizando a rede uma vez) sobre os mesmos dados.
Mais épocas = extrair mais aprendizado de cada lote, mas com o risco de **overfitting a dados obsoletos** (= memorizar padrões que eram verdadeiros sob a política antiga, mas que não são mais válidos uma vez que a política foi atualizada, como um aluno que memoriza o exame do ano passado e reprova em um novo).

**Exemplo da vida real:** Um aluno praticando com um conjunto de 20 problemas de matemática.
- K=3:  Faz cada problema 3 vezes → ainda está aprendendo, não vicia no conjunto de prática
- K=10: Faz cada problema 10 vezes → domínio sólido desses problemas específicos
- K=20: Faz cada problema 20 vezes → **memoriza as soluções sem realmente entender a matemática** (= o modelo se ajusta perfeitamente ao lote específico, mas perde a capacidade de generalizar)!

> ⚠️ **"Mas os resultados para K=20 parecem bons — por que eu deveria me importar?"**
> O truque de corte (clipping) do PPO limita o quanto a política pode mudar por passagem, então K=20 não causará um colapso repentino.
> No entanto, o agente ainda está se adaptando excessivamente a dados que não refletem mais o que a política atual realmente experimentaria.
> Isso **atrasa o aprendizado a longo prazo**: cada rollout ensina menos ao agente do que deveria, porque as passagens posteriores reciclam informações cada vez mais obsoletas.
> O dano é gradual, não dramático — que é exatamente por isso que é fácil de ignorar em experimentos curtos.

O corte evita o overfitting catastrófico, mas o excesso de épocas ainda pode retardar o aprendizado geral.

**Resultados esperados:**
- K=3:  Menos eficiente (algum potencial de aprendizado desperdiçado por lote)
- K=10: Bom equilíbrio
- K=20: Risco de a política tornar-se **muito confiante em dados obsoletos** (= as atualizações da rede são impulsionadas por experiências que não correspondem mais ao que a política atual encontraria, corroendo silenciosamente a eficiência da amostragem)

---

## Como Ler os Resultados

O gráfico mostra três painéis, cada um variando um hiperparâmetro:

```
Gráfico da esquerda:  Clip Epsilon — qual ε aprende mais rápido?
Gráfico do meio:      Taxa de Aprendizado — qual lr é mais estável?
Gráfico da direita:   Épocas de Atualização — qual K encontra a melhor política?
```

Cada linha é a **recompensa média ao longo de 3 sementes (seeds)** (para reduzir a aleatoriedade).

**O que procurar:**
1. **Velocidade de aprendizado:** Qual linha atinge a recompensa alta mais rápido?
2. **Desempenho final:** Qual linha alcança a maior recompensa final?
3. **Estabilidade:** Qual linha tem menos oscilação?

Um bom hiperparâmetro equilibra os três!

---

## Metodologia: Experimentação Científica

Este experimento usa o design de **estudo de ablação** (ablation study = um método onde você remove ou varia um componente de cada vez para medir seu impacto individual — nomeado após a prática científica de remover seletivamente tecidos para estudar sua função):
1. Escolha valores padrão: ε=0,2, lr=3e-4, K=10
2. Altere UM parâmetro por vez
3. Mantenha todo o resto fixo
4. Compare os resultados

Isso nos diz o efeito de CADA parâmetro isoladamente.

**Exemplo da vida real:** Testar se um novo fertilizante ajuda as plantas:
- Mude o fertilizante, mantenha todo o resto igual (mesmo solo, água, luz solar)
- Se as plantas crescerem melhor → o fertilizante ajudou!

---

## Descobertas Comuns na Prática

| Hiperparâmetro | Muito Pequeno | Ponto Ideal | Muito Grande |
|----------------|---------------|-------------|--------------|
| **ε (clip)** | Convergência lenta | ε ≈ 0,2 | Instabilidade |
| **lr** | Muito lento | 2,5e-4 a 3e-4 | Divergência |
| **K (épocas)** | **Desperdício de dados** (descarta rollout antes de extrair sinal total) | K = 4-10 | Overfitting a dados de rollout obsoletos |
| **n_steps** | Muito ruidoso | 128-2048 | **Erros de memória (OOM)** (usa muita RAM) |
| **batch_size** | Muito ruidoso | 32-256 | **Erros de memória (OOM)** (usa muita RAM) |

Esses "pontos ideais" podem mudar dependendo do ambiente!

---

## A Percepção Principal: O PPO é Relativamente Robusto

Comparado a algoritmos anteriores (como DQN sem redes de alvo), o PPO é relativamente robusto às escolhas de hiperparâmetros. O mecanismo de corte fornece uma rede de segurança natural.

**Exemplo da vida real:** Um carro com freios **ABS** (Sistema de Frenagem Antibloqueio — um recurso de segurança que evita que as rodas travem durante uma frenagem brusca, mantendo o motorista no controle) vs. sem ABS:
- Sem ABS (DQN): Uma curva errada (hiperparâmetro ruim) e você roda
- Com ABS (PPO): O carro se corrige — hiperparâmetros razoáveis funcionam todos razoavelmente bem

Essa robustez é uma das principais razões pelas quais o PPO é o algoritmo de RL mais popular na prática!

---

## Principais Conclusões

| Conceito | Em Português Simples |
|---------|---------------|
| **Estudo de ablação** | Mudar uma coisa de cada vez para ver seu efeito |
| **Clip epsilon ε** | Limite de segurança — 0,2 geralmente é o melhor |
| **Taxa de aprendizado** | **Tamanho do passo** — o quanto os pesos da rede são ajustados após cada lote (pense nisso como o tamanho de cada passo ao caminhar em direção a um objetivo). **2,5e-4 a 3e-4** é notação científica para 0,00025 a 0,0003 — estes são multiplicadores adimensionais, não valores de tempo |
| **Épocas de atualização K** | Quantas vezes reutilizar cada lote — 4-10 é o padrão |
| **Sementes Aleatórias (Seeds)** | Cada experimento é repetido com diferentes **sementes aleatórias** (= o número inicial fornecido ao gerador de números aleatórios, que controla todas as escolhas aleatórias no treinamento). Usar várias sementes revela se os resultados são consistentes ou se foram apenas sorte |

---

## Resumo: Métodos de Gradiente de Política em um Relance

```
REINFORCE              A2C                    PPO
     │                  │                      │
Episódios completos   Atualizações n-passos  N-passos + corte (clipping)
Simples, mas ruidoso  Rápido, mas instável   Estável + eficiente
Melhor para ambien-   Ambientes de difi-     Ambientes difíceis
tes fáceis            culdade média          (padrão da indústria)
```

**Se você aprender apenas UM algoritmo desta fase, aprenda o PPO.** Ele é a base de:
- Treinamento do ChatGPT da OpenAI (RLHF usa PPO)
- Seguimentos do AlphaGo da DeepMind
- Maior parte da pesquisa moderna em robótica
- IAs que jogam videogames
