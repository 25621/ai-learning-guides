# Clonagem de Comportamento (BC) 🐒

## O Que É?

Imagine que você queira aprender a jogar tênis. Você assiste a centenas de horas de partidas gravadas de Wimbledon e simplesmente **copia o que os jogadores fazem**. Você não pensa se o golpe deles foi o *melhor* golpe — você apenas ajusta a posição do seu corpo à deles e balança a raquete da mesma forma.

Isso é clonagem de comportamento (behavioral cloning). **Sem recompensa. Sem planejamento. Apenas imitação.**

Em termos de RL: pegue o conjunto de dados de pares `(estado, ação)` e treine uma rede neural para prever a ação a partir do estado, exatamente como um modelo de classificação de imagens prevê gato vs. cachorro. O "rótulo" (label) é qualquer ação que o coletor de dados tenha tomado.

---

## Como Diferencia do RL Offline "Real"

| Abordagem | Usa recompensas? | Pode superar os dados? |
|-----------|------------------|------------------------|
| **BC**    | ❌ não           | ❌ não — na melhor das hipóteses, iguala a qualidade média dos dados |
| **CQL** (e outros) | ✅ sim | ✅ sim — pode extrair boas transições de dados mistos |

BC é a "visão de aprendizagem supervisionada" do RL. É incrivelmente simples, muitas vezes surpreendentemente forte, e o benchmark universal. **Se um algoritmo de RL offline não consegue superar o BC no mesmo conjunto de dados, ele não serviu para nada.**

---

## Exemplos da Vida Real

- **Aprender a dirigir a partir de imagens de câmeras de painel (dashcam).** Olhe para a estrada, preveja o ângulo do volante que o humano usou. Dois exemplos marcantes:
  - **ALVINN (1989)** — o primeiríssimo motorista de rede neural; uma pequena rede de 3 camadas treinada com entradas de câmera + laser para dirigir uma van em rodovias.
  - **NVIDIA PilotNet (2016)** — uma CNN profunda moderna treinada de ponta a ponta com imagens de dashcam; aprendeu a manter a faixa e manobras básicas de direção puramente imitando motoristas humanos, sem regras projetadas manualmente.
- **Aprendiz copiando um mestre cuca.** "O que quer que o chef faça, eu faço." Funciona muito bem se o chef for ótimo; produz um chef ruim se o chef for ruim.
- **GitHub Copilot.** O preenchimento automático é treinado para prever "qual código um humano digitaria a seguir?" — pura imitação de logs de código-fonte.
- **Imitar seu irmão mais velho.** Crianças fazem isso por anos antes de começarem a raciocinar sobre *por que* o irmão mais velho faz o que faz.

---

## A Matemática (Uma Linha)

Para cada `(s, a)` no conjunto de dados, minimize:

```
perda = -log π(a | s)        (entropia cruzada)
```

Isso é tudo. A política `π` é apenas um MLP que gera logits de ação; o treinamento é idêntico ao MNIST. Vamos detalhar os termos técnicos:
- **`π` (Pi):** O símbolo padrão para "política" — a regra ou rede neural que decide o que fazer.
- **MLP (Multi-Layer Perceptron):** Uma rede neural básica e padrão.
- **Logits:** Os scores brutos e não normalizados que a rede gera antes de os transformarmos em probabilidades.
- **Entropia Cruzada (Cross-entropy):** A fórmula padrão para penalizar um modelo quando ele atribui uma probabilidade baixa à resposta correta.
- **MNIST:** O famoso conjunto de dados para iniciantes de dígitos escritos à mão.

Treinar um agente para jogar um jogo via BC é literalmente idêntico a treinar uma rede para reconhecer dígitos escritos à mão no MNIST. No MNIST, a entrada é uma imagem e a saída é um dígito (0-9). No BC, a entrada é o estado do jogo e a saída é a ação (ex: "mover para a esquerda").

---

## O Que Nosso Código Faz

O script `behavioral_cloning.py`:

1. **Carrega todos os quatro conjuntos de dados** construídos por `d4rl_dataset.py` (`random`, `medium`, `expert`, `medium-replay`).
2. Para cada conjunto de dados, **treina uma política BC separada** por 10.000 passos de gradiente de entropia cruzada. A coluna de recompensa é completamente ignorada.
3. A cada 2.500 passos, **avalia** a política atual executando-a de forma gananciosa (greedy) no ambiente real CartPole-v1 (20 episódios, média).
4. Plota:
   - Um gráfico de barras: retorno final do BC por conjunto de dados.
   - Um gráfico de curva de aprendizado: como cada variante do BC evolui durante o treinamento.

---

## O Que Você Deve Ver

Uma execução típica imprime:

```
Retornos finais de avaliação:
  BC em random          ->    ~20  ± alguns      (≈ jogo aleatório)
  BC em medium          ->   ~150  ± grande      (≈ a política medium)
  BC em expert          ->   ~480  ± pequeno     (≈ a política expert)
  BC em medium-replay   ->    ~60  ± grande      (≈ a MÉDIA dos dados mistos)
```

O gráfico de barras torna a história óbvia: **o retorno do BC acompanha o retorno médio do conjunto de dados.** Ele não pode ultrapassar esse teto porque não tem como preferir as partes "boas" de um conjunto de dados misto em detrimento das partes "ruins" — ambos são alvos de imitação igualmente válidos.

Essa é a conclusão: **o BC herda o teto dos dados.**

---

## BC vs CQL — A Comparação Mais Clara

No conjunto de dados **medium-replay** (o caso mais realista de qualidade mista):

| Método | Retorno final aprox. | Por quê? |
|--------|----------------------|----------|
| BC     | ~60   | Imita a *média* das primeiras execuções ruins + as boas posteriores |
| CQL    | ~400+ | Usa recompensas para preferir transições de alto Q; costura uma boa política a partir de dados mistos |

Portanto, o CQL **supera os dados**, enquanto o BC **iguala os dados**. Essa é toda a razão pela qual o RL offline é um campo de pesquisa e não apenas "fazer aprendizagem por imitação". Quando os dados têm qualidade mista (o que os logs reais sempre têm), métodos sensíveis à recompensa recuperam mais.

Em dados **expert** (especialistas), a comparação se inverte: o BC iguala o especialista (~480). Você pode se perguntar por que o CQL "empata" aqui em vez de perder. Como o CQL é projetado para ser *conservador* e penalizar ações não vistas no conjunto de dados, ele acaba fazendo exatamente o que o especialista fez. Ele não pode superar o especialista (porque a pontuação máxima já foi atingida), mas também não quebra a estratégia do especialista. Ele apenas empata com o desempenho do BC.

Este é o famoso equilíbrio entre "qualidade dos dados vs. algoritmo":

```
                            dados EXPERT  →  BC ganha, CQL empata
   Sofisticação do algoritmo ↑         
                            dados MISTOS  →  CQL supera claramente o BC
                            
                            dados RANDOM  →  Todos falham; precisa de exploração
```

---

## Onde o BC Vive no RL Moderno

- **Pré-treinamento para RL online.** Muitos sistemas modernos (RT-1, Voyager, bots de jogos) começam com BC em demonstrações e depois fazem o ajuste fino (fine-tuning) online com PPO/SAC.
- **RLHF.** O Passo 1 do InstructGPT é o ajuste fino supervisionado — puro BC sobre respostas escritas por humanos. PPO + modelo de recompensa vêm depois.
- **DAgger (Ross et al., 2011).** Uma extensão inteligente para corrigir o problema do **erro composto** (compounding error).
  *Por que o erro composto é um problema se o BC clona perfeitamente?* Mesmo que um modelo BC seja 99% preciso, aquele 1% de erro eventualmente acontece. Quando isso ocorre, o agente entra em um estado que nunca viu no conjunto de dados perfeitamente dirigido. Como ele está confuso, comete um erro maior, afastando-se ainda mais dos dados conhecidos, acumulando-se em um fracasso total (como cair de um penhasco).
  *A solução:* Poderíamos apenas pedir ao especialista para dirigir para sempre, mas o tempo do especialista é caro. Em vez disso, o DAgger deixa a política BC dirigir. Quando a política comete um erro e deriva para um estado estranho, pausamos, perguntamos ao especialista "o que você faria *exatamente aqui*?", e adicionamos isso ao conjunto de dados. Nós apenas "consultamos o especialista novamente nos estados que a política BC visita" porque só precisamos que o especialista nos ensine a recuperar de nossos próprios erros específicos.
- **Decision Transformer (Chen et al., 2021).** Um BC "inteligente" que condiciona a previsão da ação a um *retorno desejado* (return-to-go), transformando essencialmente o RL offline de volta em uma previsão de próximo token.

---

## Palavras-Chave para Lembrar

| Palavra | Significado |
|---------|-------------|
| **Aprendizagem por imitação** | Termo guarda-chuva para "copiar o demonstrador"; BC é o membro mais simples |
| **Erro composto** | Um pequeno erro de BC leva a estados que o dataset nunca viu, onde os erros se acumulam |
| **Dados de demonstração** | Trajetórias produzidas por um especialista, usadas como conjunto de treinamento do BC |
| **Teto dos dados** | O retorno do BC é limitado pelo retorno médio no conjunto de dados |
| **DAgger** | Uma correção interativa para o erro composto |

---

## Resumo de Uma Frase

> **A clonagem de comportamento é apenas aprendizagem supervisionada em pares (estado, ação) — forte quando os dados são bons, impotente quando os dados são mistos.**
