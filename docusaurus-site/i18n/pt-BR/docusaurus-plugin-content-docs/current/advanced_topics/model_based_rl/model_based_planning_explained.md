# Uso de um Modelo Aprendido para Planejamento (MPC) 🔮

## A Grande Ideia

Você tem um **modelo de mundo** (uma rede neural que prevê o futuro). E agora?

O uso mais direto é o **planejamento**: a cada momento, pergunte ao modelo "o que aconteceria se eu tentasse *este* plano? *aquele* plano? *aquele outro* plano?". Então, execute o plano que parecer melhor — mas **apenas o primeiro passo dele**. Como o modelo não é perfeito, executamos apenas uma ação, observamos o novo estado real do ambiente real e, em seguida, replanejamos do zero.

Esse truque tem um nome: **Controle Preditivo por Modelo** (Model Predictive Control - MPC).

---

## Uma Analogia da Vida Real

Você está em um restaurante olhando o cardápio. Você não se compromete com um pedido de cinco pratos na hora — você pede o primeiro prato, vê o quão satisfeito está e então decide a sobremesa.

Ou: você está dirigindo em uma estrada sinuosa. Você não trava as entradas de direção para os próximos 30 segundos — você olha constantemente para frente, planeja alguns segundos à frente, toma a próxima ação de direção e planeja novamente.

Esse loop de **planejar longe / agir perto / replanejar** é o MPC.

---

## Como Funciona o "Random Shooting"

Existem planejadores mais sofisticados — por exemplo:
- **CEM** (Cross-Entropy Method): refina iterativamente uma distribuição sobre planos, mantendo apenas os melhores de cada rodada.
- **MCTS** (Monte Carlo Tree Search): constrói uma árvore de busca guiada por estatísticas de simulação, usada pelo AlphaGo e MuZero.
- **Planejadores baseados em gradiente**: diferenciam as previsões do modelo em relação às ações e seguem o gradiente diretamente.

Nós usamos o mais simples que funciona: **random shooting** (disparo aleatório).

```
Dado o estado atual s:
    1. Amostrar N=200 sequências de ações aleatórias de comprimento H=15.
    2. Para cada sequência, simulá-la através do modelo de mundo a partir de s,
       somando uma recompensa moldada (shaped) a cada passo. (200 sonhos em paralelo!)
    3. Encontrar a sequência com a maior recompensa total prevista.
    4. Executar a PRIMEIRA ação dessa sequência no ambiente real.
    5. Observar o próximo estado real. Descartar o resto do plano.
    6. Voltar ao passo 1 — replanejar do zero.
```

200 planos × 15 passos = 3.000 transições imaginadas por passo real. O modelo de mundo executa todas em uma única passagem de rede neural em lote — geralmente em poucos milissegundos.

---

## Por Que Replanejar a Cada Passo?

Porque o modelo é imperfeito. Os erros se acumulam ao longo de uma projeção (rollout). O plano no passo 0 é confiável apenas para os primeiros movimentos; no passo 15, o modelo está alucinando. Por isso, confiamos apenas no **primeiro movimento** e, em seguida, atualizamos o plano com o estado real mais recente.

Essa é a mesma razão pela qual os humanos não escrevem um plano de xadrez de 100 jogadas e se prendem a ele — as circunstâncias mudam e, quanto mais longe você tenta adivinhar, menos isso corresponde à realidade.

---

## Um Detalhe: A Recompensa Precisa Dizer Algo ao Planejador

No CartPole, a recompensa real é `+1` a cada passo até que a haste caia. O modelo preverá fielmente `+1, +1, +1, ...` para quase todos os planos, porque planos aleatórios raramente terminam rápido dentro do modelo — e assim todos os planos pontuam o mesmo. O planejador não tem como escolher entre eles.

A solução: substituir a recompensa real por uma **proxy suave** durante o planejamento:

```python
recompensa_proxy(estado) = 1
                         - |angulo_da_haste| / 0.21          # haste vertical? (1=sim)
                         - 0.1 * |posicao_do_carrinho| / 2.4 # carrinho centrado? (1=sim)
```

Agora, planos que *terminariam* com a haste caída recebem pontuações visivelmente piores do que os planos que permanecem verticais. O planejador pode classificá-los.

> **Lição da vida real.** Um sinal de recompensa plano — "você sobreviveu mais um segundo" — é inútil para o planejamento de curto prazo. Sinais densos e moldados ajudam.

---

## O Que Nosso Código Faz

`model_based_planning.py`:

1. **Carrega** os pesos do modelo de mundo salvos por `world_model.py`. (Se estiverem faltando, ele treina um na hora.)
2. **Executa 20 episódios** de MPC no CartPole-v1 real.
3. **Também executa 20 episódios** com uma política uniformemente aleatória, como linha de base.
4. **Plota** ambos lado a lado e imprime as médias.

### O que você deve ver ao executá-lo

| Política | Recompensa média (passos sobrevividos) |
|----------|----------------------------------------:|
| Aleatória | ~22 (típico para CartPole — a haste cai rápido) |
| MPC (nosso) | ~150–500 (varia por semente; muitos episódios perto de 500) |
| Máximo possível | 500 |

Essa **melhoria de 5–25x** é alcançada sem rede de política, sem função de valor e sem treinamento adicional. Apenas um modelo de mundo + 200 sonhos por passo.

O gráfico `outputs/model_based_planning.png` mostra duas barras coloridas por episódio — MPC quase sempre mais alto que Aleatório, e muitos episódios atingindo o teto de 500 passos.

---

## Pontos Fortes do Planejamento Baseado em Modelo

- **Eficiência de amostragem.** Todo o aprendizado foi feito a partir de um lote de transições aleatórias. Nenhuma interação adicional com o ambiente foi necessária para derivar uma política útil.
- **Fácil de redirecionar.** Quer controlar o agente de forma diferente? Mude a proxy de recompensa — sem necessidade de retreinamento.
- **Interpretável.** Você pode inspecionar os planos que o agente considerou, as trajetórias previstas e as pontuações.

## Fraquezas (e O Que as Pessoas Fazem a Respeito)

- **Random shooting é básico.** Ele amostra planos às cegas. Para dimensões maiores, utiliza-se o **CEM** (Método de Entropia Cruzada), o **iLQR** ou planejadores baseados em gradiente.
- **Erro de modelo acumulado.** Horizontes longos divergem. As pessoas usam **ensembles probabilísticos** (vários modelos treinados com os mesmos dados) para que o planejador possa notar discordâncias e penalizar planos sobre os quais o modelo está incerto.
- **A recompensa real é o que queremos, no final das contas.** A moldagem de recompensas ajuda, mas para tarefas mais complexas aprende-se uma **função de valor** treinada *dentro* do modelo de mundo — um crítico que estima o retorno de longo prazo. Tanto o **Dreamer** quanto o **MuZero** usam essa ideia.

---

## Como Isso se Conecta a Sistemas Modernos

A receita exata que você acabou de executar — **dinâmica aprendida + planejamento** — é a base de alguns dos sistemas de RL mais fortes na pesquisa atual de IA:

- **MuZero** (DeepMind): combina um modelo de mundo aprendido com Busca em Árvore Monte Carlo. Dominou Go, xadrez e Atari sem precisar das regras antecipadamente.
- **Dreamer / DreamerV3** (Hafner et al.): treina uma política *dentro* de um modelo de mundo em **espaço latente**. Alcança desempenho de ponta em mais de 100 benchmarks.
- **PETS / PlaNet / TD-MPC**: são famílias de algoritmos que escalam exatamente essa ideia para tarefas complexas de controle contínuo, como robótica.

Você construiu — em algumas centenas de linhas — o menor membro dessa família.

---

## Palavras-Chave

| Termo | Linguagem Simples |
|-------|-------------------|
| **MPC** | Controle Preditivo por Modelo — planejar adiante, agir uma vez, replanejar |
| **Random shooting** | Pontuar muitos planos aleatórios, escolher o melhor |
| **Horizonte (H)** | Quantos passos o plano olha para frente |
| **N amostras** | Quantos planos candidatos consideramos por passo |
| **Horizonte Recessivo** | Replanejar a cada passo em vez de se comprometer com um plano |
| **Proxy de Recompensa / Shaping** | Uma recompensa substituta que dá ao planejador um sinal útil para otimizar |

---

## Resumo de Uma Frase

> **Depois que você tem um modelo de mundo, planejar é apenas "sonhar cem futuros, escolher o melhor primeiro passo e repetir".**

Esse é todo o segredo do RL baseado em modelo.
