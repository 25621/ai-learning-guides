# Conjuntos de Dados Benchmark D4RL 📦

## O Que É?

Imagine que você queira ensinar um robô a virar panquecas. Deixá-lo praticar em um fogão real por um mês seria lento, perigoso e caro. Mas você tem dez anos de vídeos gravados de chefs virando panquecas (alguns bons, outros ruins, alguns aleatórios). Você consegue ensinar o robô *apenas com esses dados*, sem nunca deixá-lo tocar em uma frigideira real?

Isso é **aprendizado por reforço offline** (offline reinforcement learning). O agente aprende a partir de um conjunto de dados fixo de experiências passadas — sem ambiente ao vivo. A parte mais difícil é que o agente nunca pode *testar* o que aprendeu até o final.

 Para tornar esse estudo justo, a comunidade de pesquisa precisava de um *conjunto de dados padrão*. Esse é o **D4RL** (**D**atasets for **D**eep **D**ata-**D**riven **R**einforcement **L**earning): uma coleção de transições pré-gravadas para tarefas de controle clássicas, lançada pela UC Berkeley em 2020. Cada artigo treina com os mesmos bytes, para que os resultados sejam comparáveis.

---

## O Que Há em um Conjunto de Dados D4RL?

Para cada tarefa, o D4RL fornece **quatro níveis de qualidade**:

| Nível | De onde vêm os dados | Por que isso importa |
|-------|----------------------|----------------------|
| **random**        | Uma política que escolhe ações uniformemente ao acaso | Pior caso: você ainda consegue aprender algo útil? |
| **medium**        | Uma política parcialmente treinada (cerca de metade da pontuação de um especialista) | Realista: a maioria dos dados registrados é medíocre |
| **expert**        | Uma política quase convergida | Melhor caso: você consegue igualar a política de origem? |
| **medium-replay** | O *buffer de replay completo* usado para treinar a política medium | Misto: contém falhas iniciais E sucessos posteriores |

A diferença entre `medium` e `medium-replay` é crucial:
- **`medium`** é gerado pegando uma única política "média" fixa e deixando-a jogar muitas partidas. Todos os dados refletem esse nível de habilidade constante e mediano.
- **`medium-replay`** é um log histórico. Ele contém todas as experiências coletadas *enquanto se aprendia* do zero até o nível médio. Ele mistura transições **boas e ruins** — exatamente como um log do mundo real (as primeiras tentativas desajeitadas de um robô *e* seu comportamento refinado posterior, tudo no mesmo balde).

---

## Exemplos da Vida Real de Conjuntos de Dados Offline

- **Registros médicos.** Anos de tuplas (estado_do_paciente, tratamento, resultado). Você não pode aleatorizar tratamentos em pessoas vivas, mas pode aprender uma política melhor a partir do histórico.
- **Logs de chat de atendimento ao cliente.** Milhões de registros (mensagem_do_usuario, resposta_do_atendente, satisfação). Treine um assistente melhor sem incomodar mais usuários.
- **Dados de frotas de direção autônoma.** Cada carro da Tesla / Waymo faz o upload de suas viagens. A frota é um conjunto de dados "medium-replay" gigante.
- **Sistemas de recomendação.** Logs de cliques do ano passado são um conjunto de dados congelado: você não pode mostrar novamente os mesmos anúncios para os mesmos usuários.

Em todos os quatro casos, **você não pode pedir uma nova amostra ao ambiente.** O conjunto de dados é o que você tem. Para sempre.

---

## O Que Nosso Código Faz

Os conjuntos de dados D4RL reais são registrados em tarefas de locomoção MuJoCo (Multi-Joint dynamics with Contact) (como HalfCheetah, Hopper, Walker2d, Ant — são simulações físicas 3D avançadas onde robôs virtuais aprendem a andar e correr). O MuJoCo é pesado para instalar, por isso recriamos a **mesma estrutura de quatro níveis no CartPole-v1** — o ambiente padrão para iniciantes das fases anteriores. As lições são transferidas diretamente.

O script `d4rl_dataset.py`:

1. **Treina um DQN** (Deep Q-Network, um algoritmo de RL padrão) no CartPole até que ele resolva a tarefa (retorno ≥ 475).
2. **Cria dois checkpoints** ao longo do caminho:
   - "medium" — o momento em que o retorno recente cruzou 150
   - "expert" — o momento em que o retorno recente cruzou 475
3. **Salva o buffer de replay completo da política média** — cada transição que ela já viu. Esse é o nosso conjunto de dados "medium-replay".
4. **Executa três novas políticas** por 10.000 transições cada:
   - `random`   — aleatório uniforme
   - `medium`   — o checkpoint medium + ruído ε=0.10
   - `expert`   — o checkpoint especialista + ruído ε=0.02
5. **Salva quatro arquivos `.npz`** (formato de array compactado do NumPy) em `outputs/`, cada um com arrays `obs / action / reward / next_obs / terminal`.

Esses quatro arquivos são as entradas para o `cql.py` e o `behavioral_cloning.py`.

---

## O Que Você Deve Ver ao Executar

Um resumo em texto simples impresso no console e salvo em `outputs/d4rl_summary.txt`:

```
dataset         |   N    |  mean return  |  min  |  max
------------------------------------------------------------
random          | 10000  |          ~22  |    ~9 |   ~80
medium          | 10000  |         ~180  |   ~50 |  ~500
expert          | 10000  |         ~490  |  ~400 |   500
medium-replay   | 10000  |          ~60  |    ~9 |  ~200
```

Ele também gera um histograma (`outputs/d4rl_returns.png`) mostrando como os quatro conjuntos de dados se sobrepõem. As principais características a serem observadas:

- **Random** se agrupa em torno de 20 (a duração média de um episódio aleatório de CartPole).
- **Expert** se agrupa no teto de 500.
- **Medium** fica no meio, com alta variância.
- **Medium-replay** tem uma "cauda" longa à direita — consiste principalmente em execuções que falharam cedo (retornos baixos), mas tem uma cauda que se estende a retornos mais altos conforme o agente aprendia.

---

## Por Que o Conjunto de Dados Importa

Seja qual for o conjunto de dados em que você treine seu algoritmo offline, você está colocando um *teto* no que é possível:

- **A partir do `expert`** — até mesmo um algoritmo simples como o BC (Behavioral Cloning, que apenas copia os dados exatamente) pode se sair bem, porque todos os dados são bons.
- **A partir do `random`** — você precisa de um algoritmo inteligente que consiga *costurar* (stitch together) as raras transições boas (encontrando um caminho para o sucesso combinando sequências curtas de boas ações de diferentes tentativas). O BC falhará completamente.
- **A partir do `medium-replay`** — o mais realista e o mais interessante. Bons algoritmos (como o **CQL** — Conservative Q-Learning, que evita ser excessivamente confiante em ações que nunca viu) podem às vezes **superar a qualidade média dos dados**, pois extraem estrutura de sinais mistos. Algoritmos simples (BC) regridem para a média.

Veremos exatamente essa história nos próximos dois scripts.

---

## Palavras-Chave para Lembrar

| Palavra | Significado |
|---------|-------------|
| **Offline RL**         | Treinar a partir de um conjunto de dados fixo; nenhuma interação com o ambiente é permitida |
| **Política de comportamento** | A política que *gerou* o conjunto de dados |
| **Qualidade do dataset** | Quão boa era a política de comportamento (random / medium / expert) |
| **Buffer de replay**   | O histórico completo de transições vistas durante uma execução de treinamento |
| **Deslocamento de distribuição** | A lacuna entre as ações no dataset e as ações que sua política treinada deseja tomar. Como o dataset nunca mostra o que acontece quando a nova política tenta algo que não foi registrado, as estimativas de valor do algoritmo para essas novas ações podem estar perigosamente erradas. |

---

## Resumo de Uma Frase

> **O D4RL congela o RL em um benchmark de estilo de aprendizagem supervisionada: os mesmos bytes para todos, sem trapaças no ambiente, que vença o melhor algoritmo.**
