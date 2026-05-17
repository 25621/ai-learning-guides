# Q-Learning Conservador (CQL) 🛡️

## O Que É?

Imagine que você está aprendendo a investir dinheiro lendo um livro de registros gigante de negociações de ações passadas feitas por outras pessoas. O livro tem compras, vendas e esperas — mas **nenhum registro de qualquer negociação que ninguém tenha feito de fato**.

Agora imagine que um estudante excessivamente confiante olha para o livro e diz:
*"E se alguém tivesse comprado bilhetes de loteria toda segunda-feira? Essa teria sido uma negociação incrível!"*

O problema: **o livro não tem dados sobre a compra de loteria às segundas-feiras**, então o estudante está apenas alucinando. No entanto, essa negociação alucinada parece ótima no papel, então a "política" do estudante continua querendo fazê-la.

Esse problema de alucinação é o **deslocamento de distribuição** (distribution shift): um aprendiz offline adora ações que o conjunto de dados nunca testou, porque não há dados para contradizer o otimismo. O CQL é a cura.

---

## Como o Q-Learning Dá Errado em Modo Offline

O alvo (target) do Q-learning normal é:

```
alvo(s, a) = r + γ · max_{a'} Q(s', a')
```

Aquele `max_{a'}` é o perigo. Quando o conjunto de dados nunca registrou a ação `a'` no estado `s'`, a rede simplesmente *chuta* um valor Q — e as redes neurais tendem a **superestimar** o Q para entradas não vistas. O alvo herda a superestimação, a rede aprende a prever esse número maior e, no próximo passo, **extrapolamos** (projetamos ainda mais além de qualquer coisa que os dados suportem) ainda mais alto. A política persegue um fantasma.

Se você pudesse continuar coletando mais dados, isso se autocorrigiria (a ação fantasma acabaria sendo ruim na realidade). Mas **no RL offline você não pode coletar mais dados.** O fantasma é para sempre.

---

## O Truque do CQL

O CQL (Kumar et al., 2020) adiciona um termo de penalidade à perda (loss):

```
perda_cql(s)  =  log Σ_a exp Q(s, a)   -   Q(s, a_dataset)
```

Duas partes:

1. **`log Σ_a exp Q(s, a)`** (leia-se: *"log-sum-exp sobre todas as ações"*) é um **máximo suave** (soft maximum) sobre todas as ações — uma aproximação suave e diferenciável de `max` que considera todas as ações de uma vez, em vez de selecionar rigidamente um vencedor. Penalizá-lo reduz os valores Q **de forma geral** (empurrando todas as previsões para baixo uniformemente) — especialmente para as ações com o Q *mais alto*, que é exatamente onde as alucinações vivem.
2. **`- Q(s, a_dataset)`** recompensa um Q alto na ação que o conjunto de dados realmente registrou — protegendo os valores Q "dentro da distribuição" da redução acima.

Efeito líquido: **O Q é puxado para baixo em ações não vistas e puxado para cima em ações vistas.** O Q aprendido torna-se um *limite inferior* (lower bound) do Q verdadeiro. A política **`argmax`** (a regra que simplesmente escolhe a ação com o Q mais alto) para de perseguir fantasmas.

Perda total:

```
L  =  Bellman_MSE   +   α · perda_cql
```

(Onde **`Bellman_MSE`** é o erro padrão do Q-learning normal, medindo o quanto o palpite atual da rede discorda de seu próprio palpite futuro).

`α` é o botão do conservadorismo. Muito pequeno → o deslocamento de distribuição volta a aparecer. Muito grande → o agente é tão conservador que nunca melhora além dos dados.

---

## Exemplos da Vida Real

- **Treinador de xadrez conservador.** Você só pode aprender com jogos que já foram jogados. Um treinador imprudente diz: "esta jogada hipotética sem precedentes poderia ser brilhante!". O CQL é o treinador que diz: "não temos dados sobre isso — vamos nos ater às jogadas que jogadores reais já testaram."
- **Escolhas de menu de restaurante.** As avaliações do Yelp nunca cobrem os itens fora do cardápio. Uma política ingênua recomendaria os itens fora do cardápio baseando-se em avaliações de cinco estrelas alucinadas. O CQL recomenda apenas o que foi pedido vezes suficientes para se confiar.
- **Garra robótica a partir de logs.** O robô tem vídeos de como segurar xícaras, garrafas e livros — mas nunca uma faca. O CQL se recusa a recomendar com confiança "segure a faca pela lâmina".

---

## O Que Nosso Código Faz

O script `cql.py`:

1. **Carrega os quatro conjuntos de dados** construídos por `d4rl_dataset.py`.
2. **Escolhe `medium-replay`** como o conjunto de treinamento — é o mais realista (qualidade mista) e o mais prejudicial para métodos ingênuos.
3. **Treina três agentes puramente offline**, em condições idênticas exceto pelo `α`:
   - `α = 0`   →  DQN offline ingênuo (sem penalidade — o benchmark quebrado)
   - `α = 1.0` →  CQL leve
   - `α = 5.0` →  CQL forte
4. **Avalia cada um a cada 2.500 passos de gradiente** executando-os de forma gananciosa no ambiente real (10 episódios). Este é o *único* contato com o ambiente; o treinamento em si nunca vê o ambiente.
5. **Plota as curvas de aprendizado** em `outputs/cql.png`.

---

## O Que Você Deve Ver

Uma execução típica imprime algo como:

```
Retornos de avaliação finais (média de 10 episódios, greedy):
  DQN offline ingênuo (alpha=0)       ->  ~30-150  (instável; muitas vezes falha)
  CQL (alpha=1.0)                     ->  ~300-450
  CQL (alpha=5.0)                     ->  ~450-500
```

No gráfico da curva de aprendizado:

- A **curva vermelha** (`α = 0`) sobe cedo e depois muitas vezes **cai num abismo** assim que as alucinações de deslocamento de distribuição infectam o **alvo de Bellman** (o número que usamos como "resposta correta" ao treinar a rede Q: `r + γ · max Q(s', ·)`). Quando valores Q fantasmas poluem esse alvo, cada passo de gradiente piora as coisas. A **perda de Bellman** (o MSE entre a previsão da rede Q e o alvo de Bellman) parece boa — essa é a **traição** do problema: a rede é perfeitamente consistente com suas próprias crenças erradas, então a perda não dá nenhum aviso.
- A **curva laranja** (`α = 1.0`) sobe mais devagar, mas **permanece alta**.
- A **curva verde** (`α = 5.0`) é a mais estável e geralmente a melhor.

O painel da perda de Bellman mostra outro indício: a perda do DQN ingênuo pode continuar pequena enquanto sua política é terrível, porque a rede é internamente consistente com suas próprias alucinações.

---

## Onde o CQL se Situa na Área

O CQL foi um marco porque deu uma solução fundamentada e simples para o deslocamento de distribuição. A linhagem:

```
DQN (online)
   │
   ▼
DQN offline ingênuo  ── quebra por causa do deslocamento de distribuição
   │
   ▼
CQL (Kumar 2020)     ── adiciona uma penalidade conservadora: Q é um limite inferior
   │
   ▼
IQL (Kostrikov 2021) ── para começar, nunca consulta o Q sobre ações não vistas
   │
   ▼
Decision Transformer (Chen 2021)  ── ignora o Q totalmente, trata o RL como modelagem de sequência
                                      (prevê a *próxima ação* dados os estados passados e
                                       um retorno total desejado, exatamente como um LLM
                                       prevê a próxima palavra)
```

Cada passo nesta linhagem é uma resposta diferente para a mesma pergunta: **como evito perguntar à minha rede Q sobre coisas que ela nunca viu?**

---

## Palavras-Chave para Lembrar

| Palavra | Significado |
|---------|-------------|
| **Deslocamento de distribuição** | A política treinada quer ações fora dos dados coletados |
| **Fora de distribuição (OOD)** | Um par (s, a) que o dataset nunca registrou |
| **Q Verdadeiro** | O retorno futuro real esperado ao tomar a ação `a` no estado `s`, se pudéssemos medir perfeitamente |
| **Q Conservador** | Uma função Q treinada que tenta ficar abaixo do Q verdadeiro em vez de prometer demais |
| **Logsumexp** | Uma aproximação suave e diferenciável de `max` |
| **Alpha (α)** | O botão de conservadorismo do CQL — quão forte empurrar o Q para baixo em ações OOD |

---

## Resumo de Uma Frase

> **O CQL adiciona uma "penalidade de pessimismo" que pune valores Q altos em ações que o conjunto de dados nunca testou — para que a política não se apaixone por alucinações.**
