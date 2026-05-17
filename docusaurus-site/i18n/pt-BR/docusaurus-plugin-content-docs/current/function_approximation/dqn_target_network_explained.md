# Target Network: Estabilizando o Alvo 🎯

## O Problema do Alvo Móvel

Imagine que você está tentando acertar o centro de um alvo com arco e flecha. Você atira, vê onde sua flecha caiu e ajusta sua mira para a próxima vez. Simples, certo?

Agora imagine que o alvo se MOVE toda vez que você atira! Cada flecha que você lança muda ligeiramente onde o alvo estará para o próximo disparo. Você nunca melhoraria — estaria perseguindo um alvo que está sempre fugindo.

Esse é exatamente o problema da DQN sem uma rede alvo!

---

## Por Que os Alvos Q Continuam se Movendo

Na DQN, o alvo para cada atualização é:
> alvo = recompensa + γ × max(Q(próximo_estado))

Aqui, **γ (gama)** é o **fator de desconto** — um número entre 0 e 1 (geralmente 0,99) que controla o quanto o agente se importa com recompensas *futuras* em comparação com recompensas *imediatas*.

**Exemplo da vida real:** Imagine que alguém lhe oferece um biscoito agora, ou dois biscoitos amanhã. Se você realmente quer biscoitos agora, seu γ é baixo (você desconta muito o futuro). Se você é paciente e está feliz em esperar, seu γ é alto (recompensas futuras importam quase tanto quanto as de agora). Em RL, γ = 0,99 significa "uma recompensa no próximo passo vale 99% de uma recompensa agora".

Os valores Q no lado direito vêm da... mesma rede que estamos treinando!

Portanto, toda vez que atualizamos a rede (para tornar os valores Q melhores), também mudamos os alvos. É um loop de feedback:

1. Atualizar a rede → valores Q mudam
2. Valores Q mudam → alvos mudam
3. Alvos mudam → atualizar a rede de forma diferente
4. Repetir para sempre — instável!

**Exemplo da vida real:** Tentar se pesar em uma balança que muda suas leituras toda vez que você sobe nela. Você nunca saberia seu peso real!

---

## A Solução: Congele o Alvo! ❄️

A **Rede Alvo (Target Network)** é uma CÓPIA da rede Q principal que fica congelada no lugar.

- **Rede online** (`qnet`): Atualizada a cada passo de treinamento — aprende rapidamente
- **Rede alvo** (`target_net`): Cópia congelada — atualizada apenas a cada 100 passos

Usamos o alvo CONGELADO para calcular os alvos:
> alvo = recompensa + γ × max(Q_TARGET(próximo_estado))

O alvo permanece parado por 100 passos! Isso dá à rede online um objetivo estável para mirar. Então, copiamos os pesos da rede online para a rede alvo, congelamos novamente e repetimos.

**Exemplo da vida real:** Pense em um aluno e um professor. O professor passa a lição de casa (o alvo). O aluno aprende e melhora. Após 100 aulas, o professor ATUALIZA a lição para ser mais difícil. O professor não muda a cada minuto — isso seria caótico demais!

---

## A Receita Completa da DQN 🍕

O algoritmo DQN completo (experience replay + target network) é:

```
1. Inicializar rede online Q e rede alvo Q_target (mesmos pesos)
2. Criar buffer de repetição (caixa de memória)

A cada passo no ambiente:
  a. Escolher ação usando ε-greedy com Q
  b. Armazenar (estado, ação, recompensa, próximo_estado) no buffer

A cada 4 passos:
  c. Amostrar mini-lote aleatório do buffer
  d. Calcular alvos usando Q_TARGET (congelado!)
  e. Atualizar Q para minimizar a perda (loss)

A cada 100 passos:
  f. Copiar pesos de Q → Q_TARGET (sincronizar alvo)
```

Este é o algoritmo exato do artigo original da DQN da DeepMind (2015)!

---

## O Que a Comparação Mostra

Ao executar `dqn_target_network.py`, você verá:

**Sem rede alvo (apenas DQN + replay):**
- O treinamento pode ser "aceitável", mas com colapsos periódicos
- Os valores Q podem divergir (explodir ou oscilar)
- O aprendizado é menos previsível

**DQN Completa (replay + rede alvo):**
- Aprendizado ascendente mais consistente
- Os valores Q permanecem em uma faixa razoável
- Convergência mais rápida para o limite de solução (195+ no CartPole)

---

## A "Tríade Mortal" ☠️

No aprendizado por reforço, combinar três coisas cria instabilidade:

1. **Aproximação de funções** (rede neural em vez de tabela) ← nós usamos isso
2. **Bootstrapping** (usar valores Q para estimar valores Q) ← nós usamos isso
3. **Aprendizado off-policy** (Q-learning usa o máximo, não a política real) ← nós usamos isso

As três juntas = a "tríade mortal". A DQN doma isso com:
- Experience replay → quebra as correlações
- Rede alvo → quebra o loop de feedback

Isso não resolve totalmente o problema, mas o torna gerenciável!

---

## Vocabulário Chave

| Palavra | Significado |
|------|---------|
| **Rede Alvo (Target Network)** | Uma cópia congelada da rede Q usada apenas para calcular alvos |
| **Rede Online** | A rede Q que está sendo ativamente treinada |
| **Sync (Sincronização)** | Copiar os pesos da rede online para a rede alvo |
| **Loop de Feedback** | Quando a saída de um sistema realimenta e altera a entrada (pode causar instabilidade) |
| **Tríade Mortal** | A combinação de aproximação de funções + bootstrapping + off-policy que causa instabilidade |

---

## Impacto no Mundo Real

Em 2015, a DeepMind publicou seu artigo sobre a DQN mostrando uma IA que conseguia jogar 49 jogos de Atari em nível sobre-humano — usando APENAS esses dois truques (replay + rede alvo).

Antes disso, as pessoas pensavam que não era possível treinar redes neurais com RL devido à instabilidade. A DeepMind provou que estavam erradas e deu início à revolução do deep RL!

A seguir, aplicaremos esta receita completa da DQN ao Atari Pong — um videogame real com pixels brutos como entrada!
