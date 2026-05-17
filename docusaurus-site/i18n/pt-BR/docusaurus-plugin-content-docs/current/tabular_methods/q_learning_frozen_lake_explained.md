# Agente de Q-Learning para o Frozen Lake 🧊

## O que é isso?

Imagine um lago congelado com gelo escorregadio. Existe um **quadrado de Início** (Start) e um **quadrado de Objetivo** (Goal)
com alguns **Buracos** (Holes) no meio. Se você cair em um buraco, você começa de novo!

O gelo é escorregadio, então mesmo que você tente caminhar para a direita, você pode deslizar para cima ou para baixo.
Um **agente de Q-Learning** é um robô que aprende — tentando repetidamente — como ir do
Início ao Objetivo sem cair nos buracos!

---

## O que significa o "Q" no Q-Learning?

O **"Q"** vem de **"Quality"** (Qualidade) — especificamente, a *qualidade* de tomar uma
ação específica em uma situação específica.

Pense nisso como a avaliação de um restaurante: "Quão boa (qualidade) é a pizza NESTE
restaurante?" Q(s, a) pergunta: "Quão bom é tomar a ação **a** quando estou no estado **s**?"

Um valor Q alto significa: "Ótima escolha! Esta ação leva a muita recompensa."
Um valor Q baixo significa: "Má ideia! Esta ação geralmente leva a problemas."

**Exemplo da vida real:** Imagine que você é uma criança decidindo se deve comer doces antes do jantar.
Seu valor Q para "comer doces agora" pode ser alto no momento (tem um gosto ótimo!), mas baixo no geral
(sua mãe fica brava, você se sente mal depois). O Q-learning aprende a levar em conta essas
consequências futuras — não apenas a sensação imediata!

---

## A Grande Ideia: Uma Tabela Mágica de Pontuações

O Q-Learning constrói uma tabela grande chamada **Tabela Q** (Q-table). Cada linha é um quadrado no gelo,
e cada coluna é uma ação (esquerda, direita, para cima, para baixo). Os números dentro são **pontuações**:
"Quão bom é tomar esta ação a partir deste quadrado?"

Toda vez que o robô tenta um movimento:
1. Ele recebe um feedback (ele caiu? ele chegou ao objetivo?)
2. Ele atualiza a pontuação na tabela usando esta fórmula:

> **Nova Pontuação = Pontuação Antiga + Taxa de Aprendizado × (O que realmente aconteceu − O que eu esperava)**

O robô está basicamente perguntando: "Este movimento foi melhor ou pior do que eu pensava?"

**Exemplo da vida real:** Pense em um bebê aprendendo a andar. Toda vez que eles tentam dar um passo e caem,
eles aprendem que "aquele passo foi ruim". Toda vez que eles conseguem, eles lembram que "aquilo funcionou!". Depois
de muitas tentativas, eles descobrem como andar. O Q-learning faz a mesma coisa, mas com uma tabela!

---

## O que torna o Q-Learning Especial: Ele é Off-Policy!

Aqui está algo inteligente: quando o Q-Learning atualiza sua tabela, ele *sempre assume que fará o
movimento perfeito na próxima vez*, mesmo que durante o treinamento ele às vezes explore movimentos aleatórios.

Isso torna o Q-Learning **off-policy**: a estratégia que ele *aprende* (sempre escolher a melhor ação
conhecida) é separada da estratégia que ele *segue* durante o treinamento (às vezes escolher uma ação
aleatória para explorar). Concretamente, a atualização da Tabela Q usa o valor Q *máximo* do próximo
estado — o melhor teórico — mesmo quando o próximo movimento real do robô for aleatório.

Em termos simples: o robô pode vagar aleatoriamente para a esquerda para explorar, mas seu aprendizado ainda
calcula como se ele fosse tomar a *melhor* ação em seguida. Essa separação permite que o Q-Learning
convirja para a estratégia ideal, independentemente de quanto ele explore.

---

## O que nosso Código Encontrou

Treinamos por **50.000 episódios** no Frozen Lake 4×4 escorregadio:

| Métrica | Resultado |
|--------|--------|
| Taxa de sucesso da avaliação gananciosa (greedy) | **73,1%** |
| Meta do marco (>70%) | ✓ **PASSOU** |

O gelo é muito escorregadio, então mesmo a melhor política não consegue vencer 100% das vezes!

A Tabela Q aprendida mostra que o agente descobriu: desça e vá para a direita evitando os buracos.

---

## Exemplos da Vida Real

- **Carro autônomo**: Aprender quais faixas pegar em cruzamentos através de testes.
- **Sistemas de recomendação**: Aprender quais filmes sugerir com base no fato de os usuários terem gostado de
  sugestões anteriores.
- **IA de videogame**: Um personagem que aprende a navegar em um labirinto tentando vários caminhos.

---

## Palavras-Chave para Lembrar

- **Tabela Q (Q-table)**: A tabela de "quão boa é cada ação em cada estado"
- **Q(s, a)**: A pontuação para tomar a ação a no estado s
- **Recompensa (Reward)**: O que o agente recebe após tomar uma ação (+1 por atingir o objetivo, 0 caso contrário)
- **Off-policy**: Aprende a estratégia ideal mesmo explorando aleatoriamente
- **ε-greedy** (ε = epsilon): Na maioria das vezes faz a melhor ação conhecida; às vezes explora aleatoriamente
- **Factor de desconto γ** (γ = gamma): Quanto valem as recompensas futuras (como preferir dinheiro agora vs mais tarde)

A grande ideia: **O Q-Learning constrói uma "folha de dicas" para cada situação e continua melhorando-a
até que conheça o melhor movimento em todos os lugares.**
