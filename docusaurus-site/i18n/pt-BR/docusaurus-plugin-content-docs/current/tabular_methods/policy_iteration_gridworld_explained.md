# Iteração de Política para GridWorld 🗺️

## O que é isso?

Imagine que você está jogando um jogo de tabuleiro em uma **grade 4×4** (como um tabuleiro de xadrez minúsculo). Você começa em um canto e precisa chegar ao outro canto. Cada passo custa 1 ponto (você não quer desperdiçar passos!) e chegar ao objetivo não te dá nada extra — você apenas quer chegar lá o mais rápido possível.

A **Iteração de Política** (Policy Iteration) é como um computador descobre os **melhores movimentos para cada quadrado** do tabuleiro — tudo de uma vez!

---

## A Grande Ideia: Dois Passos, Vez Após Vez

Pense nisso como limpar seu quarto com um ajudante:

1. **Passo 1 — Descobrir quão bom cada quadrado é (Avaliação de Política)**
   Seu ajudante caminha por cada quadrado e anota: "Se eu seguir o plano atual, quantos passos levarei para chegar à saída a partir daqui?". Eles fazem isso repetidamente até que os números parem de mudar.

2. **Passo 2 — Melhorar o plano (Melhoria de Política)**
   Agora você olha para cada quadrado e pergunta: "Existe uma direção melhor para a qual eu poderia ir a partir daqui?". Se sim, atualize o plano!

Repita os Passos 1 e 2 até que o plano pare de mudar — essa é a **política ótima**!

**Exemplo da vida real:** Imagine encontrar a rota mais rápida para a escola. Primeiro você chuta uma rota e cronometra o tempo (Passo 1). Depois, você olha para cada esquina e pergunta "existe um atalho daqui?" (Passo 2). Você atualiza sua rota e repete até não encontrar mais atalhos!

---

## O Que Nosso Código Encontrou

Nosso GridWorld 4×4 tem dois estados terminais (cantos), e o agente paga -1 por passo. A iteração de política convergiu em apenas **4 rodadas** (139 varreduras de avaliação totais):

```
Valores de Estado V(s):   Política Ótima:
 0.0  -1.0  -1.9  -2.7    T   ←   ←   ↓
-1.0  -1.9  -2.7  -1.9    ↑   ↑   ↑   ↓
-1.9  -2.7  -1.9  -1.0    ↑   ↑   ↓   ↓
-2.7  -1.9  -1.0   0.0    ↑   →   →   T
```

**Os valores fazem todo o sentido!** Quadrados próximos a um terminal têm valor -1 (um passo de distância). Quadrados a dois passos têm valor -1,9 (= -1 + 0,9 × -1), e assim por diante.

---

## Exemplos da Vida Real

- **Navegação GPS**: Descobrir a melhor curva em *cada* cruzamento no mapa.
- **Controle de elevador**: Para qual andar o elevador deve ir quando tem várias solicitações?
- **Robô de fábrica**: Planejar o caminho mais eficiente em uma grade de armazém.

---

## Palavras-Chave para Lembrar

- **Política (Policy)**: O plano — qual ação tomar em cada estado
- **Função de Valor V(s)**: Quão bom é estar no estado s (mais alto = mais perto do objetivo)
- **Avaliação de Política**: Calcular quão bom é o plano atual
- **Melhoria de Política**: Tornar o plano melhor usando a função de valor
- **Política Ótima**: O melhor plano possível — não pode ser melhorado mais

A grande ideia: **Você não precisa tentar todos os planos possíveis! Apenas continue melhorando o atual e você encontrará o melhor plano em pouquíssimas rodadas.**
