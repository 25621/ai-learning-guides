# SARSA vs Q-Learning: Caminhos Seguros vs Ideais 🐢 vs 🐇

## O que é isso?

Dois robôs precisam caminhar ao longo da **borda de um precipício** para chegar ao objetivo. Ambos os robôs
ainda estão *aprendendo* e às vezes fazem movimentos aleatórios (ops!).

- 🐢 **Robô SARSA**: "Eu sei que às vezes balanço... então vou caminhar bem longe do precipício
  para garantir a segurança, mesmo que demore mais."
- 🐇 **Robô Q-Learning**: "O caminho mais curto beira o precipício — vamos lá! (Cai algumas vezes
  enquanto aprende, mas acaba aprendendo a melhor rota.)"

Ambos os robôs são inteligentes, mas fazem uma **escolha diferente**: seguro-porém-mais-lento vs
ideal-porém-arriscado-durante-o-aprendizado.

---

## A Diferença Chave: Qual "Próxima Ação" você usa?

Ao atualizar as pontuações após cada passo, ambos os algoritmos perguntam:
> "Qual é o valor do *próximo estado*?"

| Algoritmo | Usa a próxima ação... | On-policy? |
|-----------|------------------------|------------|
| **SARSA** | ...que eu *realmente tomarei* (talvez aleatória!) | Sim |
| **Q-Learning** | ...que é *teoricamente a melhor* (sempre greedy) | Não |

**Exemplo da vida real:** Duas crianças aprendendo a andar de bicicleta.

- **Criança SARSA**: Fica perto da grama porque *ela sabe* que às vezes balança aleatoriamente.
  Ela está planejando para o seu eu real e instável.
- **Criança Q-Learning**: Pratica no meio do caminho porque está imaginando um eu futuro perfeito
  que nunca balança. Ela cai algumas vezes agora, mas aprende o melhor caminho mais rápido.

Ambas as crianças acabam aprendendo — mas durante o treinamento, a criança SARSA cai menos!

---

## O que nosso Código Encontrou

Ambos os algoritmos rodaram por **500 episódios** no Cliff Walking com ε=0,1 (ε = épsilon; aqui significa 10% de chance de fazer um movimento aleatório):

| Métrica | SARSA | Q-Learning |
|--------|-------|------------|
| Rec. média durante o treino (últimos 50 ep) | **-19,7** | **-51,0** |
| Avaliação Greedy (sem exploração) | -17 | **-13** |

- **Durante o treinamento**: O SARSA obtém **recompensas muito melhores** porque evita o precipício
  (levando em conta seus próprios movimentos aleatórios)
- **Após o treinamento** (puro greedy): O Q-Learning encontra o **caminho ideal mais curto** (-13)!

À medida que ε diminui em direção a 0, ambos os algoritmos convergem para a mesma política ideal.

---

## Resumo Visual

```
Caminho SARSA (durante o treino):    Caminho Q-Learning (greedy, após o treino):
[ ][→][→][→][→][→][→][→][→][→][→][↓]   [ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][↓]
[↑][→][→][→][→][→][→][→][→][→][→][↓]   [ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][↓]
[ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][↓]   [ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][↓]
[I][P][P][P][P][P][P][P][P][P][P][O]   [I][→][→][→][→][→][→][→][→][→][→][O]
    (desvio seguro, linhas de cima)          (ideal, beirando o precipício)
```

---

## Exemplos da Vida Real

- **Cirurgião novato vs cirurgião experiente**: O cirurgião novato (SARSA) mantém distância de técnicas arriscadas enquanto aprende. O cirurgião experiente (Q-Learning greedy) usa a técnica mais eficiente após dominá-la.
- **Direção na cidade vs rota pela rodovia**: Um planejamento tipo SARSA escolhe ruas residenciais mais seguras; o Q-Learning encontra a rodovia ideal, porém estreita.
- **Estudante estudando**: O estudante-SARSA foca em tópicos bem compreendidos durante a prática. O estudante-Q-Learning tenta os problemas mais difíceis (falha mais), mas aprende a estratégia ideal.

---

## Palavras-Chave para Lembrar

- **On-policy** (SARSA): Aprende sobre o que você *realmente faz*, incluindo a exploração aleatória
- **Off-policy** (Q-Learning): Aprende sobre o *melhor comportamento possível* independentemente do que você realmente faz
- **Caminho seguro**: Rota mais longa que evita o perigo, usada quando a exploração causa acidentes
- **Caminho ideal**: Rota mais curta/de maior recompensa, encontrada quando não há exploração
- **Exploration-exploitation tradeoff**: O equilíbrio entre tentar coisas novas e usar o que você já sabe

A grande ideia: **O SARSA é mais seguro durante o treinamento (on-policy), o Q-Learning encontra o caminho ideal mais rápido (off-policy). Qual é melhor depende se cair do precipício é um problema grave!**
