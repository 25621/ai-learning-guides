# Modelagem de Recompensa: Ensinando ao Computador o que as Pessoas Preferem

## A Grande Ideia

Um modelo de recompensa é um pequeno juiz. Você mostra a ele duas respostas para a mesma
pergunta, diz qual delas uma pessoa gostou mais e, com o tempo, ele aprende a
dar uma pontuação mais alta para respostas que as pessoas prefeririam.

Por que precisamos de tal juiz? Porque a maior parte do que queremos de um modelo de linguagem
é difícil de descrever como uma fórmula matemática. Não existe uma equação única para
"útil", "educado" ou "bem escrito". Mas as pessoas quase sempre conseguem
apontar a melhor entre duas opções. O modelo de recompensa transforma esses simples
votos de "este é melhor" em uma pontuação que um algoritmo de aprendizado pode usar.

## Uma Analogia da Vida Real

Imagine ensinar um amigo a fazer brownies.

Você não entrega a ele um livro de regras de 50 páginas sobre "o que torna um brownie bom".
Em vez disso, você prova duas fornadas e diz:

"Este é melhor."

Depois de algumas rodadas disso, seu amigo começa a notar padrões. Talvez
o mais cremoso sempre vença. Talvez o que passou do ponto sempre perca. Seu
amigo constrói um sistema de pontuação mental a partir das suas comparações.

Um modelo de recompensa faz exatamente isso, mas com números. Ele não precisa saber
*por que* a resposta escolhida é melhor. Ele só precisa de muitos exemplos de "este vence
aquele" e gradualmente aprende uma pontuação que se alinha com as preferências.

## Como o Aprendizado Funciona (Apenas Intuição)

Cada exemplo é um trio: um comando (prompt), uma resposta **escolhida** (chosen) e uma
resposta **rejeitada** (rejected). Queremos que o modelo dê uma pontuação mais alta à
escolhida do que à rejeitada — por qualquer margem.

O empurrãozinho de treinamento é simples em espírito:

- Pontuação da escolhida muito baixa? Empurra para cima.
- Pontuação da rejeitada muito alta? Empurra para baixo.
- Já estão na ordem certa com uma diferença clara? Deixa como está.

Esse empurrão é chamado de perda de Bradley-Terry (Bradley-Terry loss) e é a receita padrão
nos sistemas modernos de RLHF.

## O que o Experimento Mostra

Treinamos um modelo de recompensa em 2.000 pares de preferência sintéticos. O gráfico
abaixo mostra três visões da mesma rodada de treinamento.

![Treinamento do modelo de recompensa](outputs/reward_modeling.png)

- **Esquerda** — a perda (loss) cai rapidamente. O modelo está se tornando mais confiante
  sobre suas classificações.
- **Meio** — a precisão da preferência sobe para quase 100%. Em quase todos os
  pares, a resposta escolhida recebe uma pontuação maior que a rejeitada.
- **Direita** — as distribuições de pontuação para respostas escolhidas vs rejeitadas se
  afastam. No início, elas se sobrepunham; após o treinamento, as respostas escolhidas
  estão claramente à direita.

Essa separação é o objetivo principal. Uma etapa posterior (PPO ou DPO) pode agora usar
essa pontuação como um alvo para otimização.

## Onde isso se Enquadra no Pipeline de RLHF

O roteiro descreve o RLHF como "alinhamento de modelos com as preferências humanas".
O modelo de recompensa é o primeiro de três passos:

1. **Modelo de recompensa (este arquivo)** — converter votos de preferência em uma pontuação.
2. **Ajuste fino com PPO** — empurrar o modelo de linguagem em direção a pontuações mais altas,
   mantendo-o próximo de seu comportamento original.
3. **DPO** — um atalho mais novo que pula o modelo de recompensa inteiramente.

Portanto, a modelagem de recompensa é a ponte entre o *julgamento humano* e a
*otimização de máquina*. Se essa ponte for construída de forma errada, cada passo subsequente
será desviado do curso.

## Por que isso Importa Fora do Laboratório

A mesma ideia aparece em muitos lugares:

- **Sistemas de recomendação** aprendem do que você gosta a partir de cliques, pulos e
  tempo gasto assistindo.
- **Motores de busca** aprendem o ranking a partir de "em qual resultado você clicou".
- **Restaurantes** aprendem os pratos populares a partir de pedidos repetidos, não de
  clientes escrevendo ensaios sobre o que gostaram.

Sempre que é mais fácil *comparar* do que *dar uma nota*, um modelo de recompensa é a
ferramenta certa.

## Resumo em uma Frase

**Um modelo de recompensa é um juiz treinado que transforma preferências de "este é melhor"
em uma pontuação numérica que o resto do RLHF pode otimizar.**
