# Política Condicionada a Objetivos (Goal-Conditioned Policy)

## A Grande Ideia: Uma Política para Todos os Objetivos

Imagine que você é um motorista de entregas. Você não precisa de um conjunto de habilidades completamente diferente para cada endereço. Você sabe dirigir, ler um mapa e navegar no trânsito — você apenas insere o *destino de hoje* e vai.

Uma **política condicionada a objetivos** funciona da mesma maneira. Em vez de treinar um agente que só pode ir para um objetivo fixo, treinamos um único agente que aceita qualquer objetivo como entrada e descobre como chegar lá.

---

## Como Diferencia do RL Padrão

No RL padrão (como abordado nas fases anteriores do currículo), a função de recompensa é fixa: "chegue à célula (7, 7), ganhe +1". O agente aprende exatamente uma coisa: como chegar àquela célula específica.

No RL condicionado a objetivos, a recompensa depende se o agente atinge *qualquer objetivo que lhe foi dado desta vez*. A política aprende:

> **"Dado onde estou e onde quero estar, o que devo fazer?"**

O objetivo viaja *com* o agente, como um destino digitado em um aplicativo de navegação.

---

## O Problema da Recompensa Esparsa

Aqui está o problema: aprender com recompensas esparsas (apenas +1 no objetivo, 0 em todos os outros lugares) é brutalmente difícil. A maioria das tentativas falha — o agente vagueia aleatoriamente, nunca encontra o objetivo e a rede não recebe nada útil para aprender.

Imagine tentar aprender a jogar dardos com os olhos vendados. Você joga mil vezes e sempre erra. Depois de mil falhas, você ainda não tem ideia de como é um "bom arremesso".

É aqui que entra o **Hindsight Experience Replay (HER)**.

---

## Hindsight Experience Replay: Falhando para Frente

O truque do HER é lindamente simples. Após um episódio fracassado, o HER pergunta:

> *"Embora você não tenha atingido seu objetivo... onde você realmente terminou?"*

Ele então **reproduz esse mesmo episódio**, mas finge que a posição final real do agente **era** o objetivo o tempo todo. De repente, um episódio fracassado torna-se um sucesso — para um objetivo diferente.

É como um jogador de basquete que continua arremessando para a cesta e errando. O HER diria: "Tudo bem, você acertou a parede esquerda todas as vezes. Parabéns — você é ótimo em acertar a parede esquerda! Vamos registrar esses arremessos como tentativas bem-sucedidas de acertar a parede esquerda." Com o tempo, o jogador desenvolve habilidade em acertar *qualquer* alvo e, eventualmente, transfere isso para a cesta real.

Isso transforma milhares de "falhas" em uma rica biblioteca de navegações *bem-sucedidas* para muitos pontos diferentes. O agente aprende a alcançar todos eles, o que se generaliza para o alvo real.

---

## Analogia da Vida Real: Criança Aprendendo a Empilhar Blocos

Uma criança tentando colocar um bloco em um balde erra constantemente. Mas cada "erro" deixa o bloco em *algum lugar*. Se você reproduzir cada erro como "você estava tentando colocá-lo *exatamente ali* — e você conseguiu!", a criança desenvolve habilidades motoras finas em toda a mesa. Logo ela poderá colocar um bloco em qualquer lugar — inclusive no balde.

---

## O Que Nosso Código Faz

O script `goal_conditioned_policy.py` é executado em um **labirinto 7x7** com paredes. No início de cada episódio, uma célula de objetivo aleatória é escolhida. O agente deve encontrá-la.

A política recebe duas entradas em cada passo:
1. Onde o agente está atualmente
2. Onde ele quer ir

Após cada episódio (bem-sucedido ou não), o HER gera vários "sucessos" sintéticos adicionais, rotulando as posições reais visitadas como objetivos alternativos.

O treinamento dura 3.000 episódios com uma taxa de exploração decrescente — o agente explora mais no início e depois confia cada vez mais no que aprendeu.

---

## O Que os Gráficos Mostram

![Resultados da Política Condicionada a Objetivos](outputs/goal_conditioned_policy.png)

**Esquerda — Taxa de Sucesso Durante o Treinamento:** Cada episódio é um sucesso (atingiu o objetivo) ou uma falha. A curva sobe constantemente à medida que a habilidade de navegação universal do agente melhora. Ao final, o agente alcança qualquer objetivo quase todas as vezes.

**Direita — Mapa de Calor de Sucesso por Objetivo:** Após o treinamento, testamos o agente em cada célula de objetivo possível e colorimos cada célula pela frequência com que o agente a alcança. Verde significa que o agente chega a esse ponto de forma confiável; vermelho significa que ele ainda tem dificuldades. Um agente bem treinado mostra principalmente verde em todo o labirinto.

---

## Onde Isso Aparece no Mundo Real

| Aplicação | O "objetivo" |
|-----------|--------------|
| Braço robótico alcançando algo | Posição 3D do alvo |
| Carro autônomo | Coordenada GPS |
| Assistente de modelo de linguagem | Instrução do usuário |
| Personagem de videogame | Qualquer ponto no mapa |

Políticas condicionadas a objetivos são um dos blocos fundamentais para o HIRO (Hierarchical RL com subobjetivos) — o gerente de alto nível escolhe um subobjetivo, e o trabalhador de baixo nível é exatamente esse tipo de política condicionada a objetivos.

---

## Resumo de Uma Frase

> **Uma política condicionada a objetivos é um único agente que pode navegar para qualquer destino — e o HER torna possível aprender com a falha ao fingir que cada arremesso perdido foi direcionado para onde quer que tenha caído.**
