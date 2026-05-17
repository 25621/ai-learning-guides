# DPO: Pulando o Juiz e Indo Direto à Fonte

## A Ideia Principal

O RLHF clássico possui duas etapas: primeiro, treinar um modelo de recompensa (reward model) e, depois, usar o PPO para buscar as pontuações desse modelo. O DPO (Direct Preference Optimization) faz uma pergunta astuta:

*Se o modelo de recompensa é apenas um passo intermediário, podemos pulá-lo?*

A resposta é sim. O DPO treina o modelo de linguagem diretamente a partir de pares de preferência, sem a necessidade de um juiz separado, sem um loop de amostragem do PPO e sem a necessidade de ajustar um coeficiente KL. Ele utiliza uma única fórmula elegante e se comporta como um aprendizado supervisionado.

Isso torna o DPO mais simples de executar, mais estável e mais rápido — razão pela qual ele se tornou rapidamente a escolha padrão para muitos modelos alinhados de código aberto.

## Uma Analogia da Vida Real

Suponha que você esteja treinando um estudante para escrever ensaios.

A abordagem PPO seria: contratar um professor para dar nota aos ensaios, fazer o estudante escrever ensaio após ensaio e ajustá-los com base nas notas do professor.

A abordagem DPO seria: mostrar ao estudante dois ensaios por vez e dizer: "este aqui é melhor — tente escrever mais como este e menos como aquele outro". Sem o professor no meio do caminho. O estudante se ajusta diretamente a partir das comparações.

Ambos podem funcionar. O DPO geralmente termina mais rápido porque ninguém precisa treinar e manter um professor separado.

## Como o Aprendizado Funciona (Intuição)

O DPO usa os mesmos pares de preferência que a modelagem de recompensa: prompt, escolhido (chosen) e rejeitado (rejected). Para cada par, ele faz duas perguntas:

1. O modelo tornou-se **mais propenso** a produzir a resposta escolhida do que o modelo de referência seria?
2. O modelo tornou-se **menos propenso** a produzir a resposta rejeitada do que o modelo de referência seria?

O treinamento empurra ambos os números na direção certa ao mesmo tempo. Fundamentalmente, o modelo de referência está sempre presente na comparação — ele desempenha o mesmo papel que a penalidade KL no PPO. O modelo tem permissão para mudar, mas as mudanças são sempre *relativas* ao ponto de partida.

Um resultado sutil e belo do artigo original do DPO é que essa única função de perda (loss function) é matematicamente equivalente a "treinar um modelo de recompensa e depois executar o PPO com uma penalidade KL". Mesmo destino, jornada mais simples.

## O Que o Experimento Mostra

Treinamos uma política diretamente em 2.000 pares de preferência por 300 épocas.

![Treinamento DPO](outputs/dpo_implementation.png)

- **Esquerda** — a perda (loss) do DPO cai conforme o modelo aprende a preferir as respostas escolhidas às rejeitadas.
- **Meio** — a precisão da preferência (com que frequência a política atribui uma recompensa implícita maior à resposta escolhida) sobe para cerca de 99%.
- **Direita** — a margem de recompensa implícita cresce. O DPO nunca nomeia uma "recompensa" explicitamente, mas a lacuna entre as log-probabilidades do escolhido vs. rejeitado, escalada por beta, pode ser lida como tal. Ela aumenta constantemente, o que significa que o modelo se torna mais confiante em suas preferências.

Observe como isso parece limpo em comparação ao PPO. Não há loop de amostragem, não há ruído de exploração e não há um modelo de recompensa separado rodando. Cada época é uma atualização pura de estilo supervisionado sobre o conjunto de dados de preferências.

## Onde Isso se Encaixa no Fluxo (Pipeline) de RLHF

O DPO é uma *alternativa* às etapas dois e três do fluxo clássico:

- **Clássico:** preferências → modelo de recompensa → PPO → modelo alinhado.
- **DPO:** preferências → modelo alinhado. (Pronto.)

O detalhe é que o DPO treina sobre um conjunto de dados de preferência fixo. O PPO, por amostrar respostas novas em cada rodada, pode, em princípio, explorar mais. Na prática, o DPO vence para a maioria dos casos de uso de "alinhamento em um conjunto de dados de preferência curado".

## Por Que Isso Importa Fora do Laboratório

O padrão de "pular a medição intermediária" está em toda parte:

- Um treinador corrigindo a forma de um nadador através de demonstrações comparativas, em vez de cronometrar cada volta.
- Um fotógrafo editando duas fotos por vez, escolhendo a melhor, em vez de criar uma rubrica de pontuação para o que seria uma "foto boa".
- Um recrutador comparando dois currículos em vez de pontuar cada um em relação a uma lista de 30 requisitos.

Quando você só precisa *ranquear*, não precisa de uma escala absoluta. O DPO é essa percepção aplicada a modelos de linguagem.

## Resumo de Uma Frase

**O DPO transforma pares de preferência diretamente em um modelo melhor, sem um modelo de recompensa intermediário — mais simples que o PPO e, muitas vezes, tão bom quanto.**
