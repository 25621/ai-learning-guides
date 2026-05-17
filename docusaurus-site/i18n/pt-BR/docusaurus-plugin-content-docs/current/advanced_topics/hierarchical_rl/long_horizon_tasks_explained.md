# Tarefas de Horizonte Longo (Long-Horizon Tasks)

## A Grande Ideia: Quando a Recompensa Está Muito Longe

Imagine que você é um chef tentando aprender uma receita nova apenas provando o prato final. Você segue 40 passos — picar, refogar, temperar, cozinhar, empratar — mas só recebe feedback no final: "Muito salgado". Qual dos 40 passos causou o problema? Você não tem ideia.

Esse é o **problema do horizonte longo**: quando o sinal de recompensa está separado das decisões que o causaram por dezenas (ou centenas) de passos, o aprendizado torna-se muito difícil.

---

## Por Que Agentes Planos Têm Dificuldade

Um agente de RL plano (como os agentes DQN da Fase 3) tenta aprender o valor de cada passo individual de uma só vez. Em tarefas curtas — equilibrar uma haste, evitar uma parede — isso funciona bem. A recompensa chega rápido e o agente consegue conectar causa e efeito.

Mas em uma tarefa longa — coletar uma chave, usá-la para abrir uma porta e depois sair do labirinto — o agente precisa:

1. Tropeçar na chave (sorte!)
2. Lembrar que coletar chaves é útil
3. Tropeçar na porta (sorte de novo!)
4. Conectar toda a sequência à única recompensa na saída

Com a exploração aleatória, a chance de completar acidentalmente toda essa sequência diminui exponencialmente a cada novo passo necessário. O DQN plano essencialmente precisa ter sorte muitas, muitas vezes antes de ver uma única recompensa positiva para aprender.

---

## A Solução Hierárquica: Dividir para Conquistar

O RL hierárquico divide a tarefa longa em uma **estrutura de dois níveis**:

| Nível | Chamado de | Função |
|-------|------------|--------|
| Alto  | **Gerente** | Escolhe o próximo sub-objetivo |
| Baixo | **Trabalhador** | Navega até esse sub-objetivo |

É exatamente assim que os humanos abordam tarefas complexas. Você não planeja sua viagem de carro curva por curva antes de sair. Em vez disso:

- **Gerente (você, em casa):** "Primeira parada: o posto de gasolina. Próxima parada: a entrada da rodovia. Depois: saída 42."
- **Trabalhador (você, dirigindo):** Cuida de todas as decisões individuais do volante para chegar a cada parada.

O gerente pensa em *pontos de controle*. O trabalhador pensa em *volante*.

---

## Por Que Isso Supera o Aprendizado Plano em Tarefas Longas

O trabalhador só precisa alcançar o *próximo sub-objetivo* — uma tarefa curta com uma recompensa clara e próxima. Ele recebe feedback rapidamente e aprende de forma eficiente.

O gerente só precisa decidir a *ordem dos sub-objetivos* — um problema muito mais simples do que planejar cada passo individual.

Juntos, os dois níveis dividem o difícil problema de horizonte longo em dois problemas fáceis de horizonte curto.

---

## O Experimento do Labirinto Chave-Porta

Nosso script testa ambas as abordagens em uma **grade aberta 9x9** com dois objetos:

- Uma **CHAVE** em um canto (deve ser coletada primeiro).
- Uma **PORTA** no canto oposto (só conta se você tiver a chave).

A única recompensa real é +1 quando o agente chega à porta *depois* de pegar a chave. Essa única recompensa exige que duas sub-tarefas sequenciais sejam encadeadas corretamente.

Dois agentes competem:

**DQN Plano:** Precisa tropeçar em ambas as sub-tarefas na ordem correta por acidente e, em seguida, propagar o sinal por ambas. Como o sucesso exige dois achados sortudos em um episódio, o DQN raramente aprende algo útil.

**Agente Hierárquico:**
- Regra do gerente: "Vá para a chave primeiro, depois vá para a porta."
- O trabalhador recebe **+1 cada vez que atinge um sub-objetivo** — seja chave ou porta.
- Duas tarefas curtas separadas, cada uma com uma recompensa clara e próxima.

---

## O Que os Gráficos Mostram

![Resultados de Tarefas de Horizonte Longo](outputs/long_horizon_tasks.png)

**Esquerda — Taxa de Sucesso ao Longo do Tempo:** O agente hierárquico (azul) aprende a resolver o labirinto muito antes do que o DQN plano (vermelho). O agente plano pode eventualmente aprender também — se tiver episódios suficientes — mas o agente hierárquico chega lá mais rápido porque seu sinal de aprendizado é denso e local.

**Direita — Desempenho Final:** O gráfico de barras mostra a taxa de sucesso média nos últimos 500 episódios. A vantagem do agente hierárquico é clara: dividir o problema em sub-objetivos o torna tratável.

---

## Onde o Pensamento de Horizonte Longo Aparece

| Domínio | Exemplo de horizonte longo |
|---------|---------------------------|
| Robótica | Montar um dispositivo com 30 peças em ordem |
| Jogos | Vencer uma partida de xadrez (muitos movimentos, um vencedor) |
| Linguagem | Escrever um artigo científico completo (muitas decisões de escrita, uma nota de qualidade) |
| Ciência | Realizar um experimento de vários meses e avaliar os resultados |

É exatamente por isso que as Redes Feudais (uma arquitetura onde gerentes definem objetivos direcionais para trabalhadores de nível inferior) e o HIRO (RL Hierárquico com sub-objetivos) foram inventados — conforme o RL plano encontrava obstáculos nesses problemas, a decomposição hierárquica tornou-se a estratégia dominante.

---

## A Conexão com as Políticas Condicionadas a Objetivos

Observe que o **trabalhador** em nosso agente hierárquico é essencialmente uma **política condicionada a objetivos** — ele recebe um sub-objetivo e navega até ele. Este é o design padrão no HIRO e artigos relacionados: o gerente define objetivos, o trabalhador é uma política condicionada a objetivos que os persegue.

As duas ideias — políticas condicionadas a objetivos e estrutura hierárquica — são, portanto, dois lados da mesma moeda, e é por isso que aparecem juntas neste módulo.

---

## Resumo de Uma Frase

> **Tarefas de horizonte longo são difíceis porque a recompensa chega tarde demais para ensinar decisões individuais — o RL hierárquico resolve isso inserindo sub-objetivos próximos que permitem ao trabalhador aprender rápido, enquanto o gerente cuida da sequência geral.**
