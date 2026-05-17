# Dyna-Q: Aprendendo Mais Rápido Através da Imaginação 🧠

## O Que É?

Imagine uma criança chamada Mia aprendendo a se localizar em sua nova escola. Todos os dias ela caminha pelos corredores e descobre coisas novas: "A biblioteca fica depois do refeitório", "A sala do Sr. Smith é no andar de cima, perto da escada".

Um estudante de **Q-learning puro** aprende apenas com o que faz *hoje*. Se hoje ela apenas caminhou da sala para o refeitório, ela só atualiza sua memória sobre esse único caminho.

Um estudante de **Dyna-Q** é diferente. Após cada caminhada real, ela se senta por um minuto e **reproduz em sua mente** várias caminhadas passadas de que se lembra. Cada reprodução fortalece seu mapa mental. Depois de algumas semanas, ela conhece a escola inteira — não porque caminhou mais, mas porque **pensou mais sobre o que já viu**.

É exatamente isso que o Dyna-Q faz por um agente de RL: ele aprende com a experiência real **e** com a experiência imaginada, extraída de um modelo que ele constrói ao longo do caminho.

---

## Os Três Ingredientes

Dyna-Q é "Q-learning + modelo + planejamento". Um único passo real realiza **três** funções:

1. **RL Direto** — a atualização usual do Q-learning a partir de `(s, a, r, s')`.
2. **Aprendizado do modelo** — registrar: "Quando fiz *a* em *s*, obtive *r* e terminei em *s'*."
3. **Planejamento** — escolher *n* pares `(s, a)` aleatórios da memória do modelo e realizar *n* atualizações extras de Q-learning, **fingindo** que esses passos acabaram de acontecer.

Esse terceiro passo é a mágica. Com `n = 50`, cada passo real no mundo provoca **51 atualizações** na tabela Q. O agente aprende cerca de 50 vezes mais rápido — em termos de passos reais — do que um aprendiz de Q-learning puro.

---

## Uma Imagem do Loop

```
                   ┌────────────────────────────────────┐
                   │                                    │
   mundo real  ──► realizar ação a ──► observar (r, s') │
                            │                           │
              ┌─────────────┼──────────────┐            │
              ▼             ▼              ▼            │
        Atualização     Modelo[s,a] ← (r,s') Planejamento ─┘
        Q-learning                          (n atualizações imaginadas)
```

O modelo é apenas uma tabela de consulta (lookup table):
`(estado, ação) → (recompensa, próximo_estado)`. Barato para construir, barato para consultar.

---

## Exemplos da Vida Real

- **Estudo de xadrez.** Grandes mestres passam horas reproduzindo suas próprias partidas e partidas de mestres em suas cabeças. Cada reprodução é um "planejamento" — aprendizado extra de experiências que já aconteceram.
- **Um músico praticando escalas.** Depois de tocar um compasso difícil uma vez, ele o ensaia mentalmente mais dez vezes antes de prosseguir. Os dedos não estão se movendo, mas o cérebro está se atualizando.
- **Um carro autônomo.** Enquanto espera no sinal vermelho, ele reproduz as últimas cem mudanças de faixa em simulação para ajustar sua política sem gastar pneus.

---

## O Que Nosso Código Faz

Usamos o clássico **Dyna Maze** ([Sutton & Barto, Figura 8.2](http://incompleteideas.net/book/the-book.html)): uma grade 6×9 com algumas paredes, um início `S` no centro-esquerdo e um objetivo `G` no canto superior direito.

Executamos três variantes, cada uma com a média de 30 sementes (seeds) aleatórias:

| Configuração | Passos de planejamento por passo real | Significado |
|--------------|---------------------------------------|-------------|
| `n = 0`      | 0                                     | Q-learning puro |
| `n = 5`      | 5                                     | um pouco de prática imaginada |
| `n = 50`     | 50                                    | muita prática imaginada |

O script relata o **número médio de passos reais por episódio** à medida que o treinamento avança. Menos passos significam que o agente aprendeu um caminho mais direto para o objetivo.

### O que você deve ver ao executá-lo

O caminho mais curto neste laberinto é de cerca de 9 passos; com a exploração ε-greedy, um agente bem treinado faz uma média de 10 passos por episódio. Execute por 50 episódios e todas as três configurações convergirão para esse valor — a diferença é *quão rápido*:

| Configuração | Passos por episódio (últimos 10 eps) | O que significa |
|--------------|--------------------------------------:|-----------------|
| `n = 0`      | ~10 | Convergiu — mas levou de 30 a 50 episódios vagando para chegar aqui |
| `n = 5`      | ~10 | Convergiu em cerca de 10 episódios |
| `n = 50`     | ~10 | Convergiu em cerca de 3 a 5 episódios |

O sinal interessante é a *curva de aprendizado*, não o número final. O gráfico salvo em `outputs/dyna_q.png` mostra três curvas mergulhando em direção ao chão em ritmos muito diferentes: `n = 50` chega lá em poucos episódios, enquanto `n = 0` ainda está descendo bem tarde na execução. (Em um laberinto determinístico minúsculo como este, o Q-learning puro eventualmente chega lá — o Dyna-Q apenas precisa de muito menos episódios reais, que é todo o objetivo em ambientes onde passos reais são caros.)

---

## Por Que Funciona Tão Bem Neste Laberinto

Duas razões:

1. **O ambiente é determinístico.** Cada `(s, a)` sempre resulta no mesmo `(r, s')`, então o modelo é exato após uma única visita. A experiência imaginada é tão boa quanto a experiência real.
2. **Passos reais são caros, passos imaginados são gratuitos.** Cada atualização imaginada é apenas algumas consultas em tabelas, enquanto um passo real exige que o agente caminhe. Quando as interações reais custam caro (pense: um robô real, um jogo real), o Dyna-Q é enormemente eficiente em termos de amostragem.

---

## Onde o Dyna-Q Tem Dificuldades

- **Ambientes estocásticos.** Se `(s, a)` pode levar a muitos valores `s'` diferentes, um modelo de "lembrar o último resultado" mentirá para você. Solução: armazenar contagens de visitas ou treinar um modelo probabilístico.
- **Ambientes não estacionários.** Se o mundo mudar — por exemplo, uma porta que estava aberta se fechar de repente, ou um atalho aparecer — o modelo torna-se desatualizado e fornece previsões erradas. O **Dyna-Q+** resolve isso adicionando um *bônus de exploração*: estados que não foram revisitados por muito tempo recebem uma pequena recompensa extra, estimulando o agente a voltar e verificar se o mundo mudou.
- **Espaços de estados grandes.** Um dicionário indexado em `(s, a)` não escala para milhões de estados ou estados contínuos. É exatamente essa a lacuna que os **modelos de mundo aprendidos (redes neurais)** preenchem — veja `world_model.py` a seguir.

---

## Palavras-Chave para Lembrar

| Palavra | Significado |
|------|---------|
| **Modelo** | Memória de `(estado, ação) → (recompensa, próximo_estado)` |
| **Passo de Planejamento** | Realizar uma atualização Q usando dados imaginados do modelo |
| **RL Direto** | Uma atualização Q usando dados reais |
| **Eficiência de Amostragem** | Mede quão efetivamente um modelo ou algoritmo de IA usa os dados de treinamento para atingir um nível específico de desempenho |
| **Dyna** | Arquitetura de Sutton que intercala aprendizado + planejamento |

---

## Resumo de Uma Frase

> **O Dyna-Q aprende fazendo E imaginando — e imaginar é de graça.**

Esta ideia, em sua forma neural moderna, impulsiona alguns dos agentes de RL mais fortes já construídos (MuZero, Dreamer, World Models).
