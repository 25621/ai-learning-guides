# DQN no Atari Pong 🏓

## O que é o Atari Pong?

O Pong é um dos videogames mais antigos já criados — é como um tênis de mesa digital! Duas raquetes rebatem uma bola para frente e para trás. Você ganha um ponto se o oponente perder a bola. O jogo termina quando alguém atinge 21 pontos.

Em nossa versão, a IA controla uma raquete e o oponente (computador) controla a outra. O jogo sempre começa com a pontuação de −21 (pior caso possível). Um bom agente atinge 0 ou +21.

---

## Por que o Pong é Difícil para uma IA?

No CartPole, o robô conseguia VER os números diretamente (ângulo da haste, velocidade do carrinho...). No Pong, tudo o que ele vê são **pixels brutos** — milhares de pontinhos coloridos em uma tela!

```
Entrada do CartPole: [0.02, −0.14, 0.01, −0.23]   ← 4 números, fácil!
Entrada do Pong:     [grade de pixels: 210×160×3]  ← 100.800 números, MUITO mais difícil!
```

O robô precisa descobrir a partir dos pixels:
- Onde está minha raquete?
- Onde está a bola?
- A bola está se movendo para a esquerda ou para a direita?
- Quão rápido?

Os humanos fazem isso automaticamente (temos uma visão incrível!). Para uma IA, isso é um desafio enorme.

---

## Enxergando o Movimento: Empilhamento de Frames (Frame Stacking) 🎬

Um único frame (captura de tela) não diz se a bola está se movendo para a esquerda ou para a direita. Você precisa ver MÚLTIPLOS frames para entender o movimento — assim como um "flip book" (livro animado) só funciona quando você folheia várias páginas.

**Empilhamento de Frames:** Fornece os últimos 4 frames para a rede simultaneamente.

```
Frame 1: bola na posição 40
Frame 2: bola na posição 43    → Empilha esses 4 frames → A rede vê o MOVIMENTO!
Frame 3: bola na posição 46
Frame 4: bola na posição 49
```

A rede agora pode inferir: "a bola está se movendo para a direita na velocidade 3".

**Exemplo da vida real:** Assistir a um filme vs. olhar para um único frame. Um frame de uma corrida de carros é apenas uma imagem borrada. Assista a 4 frames e você saberá qual carro é mais rápido!

---

## Enxergando com uma CNN 🔍

Para entradas de pixels, usamos uma rede neural especial chamada **Rede Neural Convolucional (CNN)**. Em vez de olhar para todos os pixels de uma vez, uma CNN usa janelas deslizantes para detectar padrões — como olhos escaneando uma imagem.

```
Pixels brutos (84×84×4 frames)
       ↓
Camada Conv 1 (filtro 8×8, stride 4) → encontra bordas e formas
       ↓
Camada Conv 2 (filtro 4×4, stride 2) → encontra objetos (raquetes, bola)
       ↓
Camada Conv 3 (filtro 3×3, stride 1) → encontra relações
       ↓
Achatamento (Flatten) → 512 neurônios → Valores Q (um por ação)
```

**Exemplo da vida real:** Quando você procura seu amigo em uma multidão, seu cérebro primeiro percebe formas (uma pessoa), depois características (cor do cabelo) e então detalhes (o rosto). As CNNs funcionam da mesma maneira — de padrões simples para complexos!

---

## Pré-processamento: Encolhendo o Mundo

Os frames do Pong têm 210×160 pixels coloridos. Isso é grande demais! Pré-processamos cada frame:

1. **Escala de cinza (Grayscale)** — a cor não importa para o Pong (a bola é sempre branca de qualquer maneira).
2. **Redimensionar para 84×84** — menor = treinamento mais rápido, mas ainda claro o suficiente para enxergar.
3. **Normalizar para [0,1]** — dividir os valores dos pixels por 255 para que sejam números pequenos.

**Exemplo da vida real:** Como fazer uma fotocópia com 50% do tamanho. Os detalhes importantes (bola, raquetes) ainda são visíveis, apenas menores. A fotocopiadora também não se importa com cores!

---

## Corte de Recompensas (Reward Clipping): Tratando Todos os Jogos Igualmente ✂️

No Pong, você ganha +1 por marcar ponto e −1 por sofrer ponto. Em alguns outros jogos do Atari, as pontuações podem chegar aos milhares!

Nós **cortamos as recompensas** (reward clipping) para o intervalo [−1, +1] para que a rede não se importe com a escala das recompensas. Esse mesmo código pode treinar em QUALQUER jogo do Atari sem precisar ajustar as escalas de recompensa.

---

## Quanto Tempo Leva o Treinamento?

| Duração do Treinamento | O que o Agente Aprende |
|---|---|
| 100K passos | Majoritariamente aleatório, mal reage |
| 1M passos | Começa a se mover em direção à bola às vezes |
| 5M passos | Devolve algumas jogadas |
| 10M passos | Jogo competitivo, pode ganhar algumas vezes |
| 20M+ passos | Frequentemente vence o oponente do computador |

Nossa demonstração executa **300K passos** — o suficiente para ver que a arquitetura de treinamento funciona e observar o aprendizado inicial, mas não o suficiente para dominar o jogo.

**Exemplo da vida real:** Aprender piano leva meses. Uma sessão de prática de 10 minutos mostra que você está no caminho certo, mas não espere dar concertos ainda!

---

## O Que Nosso Código Encontrou

Após 300K passos no Pong:
- O agente começa com pontuações em torno de −20 (mal consegue devolver a bola).
- Ao final, ele normalmente melhora para cerca de −15 a −10.
- A curva de aprendizado mostra uma melhora gradual a partir do jogo aleatório.

Para ver um desempenho de Pong realmente competitivo, você precisaria rodar por ~10M+ de passos com uma GPU. A implementação está completa e correta — só precisa de mais tempo!

---

## Vocabulário Chave

| Palavra | Significado |
|---------|-------------|
| **CNN** | Rede Neural Convolucional — especializada para entradas de imagem |
| **Frame Stacking** | Fornecer múltiplos frames consecutivos para capturar o movimento |
| **Pré-processamento** | Transformar frames brutos (escala de cinza, redimensionar, normalizar) |
| **Reward Clipping** | Limitar recompensas a [−1, +1] para funcionar em diferentes jogos |
| **ALE** | Arcade Learning Environment — a biblioteca que roda os jogos de Atari |

---

## A Conquista Histórica

Quando a DeepMind publicou o DQN em 2015, o mundo ficou maravilhado. Um ÚNICO algoritmo, com a MESMA arquitetura, aprendeu a jogar 49 jogos diferentes de Atari — muitos em nível sobre-humano — apenas a partir de pixels brutos e da pontuação!

Antes do DQN, as pessoas pensavam que era necessário programar manualmente a estratégia de cada jogo. O DQN mostrou que um aprendiz de propósito geral poderia descobrir tudo sozinho. Foi um momento histórico na IA!
