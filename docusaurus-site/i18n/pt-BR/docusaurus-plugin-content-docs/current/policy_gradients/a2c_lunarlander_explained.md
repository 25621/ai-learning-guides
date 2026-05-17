# A2C: O Ator e o Crítico Trabalham Juntos

## A Ideia Principal

O REINFORCE espera até que o jogo termine completamente antes de realizar uma atualização. Isso é como um treinador que assiste a um jogo de futebol inteiro em silêncio e só dá o feedback ao final.

O **A2C (Advantage Actor-Critic)** fornece feedback DURANTE o jogo — a cada poucos passos, o treinador faz uma pausa para dizer: "Aquele passe foi ótimo! Aquele desarme foi ruim!"

Isso é muito mais rápido e eficiente.

---

## Conheça os Dois Personagens

> **O que é LunarLander?** Ao longo deste documento, utilizamos o ambiente **LunarLander** — uma simulação física onde você controla uma pequena espaçonave e deve pousá-la suavemente em uma plataforma na lua usando três motores (esquerdo, principal e direito). É um benchmark padrão em aprendizado por reforço, disponível na biblioteca Gymnasium.

### O Ator 🎭
O **Ator** é a política (policy) — ele decide qual ação tomar.

> "Estou neste estado. Devo acionar o motor esquerdo ou o motor direito?"

**Exemplo da vida real:** O *motorista* de um carro que gira o volante e pressiona os pedais.

### O Crítico 🎬
O **Crítico** estima quão boa é a situação atual — o valor V(s).

> "Estar NESTE estado vale cerca de +150 pontos de recompensa futura total."

**Exemplo da vida real:** O *navegador* sentado ao lado do motorista, dizendo: "Estamos em uma boa estrada — espero chegar em 30 minutos" ou "Estamos entrando em um engarrafamento — isso vai ser demorado."

### Eles Compartilham um Cérebro
Em nossa implementação, ambos usam a **mesma base da rede neural**:

```
          Estado (8 números para o LunarLander)
                       ↓
          ┌─────────────────────────┐
          │  Camadas Compartilhadas  │
          │  [256 neurônios] → ReLU  │
          │  [256 neurônios] → ReLU  │
          └────────┬────────┬───────┘
                   ↓        ↓
            Cabeça do Ator   Cabeça do Crítico
            [4 saídas]       [1 saída]
         (probs de ação)     (V(s))
```

- **ReLU** (Rectified Linear Unit): uma função de ativação aplicada após cada camada — ela retorna `max(0, x)`, mantendo valores positivos e zerando os negativos. Isso permite que a rede aprenda padrões não lineares.
- **probs de ação**: a probabilidade de tomar cada uma das 4 ações. O Ator amostra desta distribuição para escolher uma ação a cada passo.

**Exemplo da vida real:** Um cérebro, duas funções — como um taxista que dirige (ator) E sabe se a rota é boa (crítico). Compartilhar o cérebro significa aprender mais rápido!

---

## A Vantagem (Advantage): Foi Melhor do que o Esperado?

Assim como o REINFORCE com baseline, o A2C computa a **Vantagem**:

> A(s, a) = "Resultado real" − "O que esperávamos"

Mas aqui, o "resultado real" vem do **bootstrap de n-passos** do Crítico (**bootstrapping** = usar a própria previsão V(s) do Crítico para aproximar o valor de passos futuros, em vez de esperar o episódio terminar — como estimar sua nota final no meio do semestre usando sua nota atual):

```
Retorno TD real: r_t + γ · r_{t+1} + γ² · r_{t+2} + ... + γⁿ · V(s_{t+n})
Vantagem A_t = Retorno TD - V(s_t)
```

**Exemplo da vida real:** Você espera marcar 3 gols neste jogo (V(s)). Se marcar 5 gols, sua vantagem é +2. Se marcar 1 gol, sua vantagem é -2.

Vantagem positiva → "aquela ação ajudou mais do que o esperado → faça-a mais!"
Vantagem negativa → "aquela ação ajudou menos do que o esperado → faça-a menos!"

---

## Por que Usar Múltiplos Ambientes Paralelos?

Nosso A2C usa **8 cópias** do LunarLander rodando ao mesmo tempo!

**Por quê?** Porque as experiências de um único ambiente são **correlacionadas** — um passo segue o passo anterior de perto. Essa correlação engana a rede neural, fazendo-a pensar que os padrões são mais confiáveis do que realmente são.

Com 8 ambientes, cada passo fornece 8 experiências independentes de situações muito diferentes. Isso quebra a correlação e estabiliza o treinamento drasticamente.

**Exemplo da vida real:** Para aprender sobre o clima, o que é melhor?
- Observar uma cidade por 8 horas consecutivas (correlacionado — se estava sol às 14h, provavelmente estará sol às 15h)
- Observar 8 cidades simultaneamente (descorrelacionado — diferentes padrões climáticos, mais informação!)

```
Ambiente 1: [pousou na lua, motor esquerdo, colisão, reset...]
Ambiente 2: [caindo rápido demais, ambos motores, pairar, pousar...]
Ambiente 3: [inclinando à direita, motor direito, estabilizar, pousar...]
...
Ambiente 8: [derivando à esquerda, motor esquerdo, estável, ...]
```

Todos os 8 atualizam a rede simultaneamente — 8x mais experiências diversas por atualização!

---

## Atualizações de N-Passos: Não Espere o Jogo Terminar

O REINFORCE espera por um episódio completo (que pode ter 1000 passos!).

O A2C atualiza a cada **n_steps = 128 passos**:

```
Jogue 128 passos em 8 ambientes
    → Obtenha 128 × 8 = 1024 tuplas de experiência
    → Compute vantagens e retornos
    → Atualize o Ator e o Crítico
    → Jogue mais 128 passos...
```

**Exemplo da vida real:** Um estudante estudando para uma prova.
- Estilo REINFORCE: Lê o livro inteiro e SÓ DEPOIS faz os simulados.
- Estilo A2C: Lê 10 páginas, faz um quiz, lê mais 10 páginas, faz outro quiz...

Feedback mais frequente = aprendizado mais rápido!

---

## Três Perdas Combinadas

O A2C treina com três termos de perda (loss) simultaneamente:

Uma **perda** é o número que o otimizador tenta minimizar. Uma perda menor significa que o comportamento atual da rede está mais próximo do objetivo de treinamento.

### 1. Perda do Ator (Gradiente de Política)
Torna as ações vantajosas mais prováveis:
```
L_actor = -E[log π(a|s) · A(s,a)]
```
Se A > 0: aumenta a probabilidade daquela ação
Se A < 0: diminui a probabilidade daquela ação

### 2. Perda do Crítico (MSE da Função de Valor)
Torna as previsões de valor mais precisas (**MSE** = Erro Quadrático Médio: eleva o erro de previsão ao quadrado e tira a média — elevar ao quadrado penaliza erros grandes mais pesadamente do que erros pequenos):
```
L_critic = E[(V(s) - retorno)²]
```
Como treinar qualquer modelo de **regressão** (regressão = prever um número contínuo, aqui o retorno esperado V(s)) — minimiza o erro de previsão.

### 3. Bônus de Entropia (Exploração)
Evita que a política se torne confiante demais rápido demais:
```
L_entropy = -H[π(·|s)] = E[log π(a|s)]
```
Alta entropia = escolhas de ações diversas = exploração (exploration)
Baixa entropia = escolhas estreitas e confiantes = explotação (exploitation)

**Exemplo da vida real:** O bônus de entropia é como um professor dizendo: "Não chute apenas a letra A em todas as questões de múltipla escolha! Tente respostas diferentes para aprender o que funciona."

```
Perda total = L_actor + 0.5 × L_critic - 0.01 × entropia
```

---

## LunarLander: Um Desafio Maior

**LunarLander-v3** é um ambiente do Gymnasium (antigo OpenAI Gym) — "v3" é o número da versão, indicando a terceira revisão deste ambiente. O agente controla uma pequena espaçonave que deve pousar com segurança em uma plataforma designada na lua. É muito mais difícil que o CartPole:
- Espaço de estados de 8 dimensões (posição, velocidade, ângulo, contato das pernas, combustível)
- 4 ações discretas (não fazer nada, motor esquerdo, motor principal, motor direito)
- Recompensa: +100 por pousar, -100 por colidir, pequenas penalidades por combustível

A curva de treinamento mostra uma melhora gradual de recompensas altamente negativas para positivas. O A2C no LunarLander requer uma experiência significativa antes que o pousador aprenda a estabilidade básica.

---

## Equações Chave

```
retorno de n-passos:  G_t = r_t + γ·r_{t+1} + ... + γⁿ·V(s_{t+n})
Vantagem:             A_t = G_t - V(s_t)
Atualização do Ator:  θ_π ← θ_π - α ∇ L_actor
Atualização do Crítico: θ_V ← θ_V - α ∇ L_critic
```

---

## Conclusões Chave

| Conceito | Linguagem Simples |
|----------|-------------------|
| **Ator** | A política — decide o que fazer |
| **Crítico** | A função de valor — julga quão boa é a situação |
| **Vantagem** | "Isso foi melhor do que o esperado?" (real - esperado) |
| **Retorno n-passos** | Olha n passos no futuro antes de fazer o bootstrapping com V(s) |
| **Ambientes paralelos** | Múltiplos ambientes para experiências diversas e descorrelacionadas |
| **Bono de entropia** | Encoraja o ator a continuar tentando coisas novas |

---

## O Que Vem a Seguir?

O A2C é ótimo, mas tem uma fraqueza principal: às vezes ele atualiza a política de forma agressiva demais. Uma única atualização ruim pode destruir todo o bom aprendizado de atualizações anteriores.

O **PPO (Proximal Policy Optimization)** corrige isso com um "clipe de segurança" inteligente que impede que qualquer atualização única altere demais a política.

Veja `ppo_scratch.py` para a implementação do PPO!
