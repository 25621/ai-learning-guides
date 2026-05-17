# PPO: Atualizações de Política Seguras e Estáveis

## O Problema com o A2C

Imagine que você está aprendendo a equilibrar uma vassoura no dedo. Após semanas de prática,
você consegue mantê-la por 30 segundos!

Agora seu instrutor lhe dá um conselho: "Incline o pulso um pouco mais para a esquerda."

**Bom conselho → mudança cuidadosa → você ainda equilibra por 30 segundos ✓**

Mas e se o instrutor reagisse de forma exagerada e dissesse: "INCLINE MUITO PARA A ESQUERDA IMEDIATAMENTE!"
Você corrige demais → a vassoura cai → você perdeu semanas de progresso.

Este é o problema do A2C: **grandes atualizações de gradiente podem destruir uma boa política**.

O **PPO (Proximal Policy Optimization)** é um sistema de segurança que evita isso.

---

## A Ideia Central: Mantenha-se Próximo do que Estava Funcionando

A restrição fundamental do PPO:

> **"Não mude a política demais em uma única atualização."**

Antes de uma atualização, temos a política "antiga" π_old.
Depois da atualização, temos a política "nova" π_new.

O PPO mede o quanto a política mudou com a **razão de probabilidade** (probability ratio):

```
r(θ) = π_new(a|s) / π_old(a|s)
```

- r = 1,0: política inalterada
- r = 1,5: a nova política tem 50% mais chance de tomar aquela ação
- r = 0,5: a nova política tem 50% menos chance de tomar aquela ação

**Exemplo da vida real:** Você é um chef ajustando uma receita.
- r = 1,0: mesma quantidade de sal de antes
- r = 2,0: o dobro de sal — extremo demais!
- r = 0,9: 10% menos sal — mudança pequena e segura

---

## O Truque do Corte (Clipping) {#the-clipping-trick}

O PPO corta (clip) a razão para que ela permaneça dentro de [1-ε, 1+ε] (geralmente ε = 0,2):

```
L_CLIP = E[min(r(θ) · A,  clip(r(θ), 1-ε, 1+ε) · A)]
```

Vamos detalhar isso:

**Caso 1: A ação foi BOA (A > 0)**

Queremos fazer essa ação mais vezes (r > 1). Mas limitamos o quanto aumentamos:
```
se r > 1,2: corta para 1,2; não há mais incentivo para aumentar
```
Isso nos impede de oscilar DEMAIS em uma direção.

**Caso 2: A ação foi RUIM (A < 0)**

Queremos fazer essa ação menos vezes (r < 1). Mas, novamente, limitamos:
```
se r < 0,8: corta para 0,8; não há mais penalidade por diminuir ainda mais
```

**Visual:**
```
ε = 0,2; portanto, a janela de razão segura é de 0,8 a 1,2.

Ação BOA (A > 0): aumenta a probabilidade da ação, mas para de recompensar após 1,2
razão r:       0,6      0,8      1,0      1,2      1,4
incentivo:      ↑        ↑        ↑        ↑        -
significado: muito baixo   ok     antigo   máx     cortado

Ação RUIM (A < 0): diminui a probabilidade da ação, mas para de penalizar abaixo de 0,8
razão r:       0,6      0,8      1,0      1,2      1,4
incentivo:      -        ↓        ↓        ↓        ↓
significado:  cortado   máx     antigo     ok    muito alto
```

O `-` marca a região plana cortada. Nessa região, tornar a razão de probabilidade ainda
mais extrema não melhora o objetivo, então o PPO não tem incentivo extra para forçar além disso.

**Exemplo da vida real:** O limitador de velocidade de um carro. Você pode acelerar, mas ao atingir 120 km/h,
o limitador entra em ação e não deixa você ir mais rápido. Ele o mantém seguro sem impedir o movimento.

---

## Por que isso Evita Atualizações Catastróficas

Uma **atualização catastrófica** ocorre quando uma grande mudança na política destrói completamente tudo o que o
agente aprendeu — horas de treinamento perdidas em um único passo de gradiente.

Sem o corte: um grande passo de gradiente pode mudar drasticamente a política.
Com o corte: o gradiente é zero fora de [1-ε, 1+ε], então a política só pode se mover um pouco por passo.

**Exemplo da vida real:** Um bom cirurgião faz cortes pequenos e precisos — não cortes grandes e amplos.
O PPO é o "cirurgião cuidadoso" dos otimizadores de RL.

---

## GAE: Estimativas de Vantagem Mais Inteligentes {#gae-smarter-advantage-estimates}

O PPO usa a **Estimativa de Vantagem Generalizada (GAE - Generalized Advantage Estimation)** para calcular a vantagem:

```
δ_t = r_t + γ · V(s_{t+1}) - V(s_t)          (erro TD)
A_t = δ_t + γλ · δ_{t+1} + (γλ)² · δ_{t+2} + ...
```

O GAE tem um parâmetro λ (lambda):
- λ = 0: usa apenas o erro TD de um passo (baixa variância, alto viés)
- λ = 1: usa os retornos completos de Monte Carlo (alta variância, baixo viés)
- λ = 0,95: um bom equilíbrio entre os dois!

**Exemplo da vida real:** Planejando uma viagem de carro.
- λ=0: olha apenas as próximas 5 milhas (seguro, mas pode perder um atalho logo à frente)
- λ=1: considera toda a jornada de 500 milhas (mais informações, mas muita incerteza)
- λ=0,95: olha longe, mas dá mais peso às estradas próximas ← melhor equilíbrio!

---

## Múltiplas Épocas: Reutilizando Dados de Forma Eficiente

Após coletar um lote de experiência (rollout), o REINFORCE o descarta após UMA atualização.

O PPO reutiliza cada lote por **K épocas** (geralmente de 4 a 10 passagens pelos mesmos dados):

```
Coletar 512 passos × 4 ambientes = 2048 transições
Época 1: 32 minibatches × atualizar cada um
Época 2: embaralhar, mais 32 minibatches × atualizar cada um
Época 3: ...
Época 4: ...
```

**O que é um "minibatch"?** Atualizar com todas as 2048 transições de uma vez é lento e
consome muita memória; atualizar uma transição por vez é ruidoso. Um **minibatch** é um pequeno
pedaço intermediário — aqui, 2048 ÷ 32 = **64 transições por minibatch**. Calculamos um
passo de gradiente por minibatch, então cada época realiza 32 atualizações pequenas e
estáveis em vez de uma enorme. (Esta é a mesma ideia de minibatch usada em todo o deep learning — veja
[descida de gradiente por mini-lotes](https://pt.wikipedia.org/wiki/Stochastic_gradient_descent#Mini-batch_gradient_descent)).

O corte garante que essas múltiplas passagens não ultrapassem o limite — sem o corte, múltiplas
épocas destruiriam a política ao empurrá-la longe demais!

**Exemplo da vida real:** Um aluno tem 30 problemas de prática.
- REINFORCE: faz cada problema uma vez, aprende um pouco e os descarta.
- PPO: faz cada problema 4 vezes (ângulos diferentes a cada vez), corta suas mudanças
  para não memorizar padrões errados.

---

## A Perda (Loss) Total do PPO

```
L = L_CLIP - c₁ · L_entropy + c₂ · L_critic

L_CLIP    = gradiente de política cortado
L_entropy = bônus de entropia (encoraja a exploração)  
L_critic  = MSE entre V(s) e os retornos
```

Coeficientes típicos: c₁ = 0,01 (entropia), c₂ = 0,5 (crítico)

**Dois termos que valem a pena detalhar:**

- **Gradiente de política** — a metade "ator" da perda. Ele usa o sinal do gradiente para
  empurrar a política em direção a ações com maior vantagem e para longe de ações com menor
  vantagem. Esta é a mesma ideia central introduzida no REINFORCE — veja o [passo a passo do
  REINFORCE](./reinforce_cartpole_explained.md#the-old-way-vs-the-new-way) para entender a
  intuição. O PPO apenas adiciona o invólucro de corte ao redor dele.
- **MSE (Erro Quadrático Médio)** — a metade "crítico" da perda. O crítico V(s) prevê
  o retorno esperado de um estado; comparamos sua previsão com o retorno real e
  elevamos a diferença ao quadrado: `MSE = média((V(s) - retorno)²)`. Elevar ao quadrado pune erros
  grandes mais do que pequenos e fornece um sinal suave e diferenciável para o treinamento. (Perda
  de regressão padrão — veja [erro quadrático médio](https://pt.wikipedia.org/wiki/Erro_quadr%C3%A1tico_m%C3%A9dio)).

---

## Os Resultados

```
Atualização  200 | Recompensa média: ~120
Atualização  400 | Recompensa média: ~280
Atualização  800 | Recompensa média: ~280-300
```

O PPO no CartPole mostra uma melhoria constante, mas tende a estacionar em torno de 280-300.
(Um **platô** ou estacionamento significa que a curva de aprendizado se achata — a recompensa para de melhorar mesmo
com a continuação do treinamento. A política encontrou uma estratégia localmente boa, mas não está progredindo mais.)
Isso é, na verdade, esperado — o PPO foi projetado para ambientes mais difíceis e com episódios mais longos.

Uma observação interessante: **o REINFORCE resolveu o CartPole mais rápido!** (média de 500 vs média de 300)

Por quê? Os episódios do CartPole são curtos (≤500 passos), então os retornos exatos do
REINFORCE são muito precisos. As estimativas por bootstrapping do PPO adicionam uma complexidade desnecessária. O PPO brilha de verdade em
ambientes onde esperar por episódios completos é impraticável (como o BipedalWalker).

**O que é o "BipedalWalker"?** O BipedalWalker (especificamente o `BipedalWalker-v3` no
[Gymnasium](https://gymnasium.farama.org/environments/box2d/bipedal_walker/)) é um
ambiente clássico de benchmark de RL: um robô de 2 pernas que deve aprender a caminhar para frente
em um terreno irregular sem cair. Ao contrário das duas ações discretas do CartPole
(ESQUERDA / DIREITA), o BipedalWalker tem **ações contínuas** — quatro valores de torque, um para
cada articulação da perna, cada um sendo um número real em [-1, 1]. Os episódios podem durar milhares de passos,
que é exatamente o regime onde a eficiência de dados e a estabilidade do PPO se pagam.

---

## Equações Principais

```
Razão:      r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
Perda clip: L_CLIP = E[min(r_t A_t, clip(r_t, 1-ε, 1+ε) · A_t)]
GAE:        A_t = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}
```

---

## Principais Conclusões

| Conceito | Em Português Simples |
|---------|---------------|
| **Razão r(θ)** | O quanto a política mudou nesta ação |
| **Corte (Clip) ε** | A fronteira de segurança — não mude a política mais do que isso |
| **GAE** | Uma forma inteligente de estimar vantagens olhando vários passos à frente |
| **Eficiência de dados** | Cada rollout é coletado de vários ambientes paralelos (experiência decorrelacionada e estável) e depois reutilizado por K épocas de atualizações de minibatch — o corte mantém essas repetições seguras |

---

## O que vem a seguir?

Até agora, todos os nossos ambientes tinham ações **discretas** (empurrar para a ESQUERDA ou DIREITA).

Robôs reais precisam controlar ações **contínuas** — como "aplicar exatamente 0,73 Newtons de força."

O arquivo `ppo_continuous.py` estende o PPO para ações contínuas usando uma **política Gaussiana**
e o testa no ambiente BipedalWalker-v3, que é muito mais difícil!
