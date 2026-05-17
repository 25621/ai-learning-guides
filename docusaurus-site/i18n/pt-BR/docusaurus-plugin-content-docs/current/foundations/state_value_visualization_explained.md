# Funções de Valor de Estado 🗺️

## O que é um "Estado"?

Pense em jogar um jogo de tabuleiro. A qualquer momento, você está em *um*
quadrado do tabuleiro. Esse quadrado é o seu **estado** — é onde você está
agora.

No nosso jogo de grade 4×4, existem 16 quadrados (estados). Cada quadrado é um lugar
onde o agente pode estar.

---

## O que é um "Valor"?

Aqui está a pergunta mágica: **"Se eu estiver neste quadrado agora,
quanto tesouro posso esperar coletar antes que o jogo termine?"**

Essa resposta é o **valor** daquele estado!

Um quadrado com um **valor alto** significa: "Este é um ótimo lugar — eu provavelmente
coletarei muito tesouro a partir daqui!"

Um quadrado com um **valor baixo** significa: "Ih — a partir daqui, as coisas costumam
ir mal."

**Exemplo da vida real:** Imagine que você está brincando de esconde-esconde. Se você está escondido
atrás de uma árvore grande (um ótimo lugar), sua chance de vencer é alta — esse é um
estado de alto valor! Se você está escondido no meio de uma sala vazia, provavelmente
será encontrado — esse é um estado de baixo valor.

---

## Nosso Mundo de Grade (Grid World)

Aqui está o tabuleiro que usamos:

```
I  .  .  .      I = Início (Start)
.  B  .  B      B = Buraco (recompensa -1, o jogo termina)
.  .  .  B      O = Objetivo (recompensa +1, o jogo termina)
B  .  .  O      . = Quadrado vazio e seguro
```

- Se você chegar ao **O** (Objetivo): você ganha **+1 ponto** 🎉
- Se você cair no **B** (Buraco): você ganha **-1 ponto** 😢
- Outros passos: **0 pontos**

Usamos γ (gama) = 0,99, o que significa que recompensas futuras contam quase tanto
quanto as imediatas. (Um doce amanhã é quase tão bom quanto um doce hoje!)

---

## Dois Planos Diferentes (Políticas)

Testamos duas políticas e computamos o valor de cada quadrado para cada uma:

### Política 1: Aleatória Uniforme
Escolhe aleatoriamente para cima, para baixo, esquerda ou direita com chances iguais.

```
Valores (Política Aleatória Uniforme):
-0,912  -0,932  -0,912  -0,942
-0,929   (B)   -0,898   (B)
-0,901  -0,801  -0,696   (B)
 (B)   -0,630  -0,104   (O)
```

Quase todos os lugares são **negativos** — a política aleatória cai em buracos com tanta
frequência que estar em qualquer lugar é muito ruim!

---

### Política 2: Enviesada para o Objetivo
Prefere mover-se para a direita e para baixo (em direção ao objetivo), mas ainda às vezes
vai para outras direções.

```
Valores (Política Enviesada para o Objetivo):
-0,838  -0,895  -0,814  -0,961
-0,798   (B)   -0,665   (B)
-0,595  -0,143  -0,213   (B)
 (B)    0,254   0,673   (O)
```

Agora os quadrados perto do **Objetivo** têm **valores positivos** (0,254 e 0,673)!
A política inteligente torna esses quadrados bons lugares para se estar.

---

## O que as Cores em nossa Imagem Significam

Na nossa visualização:
- **Quadrados verdes** = alto valor (ótimos lugares para estar)
- **Quadrados vermelhos** = baixo valor (evite-os!)
- **Quadrados amarelos** = algo entre os dois

Você pode ver o **gradiente** — os valores ficam mais verdes conforme você se move para o Objetivo
e mais vermelhos perto dos Buracos.

---

## Por que nos Importamos com os Valores?

Os valores são a *base* do aprendizado por reforço! Uma vez que você conhece o valor
de cada estado, pode tomar decisões inteligentes:

> "Estou no quadrado A. Posso ir para o quadrado B (valor = 0,5) ou para o quadrado C (valor = -0,3).
> Vou para o B — ele tem um valor maior!"

É exatamente assim que muitos algoritmos de RL (como o Q-learning) aprendem a tomar boas
decisões sem que as regras lhes sejam ditas.

**Exemplo da vida real:** Imagine que você está escolhendo em qual fila ficar no
supermercado. Cada fila é um "estado". O valor desse estado é a rapidez com que você
sairá do caixa. Você olha para as filas (observa os estados) e escolhe a que tem o
valor mais alto (menor espera + menos itens).

---

## Como Computamos os Valores

Usamos a **Avaliação de Política Iterativa**, que funciona assim:

1. Início: chuta que todos os valores são 0.
2. Atualização: para cada quadrado, calcula qual o valor *deveria* ser baseado em
   para onde a política o levará em seguida.
3. Repete até que os valores parem de mudar (converjam).

Matematicamente: **V(s) = Σ_a π(a|s) × [R(s,a) + γ × V(proximo_estado)]**

Em português simples: "O valor deste quadrado = a recompensa média que receberei
agora + um pouco do valor de onde quer que eu termine."

---

## Palavras-Chave para Lembrar

- **Estado**: Onde você está agora (um quadrado no tabuleiro)
- **Valor V(s)**: Recompensa total esperada começando do estado s
- **Política**: Seu plano para o que fazer em cada estado
- **Fator de desconto γ**: O quanto você se importa com recompensas futuras (0,99 = muito!)
- **Avaliação de Política**: Calcular valores para cada estado sob uma dada política

A grande ideia: **Alguns lugares são melhores do que outros — e a função de valor
lhe diz exatamente o quão bom cada lugar é!**
