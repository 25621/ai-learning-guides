# SARSA para Cliff Walking 🏔️

## O que é?

Imagine um **corredor muito longo** com um **penhasco terrível** ao longo de uma das bordas. Se você cair no
penhasco, terá que voltar todo o caminho até o início! Seu objetivo é caminhar de uma extremidade à
outra o mais rápido possível, sem cair.

**SARSA** é um robô que aprende a caminhar por este corredor através da prática. Ele aprende a seguir um
*caminho seguro* que evita o penhasco — mesmo que seja um pouco mais longo — porque sabe que pode
escorregar acidentalmente para perto da borda ao explorar!

---

## A Grande Ideia: Aprender com o que você realmente faz

SARSA significa: **S**tate (Estado) → **A**ction (Ação) → **R**eward (Recompensa) → **S**tate (Estado) → **A**ction (Ação)

Estas são as cinco informações que o SARSA usa para aprender:

1. **S** — Onde estou agora? (estado atual)
2. **A** — Qual ação eu realmente tomei?
3. **R** — Qual recompensa eu recebi?
4. **S** — Onde eu fui parar?
5. **A** — Qual ação eu *realmente vou tomar em seguida*?

O último "A" é o que torna o SARSA especial! Ele se atualiza usando a ação que ele *realmente tomará
em seguida* (mesmo que seja um movimento exploratório aleatório), não a ação ideal perfeita.

**Exemplo da vida real:** Pense em aprender a andar de bicicleta. Se você sabe que às vezes balança
aleatoriamente (exploração), você fica um pouco mais longe dos carros estacionados — porque sabe que seu
eu "balançante" pode desviar! O SARSA faz isso: ele aprende um caminho seguro porque leva em conta
seus próprios erros aleatórios.

---

## O Mapa do Cliff Walking (Caminhada no Penhasco)

```
[ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]
[ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]
[ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]
[S][C][C][C][C][C][C][C][C][C][C][G]
   ← ← ← ← PENHASCO ← ← ← ← ←
```

- **S** = Start/Início (canto inferior esquerdo)
- **G** = Goal/Objetivo (canto inferior direito)
- **C** = Cliff/Penhasco — pisar aqui = recompensa de -100, reiniciar!
- Cada outro passo = recompensa de -1

---

## O que Nosso Código Encontrou

Após treinar o SARSA por 500 episódios:

| Resultado | Valor |
|-----------|-------|
| Recompensa média nos últimos 50 episódios | **-21,6** |
| Recompensa do caminho ótimo (arriscado) | -13 |

A política aprendida pelo SARSA segue **pelo topo da grade** — um desvio seguro! Custa alguns
passos extras (-21 em vez de -13), mas quase nunca cai no penhasco durante o treinamento.

---

## Exemplos da Vida Real

- **Enfermeira administrando medicamentos**: Segue o protocolo seguro comprovado (caminho seguro) mesmo que exista um
  método ligeiramente mais rápido, porque pequenos erros (exploração) podem ser perigosos.
- **Pilotos de avião**: Seguem listas de verificação rigorosas (caminhos seguros) mesmo quando atalhos podem parecer
  mais rápidos, levando em conta o erro humano.
- **Aprendendo a cozinhar**: Comece com receitas bem testadas (seguras), não atalhos arriscados.

---

## Palavras-Chave para Lembrar

- **On-policy**: Aprende sobre a política que está realmente usando (incluindo seus erros aleatórios)
- **Atualização SARSA**: Usa a próxima ação *real*, não a teoricamente melhor
- **Caminho seguro**: Um caminho mais longo que evita o perigo, levando em conta erros de exploração
- **Controle TD (Temporal Difference)**: Atualizando valores após cada passo (sem esperar pelo episódio inteiro)

A grande ideia: **O SARSA é honesto — ele aprende com o que realmente faz, não com o que gostaria de
fazer. Isso o torna cauteloso e seguro perto do perigo!**
