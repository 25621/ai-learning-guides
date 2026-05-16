# Frozen Lake con una Política Aleatoria 🧊

## ¿Qué es Frozen Lake?

Imagina que estás jugando en un **estanque congelado** con tus amigos.

El hielo es seguro en su mayor parte, pero algunos puntos tienen **agujeros** — ¡si pisas un agujero, te caes y el juego termina! En un extremo del estanque hay un **regalo** 🎁. Tu trabajo es deslizarte desde el **inicio** hasta el **regalo** sin caerte.

Así es como se ve el lago congelado (4 cuadrados × 4 cuadrados):

```
S  F  F  F
F  H  F  H
F  F  F  H
H  F  F  G
```

- **S** = Start (donde empiezas)
- **F** = Frozen ice (¡hielo congelado, seguro!)
- **H** = Hole (agujero — caerse, fin del juego 😨)
- **G** = Goal (meta — ¡el regalo! 🎁)

---

## La parte difícil: ¡Hielo resbaladizo!

En un estanque congelado real, cuando intentas caminar hacia la *derecha*, ¡a veces el hielo te hace deslizarte hacia *arriba* o hacia *abajo* en su lugar! Eso es lo que lo hace difícil.

Incluso si *quieres* ir a la derecha, el juego podría deslizarte a otro lugar. Esto se llama **estocasticidad** — una palabra elegante para "las cosas no siempre salen como planeaste".

---

## ¿Qué es una Política Aleatoria?

Una **política** es simplemente un plan: "En esta situación, haré ESTA acción".

Una **política aleatoria** significa: "¡No tengo ningún plan! Simplemente elegiré una dirección al azar cada vez — arriba, abajo, izquierda o derecha — ¡como girar una ruleta!".

Es como un bebé caminando sobre el hielo sin tener ni idea de dónde está el regalo.

---

## Qué encontró nuestro código

Probamos la política aleatoria durante **1,000 partidas**:

| Resultado | Valor |
|--------|-------|
| **Veces que se llegó al regalo** | 11 de 1,000 (1.1%) |
| **Promedio de pasos por partida** | 7.5 pasos |
| **Partida más rápida** | 2 pasos |
| **Partida más larga** | 33 pasos |

La mayoría de las veces, el caminante aleatorio cayó en un agujero rápidamente. ¡Solo 1 de cada 100 partidas terminó encontrando el regalo!

---

## ¿Por qué es útil esto?

Aunque la política aleatoria es terrible, nos da una **línea de base** (baseline) — un punto de partida para comparar.

Cuando más adelante construyamos una política *inteligente* (usando Q-learning u otros algoritmos), podremos decir: "Nuestro agente inteligente tiene éxito el 75% de las veces — ¡mucho mejor que el 1% del caminante aleatorio!".

**Ejemplo de la vida real:** Imagina intentar encontrar tu clase en una escuela nueva girando al azar a la izquierda o a la derecha en cada pasillo. Podrías llegar eventualmente, ¡pero te llevaría mucho tiempo! Una política inteligente es como tener un mapa.

---

## Qué muestra el mapa de calor (Heatmap)

En nuestra imagen, el **mapa de calor** muestra qué cuadrados visitó el caminante aleatorio con más frecuencia:

- El cuadrado de **Inicio** (Start) se visita mucho (cada partida comienza allí).
- Los cuadrados cerca de los **agujeros** se visitan menos (el caminante a menudo se cae antes de llegar a ellos).
- La **Meta** (Goal) se visita muy raramente porque el caminante aleatorio casi nunca llega allí.

Esto nos dice algo importante: la política aleatoria se queda atascada cerca del principio y nunca explora realmente todo el lago.

---

## Palabras Clave para Recordar

- **Política**: Tu plan sobre qué hacer en cada situación.
- **Política aleatoria**: Sin plan — ¡simplemente elige una acción al azar!
- **Línea de base (Baseline)**: Un mal resultado que usamos para comparar (¿cuánto mejor podemos hacerlo?).
- **Estocástico**: Las cosas no siempre salen como planeaste (¡como el hielo resbaladizo!).
- **Tasa de éxito**: ¿Con qué frecuencia ganamos? (Aquí: 1.1% — ¡muy baja!).

La gran idea: **Una política aleatoria es un punto de partida. ¡El aprendizaje real significa construir un plan mejor!**
