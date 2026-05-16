# REINFORCE con Línea de Base (Baseline): Cortando el ruido

## El Problema con REINFORCE Simple

Imagina que eres un estudiante intentando decidir si tu respuesta en un examen fue buena.

**Retroalimentación mala**: "¡Has sacado 7 puntos!".

¿Es 7 bueno? Si el máximo es 10, ¡sí! Si todos los demás sacaron 9, ¡no! Sin contexto, no puedes saber si deberías cambiar tu estilo de respuesta.

Este es exactamente el problema con REINFORCE: utiliza **retornos brutos** (G_t) para evaluar las acciones. Una puntuación de retorno total de 200 puntos puede ser increíble o terrible según la situación.

---

## Entra en escena la Línea de Base (Baseline)

Una **línea de base** b(s) es un punto de referencia: "¿Qué recompensa **espero** en esta situación?".

En lugar de preguntar "¿Fue buena esta acción?", preguntamos:

> **"¿Fue esta acción mejor de lo que normalmente esperaría?"**

```
Señal antigua: actualización ∝ G_t
Señal nueva:    actualización ∝ (G_t - b(s_t))
```

**Ejemplo de la vida real:** Has sacado un 85 en un examen de matemáticas.
- Si la media de la clase es 60 → tu respuesta estuvo **25 puntos por encima de la media** → ¡genial!
- Si la media de la clase es 90 → tu respuesta estuvo **5 puntos por debajo de la media** → ¡necesitas mejorar!

La **ventaja** (G_t - b(s)) es positiva cuando lo has hecho mejor de lo esperado y negativa cuando lo has hecho peor. ¡Esta es una señal de aprendizaje mucho más limpia!

---

## ¿Qué es la Línea de Base?

La línea de base natural es la **función de valor V(s)**:

> V(s) = "Recompensa total esperada si estoy en el estado s y juego mi política actual"

Aprendemos esto con una **Red de Valor** separada (también llamada red de línea de base o crítico):

```
Estado  →  [128 neuronas]  →  [128 neuronas]  →  V(s)   (un solo número)
```

Para cada estado que visita el agente, V(s) predice el retorno esperado. Si el retorno real G_t es mayor que V(s), ¡la acción fue mejor de lo esperado!

---

## Dos Redes Aprendiendo Juntas

```
Ocurre el episodio
     ↓
Calcular los retornos reales G_t
     ↓
         ┌─────────────────────────────┐
         │ Ventaja = G_t - V(s_t)      │
         │  +: la acción fue mejor     │
         │  -: la acción fue peor      │
         └─────────────────────────────┘
              ↓                  ↓
    Actualizar Red de Política   Actualizar Red de Valor
    (hacer las acciones buenas   (hacer las predicciones más
     más/menos probables)         precisas la próxima vez)
```

**Ejemplo de la vida real:** Dos amigos van juntos a un restaurante.

- Amigo 1 (Red de Valor): "Predigo que este plato será un 7/10".
- Amigo 2 (Red de Política): Pruebas el plato y lo puntúas con un 9/10.
- Ventaja = 9 - 7 = +2 → "¡Fue mejor de lo esperado! ¡Pídelo otra vez!".

En la próxima visita, el Amigo 1 actualiza su predicción acercándola al 9/10. El Amigo 2 es más propenso a pedir ese plato la próxima vez.

---

## ¿Por qué reduce esto la varianza?

**Prueba matemática (intuición):**

Sin línea de base: `gradiente ∝ ∇log π(a|s) × G_t`

Los valores de G_t varían mucho de un episodio a otro:
```
Episodio 1: G = [45, 44, 43, ..., 1]   (partida media)
Episodio 2: G = [500, 499, ..., 1]     (¡partida genial!)
Episodio 3: G = [12, 11, ..., 1]       (partida terrible)
```

Las estimaciones del gradiente saltan salvajemente porque G_t es grande y ruidoso.

Con línea de base: `gradiente ∝ ∇log π(a|s) × (G_t - V(s_t))`

La ventaja (G_t - V(s_t)) es mucho más pequeña y está centrada cerca de cero:
```
Episodio 1: ventaja ≈ [-2, +1, -3, ..., 0]   (pequeña, centrada)
Episodio 2: ventaja ≈ [+10, +8, ..., +3]     (esta partida FUE genial)
Episodio 3: ventaja ≈ [-5, -6, ..., -2]      (esta partida FUE mala)
```

**Ejemplo de la vida real:** Medir tu velocidad al correr.
- Sin línea de base: "He corrido a 8 km/h" (sin sentido sin contexto).
- Con línea de base: "He corrido 2 km/h MÁS RÁPIDO que mi media" (¡claramente bueno!).

La ventaja es siempre una comparación — es naturalmente más pequeña y estable.

---

## Crucialmente: ¡Sin Sesgo!

La línea de base no cambia QUÉ aprende el algoritmo — solo CÓMO DE RÁPIDO y de ESTABLE lo hace.

**¿Por qué?** Porque la ventaja esperada es siempre 0 en esperanza:

> E[G_t - V(s_t)] = E[G_t] - V(s_t) = V(s_t) - V(s_t) = 0

¡Cualquier b(s) que no dependa de la acción funciona como una línea de base válida!

**Ejemplo de la vida real:** Calificar con una curva no cambia quién obtuvo el mejor resultado — solo hace que las puntuaciones sean más fáciles de interpretar. El ranking se mantiene igual; solo cambia la escala.

---

## Los Resultados

```
Sin línea de base  — Media final 100-ep: 500.0, var grad: 599.3
Con línea de base — Media final 100-ep: 491.4, var grad: 578.8
```

Ambos métodos alcanzan un rendimiento casi perfecto en CartPole, pero fíjate:
1. La **varianza del gradiente** es medible (el gráfico de la derecha muestra la varianza durante el entrenamiento).
2. Con la línea de base, el agente alcanza un alto rendimiento **de forma más fiable** — hay menos caídas hacia recompensas bajas durante el entrenamiento.

La reducción de la varianza es más dramática en entornos más difíciles (LunarLander, MuJoCo).

---

## Ecuaciones Clave

```
Valor línea base:  V(s) ← V(s) + α(G_t - V(s))   [minimizar MSE]
Gradiente política: θ ← θ + α ∇log π(a_t|s_t) · (G_t - V(s_t))
Ventaja:           A_t = G_t - V(s_t)
```

---

## Conclusiones clave

| Concepto | En lenguaje sencillo |
|----------|---------------|
| **Línea de base b(s)** | Recompensa esperada en el estado s — nuestro punto de referencia. |
| **Ventaja A_t** | "¿Fue esta acción mejor de lo esperado?". |
| **Red de Valor** | Una red neuronal que aprende a predecir los retornos esperados. |
| **Reducción de varianza** | Menos ruido en las estimaciones del gradiente → aprendizaje más estable. |
| **Insesgado** | La línea de base no cambia la política objetivo en promedio; solo hace que la señal de aprendizaje sea menos ruidosa y más estable. |

---

## ¿Qué sigue?

La línea de base es en realidad el comienzo de algo mucho más potente: los métodos **Actor-Critic**.

En lugar de calcular V(s) solo al final de un episodio, el Actor-Critic actualiza V(s) en cada paso utilizando el aprendizaje por **Diferencia Temporal** (Temporal Difference). ¡Esto hace que las actualizaciones sean mucho más rápidas y permite al agente aprender de episodios incompletos!

Consulta `a2c_lunarlander.py` para ver la implementación completa de Actor-Critic.
