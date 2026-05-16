# Agente de Q-Learning para Frozen Lake 🧊

## ¿Qué es esto?

Imagina un estanque congelado con hielo resbaladizo. Hay un **cuadrado de Inicio** (Start) y un **cuadrado de Meta** (Goal) con algunos **Agujeros** (Holes) en medio. ¡Si te caes en un agujero, vuelves a empezar!

El hielo es resbaladizo, así que aunque intentes caminar hacia la derecha, podrías deslizarte hacia arriba o hacia abajo en su lugar. ¡Un **agente de Q-Learning** es un robot que aprende — probando una y otra vez — cómo llegar del Inicio a la Meta sin caerse!

---

## ¿Qué significa la "Q" en Q-Learning?

La **"Q"** viene de **"Quality"** (Calidad) — específicamente, la *calidad* de tomar una acción determinada en una situación particular.

Piénsalo como la calificación de un restaurante: "¿Qué tan bueno (calidad) es pedir la pizza en ESTE restaurante?". Q(s, a) pregunta: "¿Qué tan bueno es realizar la acción **a** cuando estoy en el estado **s**?".

Un valor Q alto significa: "¡Excelente elección! Esta acción conduce a una gran recompensa".
Un valor Q bajo significa: "¡Mala idea! Esta acción suele traer problemas".

**Ejemplo de la vida real:** Imagina que eres un niño decidiendo si comer caramelos antes de cenar. Tu valor Q para "comer caramelos ahora" podría ser alto en este momento (¡saben genial!) pero bajo en general (mamá se enfada, te sientes mal después). El Q-learning aprende a tener en cuenta esas consecuencias futuras, ¡no solo la sensación inmediata!

---

## La Gran Idea: Una Tabla Mágica de Puntuaciones

El Q-Learning construye una gran tabla llamada **tabla Q** (Q-table). Cada fila es un cuadrado en el hielo y cada columna es una acción (izquierda, derecha, arriba, abajo). Los números de dentro son **puntuaciones**: "¿Qué tan bueno es realizar esta acción desde este cuadrado?".

Cada vez que el robot intenta un movimiento:
1. Recibe retroalimentación (¿se cayó? ¿llegó a la meta?).
2. Actualiza la puntuación en la tabla usando esta fórmula:

> **Nueva Puntuación = Puntuación Antigua + Tasa de Aprendizaje × (Lo que realmente pasó − Lo que esperaba)**

Básicamente, el robot se pregunta: "¿Fue este movimiento mejor o peor de lo que pensaba?".

**Ejemplo de la vida real:** Piensa en un bebé aprendiendo a caminar. Cada vez que intentan dar un paso y se caen, aprenden "ese paso fue malo". Cada vez que tienen éxito, recuerdan "¡eso funcionó!". Después de muchos intentos, descubren cómo caminar. ¡El Q-learning hace lo mismo, pero con una tabla!

---

## Qué hace especial al Q-Learning: ¡Es fuera de la política (Off-Policy)!

Aquí hay algo ingenioso: cuando el Q-Learning construye su tabla, *siempre asume que hará el movimiento perfecto la próxima vez*, incluso si durante el entrenamiento a veces explora movimientos aleatorios.

Esto hace que el Q-Learning sea **fuera de la política** (off-policy): la estrategia que *aprende* (elegir siempre la mejor acción conocida) es independiente de la estrategia que *sigue* durante el entrenamiento (a veces elegir una acción al azar para explorar). Concretamente, la actualización de la tabla Q utiliza el valor Q *máximo* del siguiente estado — el mejor teórico — incluso cuando el siguiente movimiento real del robot sea aleatorio.

En términos sencillos: el robot puede vagar aleatoriamente a la izquierda para explorar, pero su aprendizaje sigue calculando como si fuera a tomar la *mejor* acción a continuación. Esta separación permite que el Q-Learning converja a la estrategia óptima independientemente de cuánto explore.

---

## Qué encontró nuestro código

Entrenamos durante **50,000 episodios** en el Frozen Lake 4×4 resbaladizo:

| Métrica | Resultado |
|--------|--------|
| Tasa de éxito de la evaluación codiciosa (greedy) | **73.1%** |
| Objetivo del hito (>70%) | ✓ **SUPERADO** |

¡El hielo es muy resbaladizo, así que incluso la mejor política no puede ganar el 100% de las veces!

La tabla Q aprendida muestra que el agente descubrió: ir hacia abajo y a la derecha evitando los agujeros.

---

## Ejemplos de la vida real

- **Coche autónomo**: Aprender qué carriles tomar en las intersecciones a través de trayectos de prueba.
- **Sistemas de recomendación**: Aprender qué películas sugerir en función de si a los usuarios les gustaron las sugerencias anteriores.
- **IA de videojuegos**: Un personaje que aprende a navegar por un laberinto probando muchos caminos.

---

## Palabras Clave para Recordar

- **Tabla Q (Q-table)**: La tabla de "qué tan buena es cada acción en cada estado".
- **Q(s, a)**: La puntuación por realizar la acción a en el estado s.
- **Recompensa (Reward)**: Lo que el agente obtiene tras realizar una acción (+1 por llegar a la meta, 0 de lo contrario).
- **Fuera de la política (Off-policy)**: Aprende la estrategia óptima incluso mientras explora al azar.
- **ε-greedy** (ε = épsilon): La mayoría de las veces hace la mejor acción conocida; a veces explora al azar.
- **Factor de descuento γ** (γ = gamma): Cuánto valen las recompensas futuras (como preferir el dinero ahora que más tarde).

La gran idea: **El Q-Learning construye una "guía de trucos" para cada situación y sigue mejorándola hasta que conoce el mejor movimiento en todas partes.**
