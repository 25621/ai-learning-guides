# Q-Learning Lineal para CartPole 🎪

## ¿Qué es CartPole?

Imagina un palo de escoba equilibrado verticalmente sobre tu dedo. Si mueves el dedo un poco a la izquierda o a la derecha, puedes evitar que el palo se caiga. ¡Eso es **CartPole**!

Un pequeño robot se sienta en un carro (una caja con ruedas) y tiene un poste en la parte superior. El robot solo puede empujar el carro hacia la **izquierda** o hacia la **derecha**. Tiene que aprender a mantener el poste equilibrado el mayor tiempo posible — ¡igual que tú equilibrando una escoba!

El robot puede ver 4 cosas sobre el mundo:
1. Dónde está el carro.
2. A qué velocidad se mueve el carro.
3. Cuánto se inclina el poste.
4. A qué velocidad se inclina el poste.

---

## El Gran Problema: ¡Demasiados Estados!

¿Recuerdas el Q-learning de la Fase 2? Utilizaba una gran tabla para recordar qué tan buena es cada acción en cada situación (estado). Eso funcionó de maravilla para Frozen Lake — solo había 16 cuadrados en el hielo.

¡Pero CartPole es diferente! El carro puede estar en **cualquier posición**, moviéndose a **cualquier velocidad**, con el poste en **cualquier ángulo**. Básicamente, ¡hay **infinitos estados posibles**! No podemos hacer una tabla con infinitas filas. ¡Necesitaríamos un cuaderno del tamaño del universo!

**Ejemplo de la vida real:** Imagina que estás aprendiendo a montar en bicicleta. No puedes memorizar cada posible balanceo — ¡hay demasiados! En su lugar, aprendes una **regla**: "cuando me incline a la izquierda, empujo a la derecha; cuando me incline a la derecha, empujo a la izquierda". Una regla simple funciona para TODOS los balanceos.

---

## La Solución: Una Fórmula Mágica

La **aproximación de funciones lineales** sustituye la tabla gigante por una **fórmula diminuta**:

> **Puntuación(situación, acción) = w₁ × posición_carro + w₂ × velocidad_carro + w₃ × ángulo_poste + w₄ × velocidad_punta_poste**

- Los números `w` se llaman **pesos** (weights) — son como perillas que puedes girar.
- Aprendemos **pesos diferentes para cada acción** (empujar a la izquierda y empujar a la derecha).
- La fórmula da una puntuación sobre qué tan buena es cada acción en este momento.

**Ejemplo de la vida real:** Piensa en una receta sencilla: "1 taza de harina + 2 huevos + ½ taza de mantequilla". Los pesos (1, 2, ½) te dicen cuánto importa cada ingrediente. ¡Estamos aprendiendo la receta para tomar buenas decisiones!

---

## ¿Cómo aprende?

El robot prueba cosas, recibe retroalimentación y ajusta los pesos:

1. **El robot empuja el carro** (elige la acción con la puntuación más alta).
2. **Ocurre la física** (el poste se inclina un poco, el carro se mueve).
3. **El robot recibe una recompensa** (+1 por cada paso que el poste se mantiene en pie, 0 si se cae).
4. **El robot pregunta**: "¿Fue el resultado real mejor o peor de lo que predije?".
5. **El robot ajusta los pesos** para estar más cerca de la realidad la próxima vez.

Esto es la **Actualización TD de Semi-Gradiente** (Semi-Gradient TD Update) — un nombre sofisticado para "ajustar la receta un poco basándose en la sorpresa".

> **Nuevo peso = Peso antiguo + Tasa de aprendizaje × (Lo que realmente pasó − Lo que predije) × Característica**

---

## Qué encontró nuestro código

Cuando ejecutes `linear_q_cartpole.py`, el robot:

- Comienza siendo terrible (el poste se cae en 10–30 pasos).
- Aprende gradualmente los pesos adecuados a lo largo de 3,000 intentos.
- ¡Al final mantiene el poste equilibrado durante 100–400+ pasos!

El gráfico muestra la **curva de aprendizaje** — cómo mejora la puntuación con el tiempo. Será irregular (¡el aprendizaje nunca es fluido!), pero la tendencia debería ser ascendente.

---

## Por qué esto es genial (¡y limitado!)

**Genial:** ¡Una fórmula diminuta con solo 8 números (4 pesos × 2 acciones) puede equilibrar un poste! No se necesita una tabla gigante.

**Limitado:** La fórmula es demasiado simple para tareas complejas. Supone que los números más grandes siempre significan efectos más grandes (lo cual no siempre es cierto). Para juegos más difíciles como Atari, necesitamos **redes neuronales**, ¡que es lo que hace DQN!

---

## Vocabulario Clave

| Palabra | Significado |
|------|---------|
| **Característica (Feature)** | Una cosa medible sobre el mundo (ej. ángulo del poste) |
| **Peso (Weight)** | Cuánto afecta una característica a la decisión |
| **Lineal** | La fórmula es solo multiplicación y suma (sin curvas complicadas) |
| **Semi-gradiente** | Actualizar los pesos siguiendo la dirección de menor error |
| **Aproximación de funciones** | Usar una fórmula en lugar de una tabla |

---

## ¿Qué sigue?

La aproximación lineal es como usar una regla recta para dibujar una curva — funciona bien para formas simples pero no para las complejas. Para juegos de Atari con millones de situaciones posibles, necesitamos **Redes Q Profundas (DQN)** — redes neuronales que pueden aprender patrones mucho más complejos. ¡Eso está en el siguiente archivo!
