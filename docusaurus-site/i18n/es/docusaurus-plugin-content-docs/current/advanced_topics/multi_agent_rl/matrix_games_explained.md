# Juegos Matriciales: El mundo multiagente más simple 🎲

## ¿Qué es un Juego Matricial?

Imagina que tú y un amigo eligen cada uno un signo con la mano — **piedra, papel o tijera** — *al mismo tiempo*. No ven la elección del otro. El ganador se decide mediante una pequeña tabla:

|          | Piedra | Papel | Tijera |
|----------|:----:|:-----:|:--------:|
| Piedra   |  0,0  | -1,+1 | +1,-1 |
| Papel    | +1,-1 |  0,0  | -1,+1 |
| Tijera   | -1,+1 | +1,-1 |  0,0  |

Esa tabla es *todo el mundo* del juego. Sin movimiento, sin tiempo, sin mapa. Solo una decisión única. Llamamos a esto un **juego matricial** porque la matriz de pagos (payoff matrix) es el entorno completo.

Los juegos matriciales son el lugar más limpio para estudiar el **RL multiagente**, porque lo único que puede cambiar durante el entrenamiento es la *política* de cada jugador — la probabilidad de elegir cada acción.

---

## Por qué es "Multiagente"

En el RL de un solo agente, el entorno es fijo: el viento siempre sopla de la misma manera, el suelo nunca se mueve. El agente mejora y finalmente gana.

En un juego matricial, tu "entorno" es *otro agente que aprende*. A medida que ellos se vuelven más inteligentes, lo que cuenta como un buen movimiento para ti *cambia*. Esto se llama **no estacionariedad** y es el problema central del RL multiagente.

> Si sigues jugando Piedra, tu oponente acabará jugando siempre Papel. Entonces tú cambias a Tijera. Entonces ellos cambian a Piedra. Entonces tú cambias a Papel... y así sucesivamente. El "mejor movimiento" nunca se queda quieto.

La solución clásica son las **estrategias mixtas**: no elijas ninguna acción de forma determinista, aleatoriza de forma que el oponente no pueda explotarte.

---

## Los Tres Juegos que Jugamos

### 1) Piedra-Papel-Tijera (suma cero)
- La ganancia de un jugador es la pérdida del otro.
- El **equilibrio de Nash** es: cada jugador elige cada acción con una probabilidad de ⅓. Cualquier desviación es explotable.
- Esperamos que nuestros dos aprendices Q oscilen alrededor de ⅓-⅓-⅓ — nunca de forma perfectamente estable, porque cada vez que uno se desvía, el otro reacciona.

### 2) Dilema del Prisionero (suma general)
Dos sospechosos son interrogados por separado:

|           | Cooperar | Traicionar |
|-----------|:---------:|:------:|
| Cooperar  |   3, 3    |  0, 5  |
| Traicionar|   5, 0    |  1, 1  |

- "Traicionar" supera a "Cooperar" sin importar lo que haga el otro — es una **estrategia dominante**.
- Ambos jugadores son racionales → ambos traicionan → ambos obtienen 1, aunque (Cooperar, Cooperate) era 3 para cada uno. La mejor respuesta egoísta destruye el bienestar del grupo.
- Esperamos que el Q-learning converja limpiamente a (Traicionar, Traicionar).

### 3) Caza del Ciervo (coordinación)
Dos cazadores pueden abatir juntos un ciervo (gran premio) o conformarse cada uno con una liebre (premio pequeño pero seguro):

|       | Ciervo | Liebre |
|-------|:----:|:----:|
| Ciervo| 4, 4 | 0, 3 |
| Liebre| 3, 0 | 2, 2 |

- (Ciervo, Ciervo) es **dominante en pagos** — lo mejor para ambos.
- (Liebre, Liebre) es **dominante en riesgo** — seguro si no confías en tu compañero.
- El resultado depende de las condiciones iniciales: los aprendices Q independientes a menudo terminan en el equilibrio *peor* (Liebre, Liebre) porque las liebres son más seguras de aprender.

---

## Ejemplos de la vida real

- **Fijación de precios en un duopolio.** Dos cafeterías en la misma calle eligen cada mañana un precio. La forma de la matriz de pagos decide si terminan en un precio "cooperativo" alto (bueno para ellas, malo para los clientes) o en un precio bajo a degüello.
- **Protocolos de red.** Los routers y los emisores eligen estrategias de tiempo; el resultado de la congestión de la red está determinado por el pago tipo juego matricial de pasar vs. retroceder.
- **Pujas en una subasta.** Cada postor elige una puja sin conocer las de los demás; los pagos dependen de todo el vector. El equilibrio de Nash es una *estrategia de puja*, no un solo número.

---

## Qué hace nuestro código

Para cada juego:
1. Creamos dos aprendices Q sin estado (Q es solo un número por acción — no hay estados en un juego de un solo turno).
2. Iteramos durante 20,000 pasos. En cada paso: ambos agentes eligen una acción ε-greedy simultáneamente, reciben una recompensa y actualizan sus valores Q.
3. Rastreamos la **frecuencia de acciones empírica** de cada agente en una ventana móvil de 500 pasos. En lugar de limitarnos a mirar las probabilidades abstractas, contamos qué acciones han elegido realmente hace poco (ej. "en las últimas 500 rondas, jugaron Piedra el 40% de las veces"). Esto nos da una imagen práctica y en tiempo real de su estrategia cambiante.
4. Graficamos las frecuencias a lo largo del tiempo, las guardamos en `outputs/<juego>.png` e imprimimos los valores Q finales.

### Qué deberías ver

| Juego | Resultado esperado del gráfico |
|------|------------------------------|
| **Piedra-Papel-Tijera** | Ambos jugadores rondan el ⅓-⅓-⅓ pero con oscilaciones visibles. Las curvas se persiguen entre sí — comportamiento cíclico clásico. |
| **Dilema del Prisionero** | La frecuencia de "Traicionar" de ambos jugadores sube rápidamente a ~1.0. "Cooperar" es aplastado. |
| **Caza del Ciervo** | La mayoría de las semillas aleatorias se asientan en (Liebre, Liebre). Algunas semillas afortunadas alcanzan (Ciervo, Ciervo) — intenta cambiar la semilla en el script y observa cómo cambia. |

---

## Donde falla el aprendizaje independiente

Nuestros agentes son *independientes* — solo ven su propia recompensa, nunca la acción o los valores Q del oponente. Esta es la línea de base más simple y tiene límites:

- **No puede garantizar la convergencia** en juegos de suma general.
- Puede quedarse atascado en **malos equilibrios** (Caza del Ciervo).
- **No puede modelar al oponente**.

Los algoritmos multiagente reales solucionan esto razonando explícitamente sobre el otro aprendiz. He aquí lo que hace cada uno, en lenguaje sencillo:

| Algoritmo | Idea central | Analogía de la vida real |
|-----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|
| **Fictitious play** | Lleva un recuento de con qué frecuencia tu oponente ha elegido cada acción. Asume que mañana harán lo que siempre han hecho y elige tu mejor respuesta a esa creencia. | Observar los hábitos de un oponente a lo largo de muchas partidas de ajedrez y ajustar tu apertura en consecuencia. |
| **CFR (Counterfactual Regret Minimisation)** | Después de cada ronda, pregunta: *"¿Cuánto me arrepentí de no haber elegido cada una de las otras acciones?"* Desplaza gradualmente la probabilidad hacia las acciones que te arrepientes de haber omitido. Se usa en el póker porque maneja juegos de **información imperfecta**. | Después de una mano de póker, repasarla y pensar: *"Debería haber apostado más — lo haré la próxima vez".* |
| **LOLA (Learning with Opponent-Learning Awareness)** | Tu paso de gradiente tiene en cuenta el hecho de que el oponente *también* está dando un paso de gradiente. Optimizas tu propia actualización anticipando la siguiente actualización del oponente. | Negociar un trato pensando: *"Si ofrezco X, ellos contraatacarán con Y, así que debería empezar con Z".* |
| **MADDPG (Multi-Agent Deep Deterministic Policy Gradient)** | El *crítico* (estimador de valor) de cada agente se entrena con la **visión global**: ve las observaciones y acciones de todos. El *actor* (la política) solo usa información local — este es el patrón CTDE (Centralized Training with Decentralized Execution). | Un entrenador de baloncesto que observa toda la cancha (crítico centralizado) pero enseña a cada jugador a reaccionar solo a lo que puede ver (actor descentralizado). |

Pero el Q-learning independiente es el primer paso correcto. Ves el problema de la no estacionariedad de frente y las soluciones cobran sentido después.

---

## Palabras Clave para Recordar

| Palabra | Significado |
|------|---------|
| **Matriz de pagos** | La tabla que define un juego multiagente de un solo turno |
| **Equilibrio de Nash** | Un perfil de políticas donde ningún agente puede mejorar desviándose individualmente |
| **Estrategia mixta** | Una política que aleatoriza sobre múltiples acciones |
| **No estacionariedad** | El entorno (= otros agentes) cambia constantemente a medida que aprende |
| **Aprendiz independiente** | Un agente que ignora la existencia de otros aprendices |
| **Suma cero** | La ganancia de un agente es exactamente la pérdida del otro |
| **Suma general** | Ambos agentes pueden ganar, ambos pueden perder, o cualquier punto intermedio |

---

## Resumen de una frase

> **En los juegos matriciales, el "entorno" es otro aprendiz — por lo que el mejor movimiento se mueve constantemente.**

Esta es la idea fundamental detrás de cada algoritmo multiagente que conocerás más adelante, desde el self-play hasta MADDPG y el MARL con comunicación.
