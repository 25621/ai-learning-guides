# Entrenamiento en Montezuma's Revenge 🏛️🔑

## Por qué este juego es famoso (en los círculos de RL)

En 2015, el DQN de DeepMind aprendió a jugar a docenas de juegos de Atari a un nivel sobrehumano a partir de píxeles brutos. Fue noticia en todo el mundo. Pero enterrado en la tabla de resultados había un juego en el que DQN obtuvo un **0** — lo mismo que no hacer nada en absoluto: **Montezuma's Revenge**.

¿Por qué? Mira lo que el juego te pide en la primera habitación:

1. Bajar una escalera.
2. Caminar por una cornisa.
3. Saltar sobre una calavera que rueda (si te equivocas en el tiempo → mueres).
4. Subir por otra escalera.
5. Coger la llave.

Eso son aproximadamente **100 pulsaciones de botón precisas**, y el juego no te da **ni un solo punto** hasta que tienes la llave en la mano. La señal de recompensa es un **cero** absoluto y plano durante toda la secuencia inicial.

Un agente de RL normal aprende ajustándose hacia las recompensas que realmente recibe. Si la recompensa es cero en todos los lugares a los que puede llegar, no hay *nada de lo que aprender* — es como intentar encontrar el fondo de un valle perfectamente plano buscando la dirección de la pendiente. Así que DQN simplemente se agitó alrededor de la plataforma de inicio para siempre. Montezuma se convirtió en *el* estándar de referencia para la **exploración difícil** (hard exploration): el juego que solo puedes ganar si exploras de forma *inteligente*, no aleatoria.

El gran avance llegó en 2018 con **Random Network Distillation (RND)** — y el truco fue exactamente el tema del trabajo 1: añadir un **bono de curiosidad intrínseca** para que el agente se recompense a *sí mismo* por llegar a pantallas nuevas, y de repente tenga una señal densa que lo arrastre más profundamente en el nivel. RND obtuvo una puntuación sobrehumana en Montezuma. (Más tarde: Go-Explore, Agent57, …)

## Ejemplos de la vida real de recompensa escasa "estilo Montezuma"

- **Una cerradura de combinación / una búsqueda del tesoro con pistas crípticas**. No hay crédito parcial. Estás en cero hasta que de repente llegas al premio.
- **Conseguir que acepten un artículo de investigación, o que una startup sea rentable**. Meses sin recompensa externa, y luego (quizás) una grande.
- **Una ruta de speedrun de un videojuego**. Docenas de entradas perfectas en el tiempo seguidas sin retroalimentación hasta que el truco funciona o no.
- **Escape rooms**. La sala no te dice casi nada hasta que has encadenado varios descubrimientos.

En todos estos casos, "simplemente probar cosas al azar" es inútil. Necesitas explorar de forma *sistemática* — y una señal interna de "oh, eso es nuevo, sigue adelante" es lo que te mantiene sistemático.

## Por qué no entrenamos realmente en el Montezuma de píxeles aquí

Hacerlo *realmente* bien significa:

- una red convolucional para ver la pantalla RGB de 210×160,
- apilamiento de frames (para que el agente sepa hacia dónde se mueve la calavera),
- un módulo RND (dos redes más: un "objetivo" aleatorio fijo y un "predictor" entrenado),
- y **decenas de millones de frames del entorno** — muchas horas de GPU.

Eso es un proyecto de investigación, no un script de enseñanza. Así que `montezuma_revenge.py` hace dos cosas honestas en su lugar:

### 1. "Toca" el juego real (si `ale-py` está instalado)

Carga `ALE/MontezumaRevenge-v5` a través de Gymnasium, ejecuta un **agente aleatorio uniforme durante 2000 pasos**, e informa de la recompensa total del juego. El número que imprime es casi siempre **0.0** — la frase abstracta "recompensa escasa" convertida en un hecho concreto que puedes ejecutar tú mismo. Si el paquete de Atari no está instalado, imprime el comando `pip install` de una línea y continúa.

### 2. Entrena a un agente tabular en un *modelo a escala*: `MiniMontezumaEnv`

Este es un pequeño mundo de cuadrícula con el mismo *esqueleto* que la primera habitación de Montezuma:

```
###############
#S....#.......#
#.....#.......#
#.....#...T...#     S = inicio (start)
#.....D.......#     K = llave (key)      D = puerta (solo pasable con la llave)
#..K..#.......#     T = tesoro (la ÚNICA casilla que da recompensa)
###############
```

Para ganar debes: caminar hasta la **llave** (~6 movimientos), recogerla; caminar hasta la **puerta** (~4 movimientos) — que ahora se abre; atravesarla y llegar al **tesoro** (~5 movimientos). Unos **15 movimientos perfectos**, con **cero retroalimentación hasta el tesoro**. El flag `has_key` es parte del estado del agente, así que una vez que tomas la llave hay toda una segunda habitación de *nuevos* estados por descubrir — igual que las nuevas pantallas que se abren en el juego real.

Luego entrenamos un **Q-learner tabular** normal dos veces:

| Agente | Resultado en MiniMontezuma |
|-------|--------------------------|
| **sin curiosidad (epsilon-greedy)** | El retorno se mantiene en **0** durante los 1,500 episodios. Ni siquiera llega a la llave. (¿Te suena? Eso es DQN en el juego real). |
| **con un bono de curiosidad por error de predicción** | Llega al tesoro en unos 20–25 episodios y luego aprende la **ruta óptima de 15 pasos**. (Esa es la idea de RND, encogida para caber en una tabla Q). |

La figura muestra las dos curvas de aprendizaje lado a cara, además de la ruta real que aprendió el agente curioso, dibujada sobre la cuadrícula (inicio → llave → puerta → tesoro). El script también imprime esa ruta como frames ASCII.

## La Lección

> **La "recompensa escasa" no es una peculiaridad de un juego extraño de Atari — es lo predeterminado en cualquier mundo donde el éxito requiere una secuencia larga y específica de acciones.** Un agente que solo se basa en recompensas (DQN vainilla) literalmente no puede empezar: no hay gradiente que seguir. Un bono de curiosidad fabrica uno — una señal densa y autogenerada de "esto es nuevo, sigue adelante" — y esa señal es lo que lleva al agente a través del desierto de ceros hasta la primera recompensa real. Todo lo que viene después (RND, Go-Explore, Agent57) es una versión a mayor escala con redes neuronales del mismo movimiento.

## Palabras Clave para Recordar

| Palabra | Significado |
|------|---------|
| **Exploración difícil (Hard exploration)** | Problemas donde solo tienes éxito explorando inteligentemente; la exploración aleatoria falla |
| **Recompensa escasa (Sparse reward)** | La recompensa es cero en casi todas partes; solo la obtienes después de una secuencia correcta larga |
| **Montezuma's Revenge** | El juego de Atari en el que los agentes de RL profundo clásicos (DQN, A3C) obtuvieron un 0 — el benchmark canónico de exploración difícil |
| **RND (Random Network Distillation)** | El método de 2018 que superó Montezuma usando un bono de curiosidad por error de predicción |
| **Go-Explore** | "Recordar estados prometedores, volver a ellos y explorar desde allí" — otro sistema que superó Montezuma |
| **Modelo a escala** | Un entorno pequeño y barato que mantiene la *estructura* de un problema difícil para que puedas estudiarlo rápidamente |

## Resumen de una frase

> **Montezuma's Revenge es el juego que enseñó al RL que "las recompensas que nunca recibes no pueden enseñarte nada" — y la solución, entonces y ahora, es un bono de curiosidad que permite al agente recompensarse a sí mismo por explorar hasta que encuentra el premio real.**
