# Clonación de Comportamiento (BC) 🐒

## ¿Qué es?

Imagina que quieres aprender a jugar al tenis. Ves cientos de horas de partidos grabados de Wimbledon y simplemente **copias lo que hacen los jugadores**. No piensas si su golpe fue el *mejor* golpe — simplemente ajustas tu posición corporal a la suya y balanceas la raqueta de la misma manera.

Eso es la clonación de comportamiento. **Sin recompensa. Sin planificación. Solo imitación.**

En términos de RL: toma el conjunto de datos de pares `(estado, acción)` y entrena una red neuronal para predecir la acción a partir del estado, exactamente como un modelo de clasificación de imágenes predice gato vs. perro. La "etiqueta" es cualquier acción que haya tomado el recolector de datos.

---

## En qué se diferencia del RL Offline "Real"

| Enfoque | ¿Usa recompensas? | ¿Puede superar a los datos? |
|----------|---------------|---------------------|
| **BC**   | ❌ no         | ❌ no — en el mejor de los casos, iguala la calidad media de los datos |
| **CQL** (y amigos) | ✅ sí | ✅ sí — puede extraer buenas transiciones de datos mixtos |

BC es la "visión de aprendizaje supervisado" del RL. Es increíblemente simple, a menudo sorprendentemente fuerte, y el estándar de referencia universal. **Si un algoritmo de RL offline no puede superar a BC en el mismo conjunto de datos, no ha hecho nada.**

---

## Ejemplos de la vida real

- **Aprender a conducir a partir de grabaciones de cámaras de salpicadero.** Mirar la carretera, predecir el ángulo del volante que usó el humano. Dos ejemplos históricos:
  - **ALVINN (1989)** — el primer conductor de red neuronal; una pequeña red de 3 capas entrenada con entradas de cámara + láser para conducir una furgoneta por autopistas.
  - **NVIDIA PilotNet (2016)** — una CNN profunda moderna entrenada de extremo a extremo con grabaciones de cámaras de salpicadero; aprendió a mantener el carril y a realizar maniobras básicas de dirección puramente imitando a conductores humanos, sin reglas diseñadas a mano.
- **Aprendiz copiando a un maestro chef.** "Lo que sea que haga el chef, yo lo hago". Funciona muy bien si el chef es excelente; produce un mal chef si el chef es malo.
- **GitHub Copilot.** El autocompletado está entrenado para predecir "¿qué código escribiría un humano a continuación?" — pura imitación de registros de código fuente.
- **Mimetizar a tu hermano mayor.** Los niños hacen esto durante años antes de empezar a razonar sobre *por qué* su hermano mayor hace lo que hace.

---

## Las Matemáticas (Una Línea)

Para cada `(s, a)` en el conjunto de datos, minimizar:

```
pérdida = -log π(a | s)        (entropía cruzada)
```

Eso es todo. La política `π` es solo un MLP que genera logits de acción; el entrenamiento es idéntico a MNIST. Vamos a desglosar la jerga:
- **`π` (Pi):** El símbolo estándar para "política" — la regla o red neuronal que decide qué hacer.
- **MLP (Multi-Layer Perceptron):** Una red neuronal básica y estándar.
- **Logits:** Las puntuaciones brutas y no normalizadas que la red escupe antes de que las convirtamos en probabilidades.
- **Entropía cruzada (Cross-entropy):** La fórmula estándar para penalizar a un modelo cuando asigna una probabilidad baja a la respuesta correcta.
- **MNIST:** El famoso conjunto de datos para principiantes de dígitos escritos a mano.

Entrenar a un agente para jugar un juego a través de BC es literalmente idéntico a entrenar a una red para reconocer dígitos escritos a mano en MNIST. En MNIST, la entrada es una imagen y la salida es un dígito (0-9). En BC, la entrada es el estado del juego y la salida es la acción (por ejemplo, "moverse a la izquierda").

---

## Qué hace nuestro código

El script `behavioral_cloning.py`:

1. **Carga los cuatro conjuntos de datos** creados por `d4rl_dataset.py` (`random`, `medium`, `expert`, `medium-replay`).
2. Para cada conjunto de datos, **entrena una política BC separada** durante 10,000 pasos de gradiente de entropía cruzada. La columna de recompensa se ignora por completo.
3. Cada 2,500 pasos, **evalúa** la política actual ejecutándola de forma codiciosa (greedy) en el entorno real CartPole-v1 (20 episodios, promediados).
4. Grafica:
   - Un gráfico de barras: retorno final de BC por conjunto de datos.
   - Un gráfico de curva de aprendizaje: cómo sube cada variante de BC durante el entrenamiento.

---

## Qué deberías ver

Una ejecución típica imprime:

```
Retornos finales de evaluación:
  BC en random          ->    ~20  ± unos pocos   (≈ juego aleatorio)
  BC en medium          ->   ~150  ± grande       (≈ la política medium)
  BC en expert          ->   ~480  ± pequeño      (≈ la política expert)
  BC en medium-replay   ->    ~60  ± grande       (≈ el PROMEDIO de datos mixtos)
```

El gráfico de barras hace que la historia sea obvia: **el retorno de BC sigue el retorno promedio del conjunto de datos.** No puede superar ese techo porque no tiene forma de preferir las partes "buenas" de un conjunto de datos mixto sobre las partes "malas" — ambos son objetivos de imitación igualmente válidos.

Ese es el remate: **BC hereda el techo de los datos.**

---

## BC vs CQL — La Comparación más Clara

En el conjunto de datos **medium-replay** (el caso más realista de calidad mixta):

| Método | Retorno final aprox. | ¿Por qué? |
|--------|--------------------:|------|
| BC     | ~60   | Imita el *promedio* de las primeras ejecuciones fallidas + las buenas posteriores |
| CQL    | ~400+ | Usa recompensas para preferir las transiciones con Q alto; teje una buena política a partir de datos mixtos |

Así que CQL **supera a los datos**, BC **iguala a los datos**. Esa es la razón por la que el RL offline es un campo de investigación y no solo "hacer aprendizaje por imitación". Cuando los datos son de calidad mixta (que los registros reales siempre lo son), los métodos que consideran la recompensa recuperan más.

En datos de **expertos**, la comparación se invierte: BC iguala a los expertos (~480). Quizás te preguntes por qué CQL "empata" aquí en lugar de perder. Debido a que CQL está diseñado para ser *conservador* y penalizar las acciones no vistas en el conjunto de datos, termina haciendo exactamente lo que hizo el experto. No puede superar al experto (porque ya se ha alcanzado la puntuación máxima posible), pero tampoco rompe activamente la estrategia del experto. Simplemente empata con el rendimiento de BC.

Este es el famoso equilibrio entre "calidad de los datos vs. algoritmo":

```
                            Datos EXPERTOS  →  BC gana, CQL empata
   Sofisticación del algoritmo ↑         
                            Datos MIXTOS    →  CQL supera claramente a BC
                            
                            Datos RANDOM    →  Todos fallan; se necesita exploración
```

---

## Dónde vive el BC en el RL Moderno

- **Pre-entrenamiento para RL online.** Muchos sistemas modernos (RT-1, Voyager, bots de juegos) comienzan con BC en demostraciones y luego se ajustan online con PPO/SAC.
- **RLHF.** El paso 1 de InstructGPT es un ajuste fino supervisado — puro BC sobre respuestas escritas por humanos. PPO + modelo de recompensa vienen después.
- **DAgger (Ross et al., 2011).** Una extensión inteligente para solucionar el problema del **error compuesto**.
  *¿Por qué el error compuesto es un problema si BC clona perfectamente?* Incluso si un modelo BC tiene una precisión del 99%, ese 1% de error acaba ocurriendo. Cuando ocurre, el agente entra en un estado que nunca ha visto en el conjunto de datos conducido perfectamente. Debido a que está confundido, comete un error mayor, alejándose aún más de los datos conocidos, lo que se traduce en un fallo total (como caer por un acantilado).
  *La solución:* Podríamos pedirle al experto que conduzca siempre, pero el tiempo del experto es caro. En su lugar, DAgger deja que la política BC conduzca. Cuando la política comete un error y se desvía hacia un estado extraño, hacemos una pausa, preguntamos al experto "¿qué harías *justo aquí*?", y añadimos eso al conjunto de datos. Solo "volvemos a consultar al experto en los estados que visita la política BC" porque solo necesitamos que el experto nos enseñe a recuperarnos de nuestros propios errores específicos, en lugar de consultarle siempre.
- **Decision Transformer (Chen et al., 2021).** Un BC "inteligente" que condiciona la predicción de la acción a un *retorno por alcanzar* deseado, convirtiendo esencialmente el RL offline de nuevo en una predicción del siguiente token.

---

## Palabras Clave para Recordar

| Palabra | Significado |
|------|---------|
| **Aprendizaje por imitación** | Término general para "copiar al demostrador"; BC es el miembro más simple |
| **Error compuesto** | Un pequeño error de BC te lleva a estados que el conjunto de datos nunca vio, donde los errores se acumulan |
| **Datos de demostración** | Trayectorias producidas por un experto, usadas como conjunto de entrenamiento de BC |
| **Techo de datos** | El retorno de BC está limitado por el retorno promedio en el conjunto de datos |
| **DAgger** | Una solución interactiva para el error compuesto |

---

## Resumen de una frase

> **La clonación de comportamiento es simplemente aprendizaje supervisado sobre pares (estado, acción) — fuerte cuando los datos son buenos, inútil cuando los datos son mixtos.**
