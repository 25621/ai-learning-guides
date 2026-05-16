# Conjuntos de Datos Benchmark D4RL 📦

## ¿Qué es?

Imagina que quieres enseñar a un robot a dar la vuelta a las tortitas. Dejar que practique en una estufa real durante un mes sería lento, peligroso y caro. Pero tienes diez años de vídeo grabado de chefs dando la vuelta a las tortitas (algunos buenos, otros malos, otros aleatorios). ¿Puedes enseñar al robot *solo con esos datos*, sin dejar que toque nunca una sartén real?

Eso es el **aprendizaje por refuerzo offline** (offline reinforcement learning). El agente aprende de un conjunto de datos fijo de experiencia pasada — sin entorno en vivo. La parte más difícil es que el agente nunca puede *probar* lo que aprendió hasta el final.

Para que este estudio fuera justo, la comunidad de investigación necesitaba un *conjunto de datos estándar*. Eso es **D4RL** (**D**atasets for **D**eep **D**ata-**D**riven **R**einforcement **L**earning): una colección de transiciones pre-grabadas para tareas de control clásicas, lanzada por la UC Berkeley en 2020. Cada artículo se entrena con los mismos bytes, por lo que los resultados son comparables.

---

## ¿Qué hay en un conjunto de datos D4RL?

Para cada tarea, D4RL ofrece **cuatro niveles de calidad**:

| Nivel | De dónde vienen los datos | Por qué importa |
|-------|---------------------------|----------------|
| **random** (aleatorio) | Una política que elige acciones uniformemente al azar | El peor caso: ¿puedes aprender algo útil? |
| **medium** (medio) | Una política parcialmente entrenada (aproximadamente la mitad de la puntuación de un experto) | Realista: la mayoría de los datos registrados son mediocres |
| **expert** (experto) | Una política casi convergente | El mejor caso: ¿puedes igualar la política de origen? |
| **medium-replay** | El *búfer de repetición completo* usado para entrenar la política medium | Mixto: contiene fallos tempranos Y éxitos posteriores |

La diferencia entre `medium` y `medium-replay` es crucial:
- **`medium`** se genera tomando una única política "media" fija y dejándola jugar muchas partidas. Todos los datos reflejan este nivel de habilidad constante y medio.
- **`medium-replay`** es un registro histórico. Contiene todas las experiencias recopiladas *mientras se aprendía* desde cero hasta el nivel medio. Mezcla transiciones **malas y buenas** — exactamente como se ve un registro del mundo real (los primeros intentos torpes de un robot *y* su comportamiento refinado posterior, todo en un mismo cubo).

---

## Ejemplos de la vida real de conjuntos de datos Offline

- **Registros médicos.** Años de tuplas de (estado_del_paciente, tratamiento, resultado). No puedes aleatorizar los tratamientos en personas vivas, pero puedes aprender una mejor política a partir del registro.
- **Registros de chat de servicio al cliente.** Millones de registros de (mensaje_del_usuario, respuesta_del_agente, satisfacción). Entrena un mejor asistente sin molestar a más usuarios.
- **Datos de flotas de conducción autónoma.** Cada coche Tesla / Waymo sube sus trayectos. La flota es un conjunto de datos medium-replay gigante.
- **Sistemas de recomendación.** Los registros de clics del año pasado son un conjunto de datos congelado: no puedes volver a mostrar los mismos anuncios a los mismos usuarios.

En los cuatro casos, **no puedes pedir al entorno una nueva muestra.** El conjunto de datos es lo que tienes. Para siempre.

---

## Qué hace nuestro código

Los conjuntos de datos D4RL reales se registran en tareas de locomoción de MuJoCo (Multi-Joint dynamics with Contact) (como HalfCheetah, Hopper, Walker2d, Ant — estas son simulaciones físicas 3D avanzadas donde robots virtuales aprenden a caminar y correr). MuJoCo es pesado de instalar, así que recreamos la **misma estructura de cuatro niveles en CartPole-v1** — el entorno estándar para principiantes de fases anteriores. Las lecciones se transfieren directamente.

El script `d4rl_dataset.py`:

1. **Entrena un DQN** (Deep Q-Network, un algoritmo estándar de RL) en CartPole hasta que resuelve la tarea (retorno ≥ 475).
2. **Realiza capturas de dos puntos de control** (checkpoints) a lo largo del camino:
   - "medium": el momento en que el retorno reciente superó los 150.
   - "expert": el momento en que el retorno reciente superó los 475.
3. **Captura el búfer de repetición completo de la política media** — cada transición que vio. Ese es nuestro conjunto de datos "medium-replay".
4. **Ejecuta tres nuevas políticas** durante 10,000 transiciones cada una:
   - `random`: aleatorio uniforme.
   - `medium`: el punto de control medium + ruido ε=0.10.
   - `expert`: el punto de control expert + ruido ε=0.02.
5. **Guarda cuatro archivos `.npz`** (formato de matriz comprimida de NumPy) en `outputs/`, cada uno con matrices `obs / action / reward / next_obs / terminal`.

Estos cuatro archivos son las entradas para `cql.py` y `behavioral_cloning.py`.

---

## Qué deberías ver cuando lo ejecutes

Un resumen en texto plano impreso en la consola y guardado en `outputs/d4rl_summary.txt`:

```
dataset         |   N    |  mean return  |  min  |  max
------------------------------------------------------------
random          | 10000  |          ~22  |    ~9 |   ~80
medium          | 10000  |         ~180  |   ~50 |  ~500
expert          | 10000  |         ~490  |  ~400 |   500
medium-replay   | 10000  |          ~60  |    ~9 |  ~200
```

También genera un histograma (`outputs/d4rl_returns.png`) que muestra cómo se solapan los cuatro conjuntos de datos. Los rasgos clave a notar son:

- **Random** se agrupa alrededor de 20 (la duración promedio de un episodio aleatorio de CartPole).
- **Expert** se agrupa en el techo de 500.
- **Medium** se sitúa en medio, con alta varianza.
- **Medium-replay** tiene una larga cola a la derecha — consiste principalmente en ejecuciones fallidas tempranas (bajos retornos) pero tiene una cola que se extiende a retornos más altos a medida que el agente aprendía.

---

## Por qué importa el conjunto de datos

Sea cual sea el conjunto de datos en el que entrenes tu algoritmo offline, estás poniendo un *techo* a lo que es posible:

- **Desde `expert`**: incluso un algoritmo simple como BC (Behavioral Cloning, que solo copia los datos exactamente) puede hacerlo bien, porque todos los datos son buenos.
- **Desde `random`**: necesitas un algoritmo inteligente que pueda *unir* (stitch together) las raras transiciones buenas (encontrar un camino al éxito combinando secuencias cortas de buenas acciones de diferentes intentos). BC fallará por completo.
- **Desde `medium-replay`**: el más realista y el más interesante. Los buenos algoritmos (como **CQL** — Conservative Q-Learning, que evita confiar demasiado en acciones que nunca ha visto) a veces pueden **superar la calidad promedio de los datos** porque extraen estructura de señales mixtas. Los algoritmos simples (BC) regresan a la media.

Veremos exactamente esta historia en los próximos dos scripts.

---

## Palabras Clave para Recordar

| Palabra | Significado |
|------|---------|
| **Offline RL** | Entrenar a partir de un conjunto de datos fijo; no se permite interacción con el entorno |
| **Política de comportamiento** | La política que *produjo* el conjunto de datos |
| **Calidad del conjunto de datos** | Qué tan buena fue la política de comportamiento (random / medium / expert) |
| **Búfer de repetición (Replay buffer)** | El historial completo de transiciones vistas durante una ejecución de entrenamiento |
| **Desplazamiento de la distribución (Distribution shift)** | La brecha entre las acciones en el conjunto de datos y las acciones que tu política entrenada quiere tomar. Debido a que el conjunto de datos nunca muestra qué sucede cuando la nueva política intenta algo que no fue registrado, las estimaciones de valor del algoritmo para esas acciones novedosas pueden ser peligrosamente erróneas. |

---

## Resumen de una frase

> **D4RL congela el RL en un benchmark de estilo de aprendizaje supervisado: los mismos bytes para todos, sin trampas del entorno, que gane el mejor algoritmo.**
