# Dyna-Q: Aprendiendo más rápido a través de la imaginación 🧠

## ¿Qué es?

Imagina a una niña llamada Mia aprendiendo a moverse en su nueva escuela. Cada día camina por los pasillos y descubre cosas nuevas: "La biblioteca está después de la cafetería", "La clase del Sr. Smith está arriba, cerca de la escalera".

Un estudiante de **Q-learning puro** solo aprende de lo que hace *hoy*. Si hoy solo caminó de clase a la cafetería, solo actualiza su memoria sobre ese único camino.

Un estudiante de **Dyna-Q** es diferente. Después de cada caminata real, se sienta un minuto y **reproduce en su cabeza** varias caminatas pasadas que recuerda. Cada reproducción fortalece su mapa mental. Después de unas semanas, conoce la escuela de memoria — no porque caminara más, sino porque **pensó más en lo que ya vio**.

Eso es exactamente lo que Dyna-Q hace por un agente de RL: aprende de la experiencia real **y** de la experiencia imaginada extraída de un modelo que construye por el camino.

---

## Los Tres Ingredientes

Dyna-Q es "Q-learning + modelo + planificación". Un solo paso real realiza **tres** tareas:

1. **RL Directo**: la actualización habitual de Q-learning a partir de `(s, a, r, s')`.
2. **Aprendizaje del modelo**: anotar: "Cuando hice *a* en *s*, obtuve *r* y terminé en *s'*".
3. **Planificación**: elegir *n* pares `(s, a)` al azar de la memoria del modelo y realizar *n* actualizaciones más de Q-learning, **fingiendo** que esos pasos acaban de ocurrir.

Ese tercer paso es la magia. Con `n = 50`, cada paso real en el mundo provoca **51 actualizaciones** en la tabla Q. El agente aprende unas 50 veces más rápido — en términos de pasos reales — que un aprendiz de Q-learning puro.

---

## Una imagen del bucle

```
                   ┌────────────────────────────────────┐
                   │                                    │
   mundo real  ──► tomar acción a ──► observar (r, s')  │
                            │                           │
              ┌─────────────┼──────────────┐            │
              ▼             ▼              ▼            │
      actualización      Modelo[s,a] ← (r,s')  Planificación ─┘
       Q-learning                            (n actualizaciones imaginadas)
```

El modelo es simplemente una tabla de búsqueda:
`(estado, acción) → (recompensa, siguiente_estado)`. Barato de construir, barato de consultar.

---

## Ejemplos de la vida real

- **Estudio de ajedrez**. Los grandes maestros pasan horas reproduciendo sus propias partidas y partidas maestras en sus cabezas. Cada reproducción es "planificación" — aprendizaje extra de experiencias que ya ocurrieron.
- **Un músico practicando escalas**. Después de tocar un compás difícil una vez, lo ensayan mentalmente diez veces más antes de seguir adelante. Los dedos no se mueven, pero el cerebro se está actualizando.
- **Un automóvil autónomo**. Mientras está parado en un semáforo en rojo, reproduce los últimos cien cambios de carril en simulación para ajustar su política sin gastar neumáticos.

---

## Qué hace nuestro código

Usamos el clásico **Dyna Maze** ([Sutton & Barto, Figura 8.2](http://incompleteideas.net/book/the-book.html)): una cuadrícula de 6×9 con algunas paredes, un inicio `S` en el centro-izquierda y una meta `G` en la parte superior derecha.

Ejecutamos tres variantes, cada una promediada sobre 30 semillas aleatorias:

| Configuración | Pasos de planificación por paso real | Significado |
|---------|------------------------------|---------|
| `n = 0` | 0 | Q-learning puro |
| `n = 5` | 5 | un poco de práctica imaginada |
| `n = 50` | 50 | mucha práctica imaginada |

El script informa del **número promedio de pasos reales por episodio** a medida que avanza el entrenamiento. Menos pasos significa que el agente ha aprendido un camino más directo a la meta.

### Qué deberías ver cuando lo ejecutes

El camino más corto en este laberinto es de unos 9 pasos; con la exploración ε-greedy, un agente bien entrenado promedia unos 10 pasos por episodio. Ejecútalo durante 50 episodios y las tres configuraciones convergerán allí — la diferencia es *qué tan rápido*:

| Configuración | Pasos por episodio (últimos 10 eps) | Qué significa |
|---------|--------------------------------:|---------------|
| `n = 0`   | ~10 | Convergió — pero le llevó entre 30 y 50 episodios de deambular llegar aquí |
| `n = 5`   | ~10 | Convergió en unos 10 episodios |
| `n = 50`  | ~10 | Convergió en unos 3–5 episodios |

La señal interesante es la *curva de aprendizaje*, no el número final. El gráfico guardado en `outputs/dyna_q.png` muestra tres curvas cayendo hacia el suelo a ritmos muy diferentes: `n = 50` llega en un puñado de episodios, mientras que `n = 0` todavía está bajando bien entrada la carrera. (En un laberinto determinista diminuto como este, el Q-learning puro acaba llegando — Dyna-Q simplemente necesita muchos menos episodios reales, que es el objetivo principal en entornos donde los pasos reales son costosos).

---

## Por qué funciona tan bien en este laberinto

Dos razones:

1. **El entorno es determinista**. Cada `(s, a)` siempre da el mismo `(r, s')`, por lo que el modelo es exacto después de una sola visita. La experiencia imaginada es tan buena como la real.
2. **Los pasos reales son caros, los imaginados son gratis**. Cada actualización imaginada es solo unas pocas búsquedas en una tabla, mientras que un paso real requiere que el agente camine. Cuando las interacciones reales son costosas (piensa en un robot real, un juego real), Dyna-Q es enormemente eficiente en cuanto a muestras.

---

## Donde Dyna-Q tiene dificultades

- **Entornos estocásticos**. Si `(s, a)` puede conducir a muchos valores `s'` diferentes, un modelo de "recordar el último resultado" te miente. Solución: almacenar recuentos de visitas o entrenar un modelo probabilístico.
- **Entornos no estacionarios**. Si el mundo cambia — por ejemplo, una puerta que estaba abierta se cierra de repente, o aparece un atajo — el modelo queda desactualizado y da predicciones erróneas. **Dyna-Q+** soluciona esto añadiendo un *bono de exploración*: los estados que no han sido revisitados durante mucho tiempo reciben una pequeña recompensa extra, incitando al agente a volver y comprobar si el mundo ha cambiado.
- **Espacios de estados grandes**. Un diccionario indexado por `(s, a)` no escala a millones de estados o a estados continuos. Ese es exactamente el hueco que llenan los **modelos de mundo aprendidos (redes neuronales)** — ver `world_model.py` a continuación.

---

## Palabras Clave para Recordar

| Palabra | Significado |
|------|---------|
| **Modelo** | Memoria de `(estado, acción) → (recompensa, siguiente_estado)` |
| **Paso de planificación** | Realizar una actualización Q usando datos imaginados del modelo |
| **RL Directo** | Una actualización Q usando datos reales |
| **Eficiencia de muestras** | Mide con qué eficacia un modelo o algoritmo de IA utiliza los datos de entrenamiento para alcanzar un nivel específico de rendimiento |
| **Dyna** | Arquitectura de Sutton que entrelaza aprendizaje + planificación |

---

## Resumen de una frase

> **Dyna-Q aprende de la acción Y de la imaginación — e imaginar es gratis.**

Esta idea, en su forma neuronal moderna, impulsa a algunos de los agentes de RL más fuertes jamás construidos (MuZero, Dreamer, World Models).
