# Uso de un Modelo Aprendido para la Planificación (MPC) 🔮

## La Gran Idea

Ya tienes un **modelo de mundo** (una red neuronal que predice el futuro). ¿Y ahora qué?

El uso más directo es la **planificación**: en cada momento, pregunta al modelo "¿qué pasaría si intentara *este* plan? ¿*ese* plan? ¿*aquel otro* plan?". Luego, ejecuta el plan que parezca mejor — pero **solo el primer paso del mismo**. Debido a que el modelo no es perfecto, ejecutamos solo una acción, observamos el nuevo estado real del entorno real y luego volvemos a planificar desde cero.

Este truco tiene un nombre: **Control Predictivo por Modelo** (Model Predictive Control, MPC).

---

## Una Analogía de la Vida Real

Estás en un restaurante mirando el menú. No te comprometes a pedir cinco platos a la vez; pides el primero, ves cómo de lleno estás y luego decides el postre.

O bien: estás conduciendo por una carretera con muchas curvas. No fijas los movimientos del volante para los próximos 30 segundos; miras constantemente hacia adelante, planeas unos segundos, realizas la siguiente acción de dirección y vuelves a planificar.

Ese bucle de **planear lejos / actuar cerca / volver a planificar** es el MPC.

---

## Cómo funciona el "Random Shooting"

Existen planificadores más sofisticados; por ejemplo:
- **CEM** (Cross-Entropy Method): perfecciona iterativamente una distribución sobre los planes quedándose solo con los mejores de cada ronda.
- **MCTS** (Monte Carlo Tree Search): construye un árbol de búsqueda guiado por estadísticas de simulación, utilizado por AlphaGo y MuZero.
- **Planificadores basados en gradiente**: diferencian las predicciones del modelo con respecto a las acciones y siguen el gradiente directamente.

Nosotros usamos el más simple que funciona: **random shooting** (disparo aleatorio).

```
Dado el estado actual s:
    1. Muestrear N=200 secuencias de acciones aleatorias de longitud H=15.
    2. Para cada secuencia, simularla a través del modelo de mundo desde s, sumando
       una recompensa moldeada (shaped) en cada paso. (¡200 sueños en paralelo — rápido!)
    3. Encontrar la secuencia con la mayor recompensa total predicha.
    4. Ejecutar la PRIMERA acción de esa secuencia en el entorno real.
    5. Observar el estado real siguiente. Descartar el resto del plan.
    6. Ir al paso 1 — volver a planificar desde cero.
```

200 planes × 15 pasos = 3,000 transiciones imaginadas por cada paso real. El modelo de mundo las ejecuta todas en una sola pasada de red neuronal por lotes — normalmente unos pocos milisegundos.

---

## ¿Por qué volver a planificar en cada paso?

Porque el modelo es imperfecto. Los errores se acumulan a lo largo de una ejecución (como se ve en el gráfico generado por `world_model.py`, guardado en `outputs/world_model.png`). El plan en el paso 0 es fiable solo para los primeros movimientos; en el paso 15 el modelo está alucinando. Por eso confiamos solo en el **primer movimiento** y luego refrescamos el plan con el estado real más reciente.

Esta es la misma razón por la que los humanos no escriben un plan de ajedrez de 100 jugadas y se ciñen a él: las circunstancias cambian y, cuanto más lejos intentas adivinar, menos coincide con la realidad.

---

## Un detalle: La recompensa tiene que decirle algo al planificador

En CartPole, la recompensa real es `+1` en cada paso hasta que el poste se cae. El modelo predecirá fielmente `+1, +1, +1, ...` para casi cualquier plan, porque los planes aleatorios rara vez terminan rápido dentro del modelo — y por tanto todos los planes puntúan igual. El planificador no tiene nada que elegir.

La solución: sustituir la recompensa real por un **sustituto suave** (smooth proxy) durante la planificación:

```python
recompensa_sustituta(estado) = 1
                             - |angulo_poste| / 0.21          # ¿poste vertical? (1=sí)
                             - 0.1 * |posicion_carro| / 2.4   # ¿carro centrado? (1=sí)
```

Ahora los planes que *terminarían* con un poste caído obtienen puntuaciones visiblemente peores que los planes que se mantienen verticales. El planificador puede clasificarlos.

> **Lección de la vida real**: Una señal de recompensa plana — "has sobrevivido un segundo más" — es inútil para la planificación a corto plazo. Las señales densas y moldeadas ayudan.

---

## Qué hace nuestro código

`model_based_planning.py`:

1. **Carga** los pesos del modelo de mundo guardados por `world_model.py`. (Si faltan, entrena uno sobre la marcha).
2. **Ejecuta 20 episodios** de MPC en el CartPole-v1 real.
3. **También ejecuta 20 episodios** con una política uniformemente aleatoria, como línea de base.
4. **Grafica** ambos cara a cara e imprime los promedios.

### Qué deberías ver cuando lo ejecutes

| Política | Recompensa promedio (pasos sobrevividos) |
|--------|-------------------------------:|
| Aleatoria     | ~22 (típico de CartPole — el poste cae rápido) |
| MPC (nuestro) | ~150–500 (varía según la semilla; muchos episodios cerca de 500) |
| Máximo posible| 500 |

Esa **mejora de 5–25 veces** se logra sin red de política, sin función de valor y sin más entrenamiento. Solo un modelo de mundo + 200 sueños por paso.

El gráfico `outputs/model_based_planning.png` muestra dos barras de colores por episodio — el MPC casi siempre es más alto que el Aleatorio, y muchos episodios alcanzan el techo de 500 pasos.

---

## Fortalezas de la Planificación Basada en Modelos

- **Eficiencia de muestras**. Todo el aprendizaje se realizó a partir de un lote de transiciones aleatorias. No se necesitó más interacción con el entorno para derivar una política útil.
- **Fácil de redirigir**. ¿Quieres controlar al agente de forma diferente? Cambia la recompensa sustituta — no hace falta reentrenar. (Intenta maximizar la velocidad del carro por diversión).
- **Interpretable**. Puedes inspeccionar los planes que el agente consideró, las trayectorias predichas y las puntuaciones.

## Debilidades (y qué hace la gente al respecto)

- **El "random shooting" es básico**. Muestrea planes a ciegas. Para dimensiones más altas, se cambia a **CEM** (Método de Entropía Cruzada — ver arriba) o **iLQR** (Iterative Linear-Quadratic Regulator, un método clásico de control óptimo que aproxima el modelo como lineal localmente y lo resuelve analíticamente) o un planificador basado en gradiente que mejora las acciones siguiendo los gradientes a través de un modelo diferenciable.
- **Error de modelo acumulativo**. Los horizontes largos se desvían. La gente usa **conjuntos (ensembles) probabilísticos** (varios modelos entrenados con los mismos datos, como en PETS, Chua et al. 2018) para que el planificador pueda notar el desacuerdo y penalizar los planes sobre los que el modelo no está seguro.
- **La recompensa real es lo que queremos, al fin y al cabo**. El moldeado de recompensas ayuda, pero para tareas más complejas se aprende una **función de valor** entrenada *dentro* del modelo de mundo — un crítico aprendido que estima el retorno a largo plazo desde cualquier estado sin requerir un sustituto diseñado a mano. Tanto **Dreamer** (que entrena un actor-crítico enteramente en la imaginación latente) como **MuZero** (que combina MCTS con una red de valor aprendida) utilizan esta idea.

---

## Cómo conecta esto con los sistemas modernos

La receta exacta que acabas de ejecutar — **dinámica aprendida + planificación** — es la base de algunos de los sistemas de RL más potentes en la investigación moderna de IA:

- **MuZero** (DeepMind): combina un modelo de mundo aprendido con Monte Carlo Tree Search. Dominó el Go, el ajedrez, el shogi y Atari sin necesidad de conocer las reglas de antemano.
- **Dreamer / DreamerV3** (Hafner et al.): entrena una política *dentro* de un modelo de mundo de **espacio latente** (lo que significa que el modelo comprime imágenes o estados brutos en una representación abstracta y compacta antes de predecir el futuro). Logra un rendimiento de vanguardia en más de 100 benchmarks.
- **PETS / PlaNet / TD-MPC**: estas son familias de algoritmos que escalan exactamente esta idea a tareas complejas de control continuo como la robótica.

Has construido — en unos pocos cientos de líneas — el miembro más pequeño de esa familia.

---

## Palabras Clave

| Término | En lenguaje sencillo |
|------|---------------|
| **MPC** | Control Predictivo por Modelo — planear a futuro, actuar una vez, volver a planificar |
| **Random shooting** | Puntuar muchos planes aleatorios, elegir el mejor |
| **Horizonte (H)** | Cuántos pasos mira el plan hacia adelante |
| **N muestras** | Cuántos planes candidatos consideramos por paso |
| **Horizonte recesivo** | Volver a planificar en cada paso en lugar de comprometerse con un plan |
| **Recompensa sustituta / Moldeado** | Una recompensa sustituta suave que da al planificador una señal útil para optimizar |

---

## Resumen de una frase

> **Una vez que tienes un modelo de mundo, planificar es simplemente "soñar cien futuros, elegir el mejor primer paso, repetir".**

Ese es todo el secreto del RL basado en modelos.
