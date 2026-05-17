# Entrenamiento de un Modelo de Mundo: Enseñar al agente a soñar 🌍

## ¿Qué es un "Modelo de Mundo"?

Un **modelo de mundo** (world model) es la *copia interna del universo* que tiene el agente. Dale un estado y una acción, y predecirá qué sucederá a continuación:

```
(estado, acción)  ──►  Red Neuronal  ──►  (siguiente_estado, recompensa)
```

No es el mundo real — es un **simulador que el agente construyó para sí mismo** observando la realidad y aprendiendo a imitarla.

Una vez entrenado, el modelo permite al agente hacerse preguntas de tipo "¿qué pasaría si...?" sin realizar ninguna acción real:

> *"Si empujo a la izquierda ahora y luego dos veces a la derecha, ¿dónde acabaré? ¿se caerá el poste?"*

El agente puede reflexionar sobre cien planes dentro de su modelo en el tiempo que le llevaría realizar un solo movimiento real. Ese es el objetivo principal.

---

## Una Analogía de la Vida Real

Piensa en cómo resuelves *tú* un rompecabezas. No mueves físicamente cada pieza en cada hueco. **Imaginas** qué pasa si la pieza A va aquí. Si esa simulación mental parece incorrecta, la rechazas antes de mover un dedo.

Tu cerebro tiene un modelo de mundo aprendido — construido tras años de ver cómo se comportan los objetos — que te permite simular resultados antes de comprometerte.

Otros ejemplos:

- **Un jugador de ajedrez** imagina jugadas varios turnos por delante.
- **Un conductor** que piensa: "Si freno ahora, el automóvil de atrás tiene suficiente espacio".
- **Un niño** apilando bloques: "Si pongo el grande arriba, la torre se tambaleará". (Aprendieron este modelo al haber derribado torres anteriormente).

En todos los casos, **un modelo mental + imaginación = mejores decisiones con menos riesgo**.

---

## ¿Cómo construye el agente su modelo?

Simplemente **observa**. Específicamente:

1. **Recopilar datos.** Dejar que cualquier política (incluso una aleatoria) interactúe con el entorno real durante un tiempo. Guardar cada transición:
   ```
   (estado, acción, recompensa, siguiente_estado)
   ```
2. **Entrenar una red neuronal** para predecir el `siguiente_estado` y la `recompensa` a partir de `(estado, acción)`. Esto es aprendizaje supervisado: cada transición guardada es un ejemplo etiquetado donde la entrada es "lo que el agente vio y hizo" y la etiqueta es "lo que realmente sucedió después".
3. **Validar.** Reservar el 10% de los datos y comprobar las predicciones del modelo frente a las reales. Un error bajo significa que el modelo ha capturado la **dinámica** del entorno: cómo cambian los estados tras las acciones.

El truco que utilizamos: en lugar de predecir el `siguiente_estado` directamente, predecimos el **delta** `siguiente_estado − estado`. La mayor parte de la física es incremental ("el carro se movió un poquito"), y los objetivos pequeños son más amables con las redes neuronales.

---

## Nuestra Configuración

| Elección | Valor | Por qué |
|--------|-------|-----|
| Entorno | `CartPole-v1` | Estado 4-D, 2 acciones — fácil de modelar |
| Datos | 20,000 transiciones de una política aleatoria | Amplia cobertura del espacio de estados |
| Red | MLP, 2 × 128 ReLU ocultas | MLP = Perceptrón Multicapa (red neuronal "vainilla" estándar). Dos capas ocultas de 128 neuronas usando activaciones ReLU. Suficiente capacidad, rápida de entrenar. |
| Pérdida | MSE en `(delta_estado, recompensa)` | MSE = Error Cuadrático Medio (promedio de los errores de predicción al cuadrado). Pérdida de regresión estándar. |
| Optimizador | Adam, lr = 1e-3, 30 épocas | Adam = optimizador adaptativo (ajusta las tasas de aprendizaje por parámetro automáticamente). Al ser estándar, no necesita ajustes especiales. |

Todo el entrenamiento finaliza en unos segundos en CPU.

---

## ¿Qué aspecto tiene un resultado "bueno"?

Importan dos diagnósticos:

### 1. Precisión de un solo paso (MSE de validación)

Esto es "¿qué tan bien predice el modelo UN paso hacia el futuro?". Después de 30 épocas deberías ver el MSE de validación en el rango de **1e-4 a 1e-3**. Eso es minúsculo — los ángulos del poste y las posiciones del carro son precisos hasta unos pocos decimales.

### 2. **Error acumulado** en rollouts de k-pasos

Esta es la prueba *real*. Toma un estado, pásalo por el modelo, luego toma su predicción y vuelve a pasarla por el modelo — durante `k` pasos seguidos. El error crece porque cada paso añade un poco de ruido sobre la predicción anterior.

```
Paso  1:  Error L2 ≈ 0.01   (casi perfecto)
Paso  5:  Error L2 ≈ 0.05
Paso 10:  Error L2 ≈ 0.15
Paso 20:  Error L2 ≈ 0.40   (deriva visible)
```

*(Error L2 = distancia euclidiana entre el siguiente estado predicho y el real — piénsalo como "¿cuánto se aleja la suposición del modelo en el espacio de estados 4-D?")*

**Por qué esto importa.** Si planeamos 15 pasos por delante con el modelo, el estado *exacto* en el paso 15 será incorrecto — pero si se mantiene la clasificación relativa de "planes buenos frente a planes malos", la planificación sigue funcionando. (Esto es lo que aprovecha `model_based_planning.py`).

El gráfico en `outputs/world_model.png` muestra ambos diagnósticos lado a lado: la curva de pérdida de entrenamiento baja agradablemente en una escala logarítmica, y la curva de error de rollout sube agradablemente.

---

## ¿Por qué predecir el *Delta*?

Compara dos formas de plantear el mismo problema a la red:

| Objetivo | Magnitud típica | ¿Fácil o difícil? |
|--------|------------------:|--------------|
| `siguiente_estado` | 0–2.4 (pos carro) | La red debe reproducir la posición **y** el cambio minúsculo |
| `siguiente_estado - estado` | ~0.02 | La red solo aprende el cambio minúsculo |

Predecir el delta también significa: si la red devuelve ceros (como suele hacer una red principiante no entrenada), la predicción es simplemente "nada se movió" — un valor por defecto sensato y seguro para un solo paso de tiempo. En cambio, predecir el `siguiente_estado` absoluto directamente daría inicialmente valores basura completamente aleatorios, haciendo que el entrenamiento temprano fuera muy inestable.

---

## Qué nos aporta esto

Un modelo de mundo entrenado es la base de:

- **Planificación**: búsqueda sobre secuencias de acciones imaginadas (ver `model_based_planning.py`).
- **Aumentación tipo Dyna**: entrenar una red Q con datos imaginados para multiplicar la eficiencia de las muestras.
- **Curiosidad / exploración**: visitar estados que el modelo no puede predecir bien.
- **Artículos Dreamer / World-Models**: entrenar una *política* enteramente dentro del modelo con cero interacción con el mundo real más allá de la recopilación inicial de datos.

---

## Límites y Precauciones

- **Deriva fuera de la distribución.** El modelo solo conoce la parte del mundo que ha visto. Planifica de forma demasiado agresiva y acabarás en regiones que el modelo nunca ha visitado — las predicciones allí son pura fantasía.
- **Error acumulado.** La planificación sobre **horizontes** largos (muchos pasos en el futuro) no es fiable debido a la acumulación de errores, como muestra el gráfico. Los sistemas modernos abordan esto utilizando **conjuntos (ensembles) probabilísticos** (entrenando múltiples modelos y comprobando si están de acuerdo, como en PETS o Dreamer) para que el planificador sepa exactamente *qué tan incierto* es el modelo en cada paso y pueda evitar caminos arriesgados y desconocidos.
- **Entornos estocásticos.** Un regresor determinista estándar predice solo el resultado *medio* y se pierde por completo la dispersión de los resultados posibles. Los entornos complejos del mundo real requieren modelos probabilísticos (como los que tienen salidas Gaussianas, o **modelos estocásticos latentes** — redes que codifican el estado del mundo como una distribución de probabilidad en un espacio comprimido, permitiéndoles capturar la aleatoriedad genuina en lugar de promediarla) para representar con precisión la incertidumbre y la aleatoriedad.

---

## Palabras Clave

| Término | En lenguaje sencillo |
|------|---------------|
| **Modelo de mundo** | Una red neuronal que imita al entorno. |
| **Dinámica (Dynamics)** | La función `(s, a) → s'`. |
| **Modelo de recompensa** | La función `(s, a) → r` (a menudo incluida). |
| **Predicción de un paso** | Lo que el modelo devuelve a partir de un estado real. |
| **Rollout** | Predicciones repetidas de un paso, realimentando las salidas. |
| **Error acumulado** | Pequeños errores que crecen a lo largo de un rollout. |

---

## Resumen de una frase

> **Un modelo de mundo es una pequeña copia neuronal del universo a la que el agente puede consultar — y dentro de la cual puede soñar — antes de arriesgarse a una acción real.**

A continuación: `model_based_planning.py` pone a trabajar este modelo para la toma de decisiones real.
