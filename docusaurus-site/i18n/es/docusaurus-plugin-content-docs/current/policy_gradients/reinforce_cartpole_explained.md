# REINFORCE: Enseñando a un robot a tomar mejores decisiones

## ¿Qué intentamos hacer?

Imagina que tienes un robot jugando a un videojuego. Cada segundo, el robot debe elegir: **"¿Debo pulsar el botón o no?"**.

En lugar de memorizar cada situación en una tabla (como en el Q-learning), queremos que el robot aprenda una **receta** — un conjunto de reglas que diga directamente: "En esta situación, haz esta acción".

Esta receta se llama **política** (π, pi). En el aprendizaje por refuerzo, π significa "la regla para elegir acciones".

---

## La forma antigua frente a la forma nueva {#the-old-way-vs-the-new-way}

**Forma antigua (Q-learning / DQN)**: Aprender qué tan BUENA es cada acción (valores Q) y luego elegir la mejor.
> "Pulsar IZQUIERDA tiene una puntuación de 7, pulsar DERECHA tiene una puntuación de 5 → ¡pulsa IZQUIERDA!".

**Forma nueva (Gradiente de Política)**: Aprender directamente qué acción ELEGIR.
> "Cuando el poste se incline a la derecha, pulsa DERECHA con un 80% de probabilidad, pulsa IZQUIERDA con un 20% de probabilidad".
*(La palabra **Gradiente** se refiere al "paso" matemático que damos para ajustar lentamente estas probabilidades en la dirección correcta).*

**Ejemplo de la vida real:** Aprender a montar en bicicleta.
- La forma antigua: calcular la *puntuación exacta* para "inclinarse a la izquierda 5 grados" frente a "inclinarse a la izquierda 7 grados".
- La forma nueva: simplemente practicar hasta que tu **cuerpo** aprenda — ¡empuja el pie que sientas que es el correcto!

---

## ¿Cómo funciona REINFORCE?

REINFORCE observa al robot jugar una partida completa de principio a fin (un **episodio**), y luego pregunta: "¿Qué acciones llevaron a una buena puntuación? ¡Hagamos más de esas!".

### Paso a paso

**1. Jugar un episodio**

El robot toma decisiones y recopila experiencia:
```
Paso 1: Estado = [poste inclinándose a la derecha] → Acción = pulsar DERECHA → Recompensa = +1
Paso 2: Estado = [poste casi equilibrado] → Acción = pulsar DERECHA → Recompensa = +1
Paso 3: Estado = [poste inclinándose a la izquierda] → Acción = pulsar IZQUIERDA → Recompensa = +1
...
Paso 47: Estado = [¡el poste se cayó!] → Episodio terminado
```

**2. Calcular los retornos**

Para cada paso, calcular G_t — la **recompensa total desde ese momento en adelante**:
```
G_en_paso_47 = 1
G_en_paso_46 = 1 + 0.99 × 1 = 1.99
G_en_paso_45 = 1 + 0.99 × 1.99 = 2.97
...
G_en_paso_1  = 47 (aproximadamente — retorno más alto porque fue desde el principio)
```

El **factor de descuento** γ = 0.99 significa que las recompensas lejanas en el futuro cuentan un poco menos.

**Ejemplo de la vida real:** Recibir una estrella dorada el primer día de clase resulta más emocionante que saber que *podrías* recibir una el día 100. Las recompensas futuras se "descuentan" ligeramente.

**3. Actualizar la política**

Por cada acción realizada:
> Si G_t fue ALTO (esa acción llevó a un gran resultado): **¡hazla más!**
> Si G_t fue BAJO (esa acción llevó a un mal resultado): **¡hazla menos!**

La matemática: `pérdida = -log_prob(accion) × G_t`

Tomar el gradiente y actualizar la política es como decirle al robot:
*"¿Esa acción que tomaste en el paso 20? ¡Deberías hacerla un 5% más a menudo la próxima vez!"*.

---

## ¿Qué es una Red de Política?

En lugar de una tabla, utilizamos una **red neuronal** para representar la política.

```
Observación        Red de Política       Probabilidades de Acción
[pos carro]   →   [128 neuronas]   →  →  [pulsar IZQUIERDA: 30%]
[vel carro]   →   [128 neuronas]         [pulsar DERECHA: 70%]
[ángulo poste] →
[vel poste]    →
```

La red genera **probabilidades** para cada acción. Luego tomamos una muestra:
> Tirar un dado → 1-30: pulsar IZQUIERDA, 31-100: pulsar DERECHA.

**Ejemplo de la vida real:** Una aplicación del tiempo dice "70% de probabilidad de lluvia". No SABES que lloverá — decides basándote en la probabilidad. ¡El robot hace lo mismo!

---

## Normalización: Por qué restamos la media

Antes de usar G_t para actualizar, normalizamos:
```
G_normalizado = (G - media(G)) / std(G)
```

**¿Por qué?** Imagina que todas las recompensas son positivas (como ocurre en CartPole — siempre +1 por paso). Sin normalización, CUALQUIER acción parece "buena" y la señal de actualización resulta confusa.

Después de la normalización, algunos retornos son positivos (por encima de la media → pulsar más) y otros son negativos (por debajo de la media → pulsar menos). ¡La señal se vuelve mucho más limpia!

**Ejemplo de la vida real:** Tu profesor califica con una curva. Si la puntuación media es 70 y tú has sacado 85, ¡eso es genial! Pero si la media es 90 y has sacado 85, eso está por debajo de la media. La puntuación bruta por sí sola no cuenta toda la historia.

---

## El Problema: Alta Varianza

REINFORCE tiene una gran debilidad: la **varianza**. Los retornos G_t son muy ruidosos.

**Ejemplo de la vida real:** Imagina juzgar a un chef probando solo UNA comida de cada restaurante. A veces el chef tuvo un mal día, a veces los ingredientes no eran los mejores. ¡Una comida no es suficiente para saber de forma fiable si el restaurante es bueno!

REINFORCE espera a un episodio COMPLETO antes de actualizar. Un episodio puede tener mucha suerte y otro muy mala. Los gradientes saltan por todas partes.

Por eso la curva de aprendizaje (en el gráfico) se ve dentada:
- Algunas ejecuciones llegan a 500 (¡increíble!).
- Otras caen a 50 (¡terrible!).

A pesar del ruido, REINFORCE acaba aprendiendo, pero requiere paciencia.

---

## Los Resultados

```
Episodio  100 | Recompensa media (últimos 100):  43.1
Episodio  200 | Recompensa media (últimos 100): 193.9
Episodio  500 | Recompensa media (últimos 100): 408.4
Episodio 1000 | Recompensa media (últimos 100): 500.0  ← ¡Resuelto!
```

El robot aprende a equilibrar el poste durante el máximo de 500 pasos — ¡RESUELTO!

A pesar de sus problemas de varianza, REINFORCE en CartPole es efectivo porque:
1. Los episodios son cortos (así que obtenemos muchos por cada ejecución de entrenamiento).
2. La política óptima es sencilla (pulsar principalmente en la dirección en la que se inclina el poste).

---

## Conclusiones clave

| Concepto | En lenguaje sencillo |
|----------|---------------|
| **Política** | La receta del robot para elegir acciones. |
| **Episodio** | Una partida completa de principio a fin. |
| **Retorno G_t** | Recompensa futura total desde este momento. |
| **Descuento γ** | Las recompensas futuras cuentan un poco menos que las inmediatas. |
| **Normalizar** | Restar la media para que la señal sea más limpia. |
| **Varianza** | Cuánto saltan las estimaciones del gradiente. |

---

## ¿Qué sigue?

La gran debilidad de REINFORCE es la **varianza**. En el siguiente script (`reinforce_baseline.py`), añadimos una **línea de base** (baseline) que reduce drásticamente este ruido — sin cambiar lo que el algoritmo aprende de media.

La idea clave: en lugar de preguntar "¿fue buena esta acción?", preguntamos "¿fue esta acción **mejor de lo esperado**?". Ese pequeño cambio hace que el aprendizaje sea mucho más estable.
