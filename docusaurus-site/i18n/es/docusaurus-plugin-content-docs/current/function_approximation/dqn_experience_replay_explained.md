# Búfer de Repetición (Experience Replay): Enseñando al robot a recordar 🎒

## El Problema: El Olvido (y la Confusión)

¿Recuerdas cómo el DQN ingenuo era inestable? La razón principal es el **aprendizaje correlacionado**.

Cuando el robot juega, experimenta las cosas en orden:
> Paso 1 → Paso 2 → Paso 3 → Paso 4 → ...

¡Estos pasos están conectados! Si el robot se inclina a la izquierda en el paso 10, en el paso 11 también se inclinará a la izquierda. No son independientes, dependen unos de otros.

Cuando actualizamos la red utilizando estos pasos correlacionados, es como intentar aprender historia leyendo el mismo capítulo una y otra vez. ¡Te volverías muy bueno en ese capítulo y olvidarías todo lo demás!

**Ejemplo de la vida real:** Imagina estudiar para un examen practicando solo la tarea de ayer. Te vuelves increíble en esos problemas exactos, ¡pero el examen tiene preguntas diferentes! Necesitas practicar una MEZCLA de problemas diferentes.

---

## La Solución: Una Caja de Recuerdos 📦

El **Búfer de Repetición (Experience Replay)** añade una gran caja de memoria (el **búfer de repetición**) al robot.

En lugar de aprender de la experiencia más reciente, el robot:
1. **Almacena** cada experiencia en la caja de recuerdos: (estado, acción, recompensa, siguiente estado).
2. **Elige al azar** un puñado de recuerdos de la caja.
3. **Aprende de esa mezcla aleatoria** en lugar de solo del último paso.

```
Paso del juego 1 → [guardar en la caja]
Paso del juego 2 → [guardar en la caja]
Paso del juego 3 → [guardar en la caja]
...
Paso del juego 50 → [guardar en la caja] → elegir 64 recuerdos al azar → actualizar red
Paso del juego 51 → [guardar en la caja] → elegir 64 recuerdos al azar → actualizar red
```

**Ejemplo de la vida real:** Piensa en un álbum de fotos. No aprendes sobre tu vida mirando solo las fotos de hoy. También pasas las fotos ANTIGUAS — una mezcla de buenos recuerdos y momentos difíciles. Esto te ayuda a entender los patrones de toda tu vida, no solo los de hoy.

---

## Por qué ayuda el muestreo aleatorio

Al elegir los recuerdos al azar, rompemos las correlaciones. El robot podría aprender de:
- Un recuerdo donde el poste estaba perfecto (de hace 500 pasos).
- Un recuerdo donde el poste estaba a punto de caer (de hace 20 pasos).
- Un recuerdo donde tuvo suerte (del paso 3).

Esta mezcla aleatoria significa:
✅ El robot aprende de una variedad de situaciones.
✅ Cada recuerdo puede ser "reproducido" muchas veces (uso eficiente de la experiencia).
✅ La red no se sobreajusta a los eventos recientes.

---

## Aprendizaje por Mini-Lotes (Mini-Batch)

En lugar de actualizar con UNA experiencia cada vez, actualizamos con **64 experiencias a la vez** (un "mini-lote"). Esto es como:
- Forma antigua: Lee una tarjeta de memoria, ponte a prueba.
- Forma nueva: Lee 64 tarjetas de memoria diferentes, luego ponte a prueba con la mezcla.

Los mini-lotes hacen que la señal de aprendizaje sea mucho más fiable y menos ruidosa.

---

## Período de Calentamiento (Warmup)

¡No empezamos a aprender de inmediato! El búfer de repetición necesita primero algunos recuerdos. Esperamos hasta que haya al menos **500 experiencias** en la caja antes de que comience el entrenamiento.

**Ejemplo de la vida real:** No intentarías cocinar una comida hasta que hayas reunido los ingredientes. ¡El período de calentamiento es como ir a comprar antes de cocinar!

---

## Qué muestra la comparación

Cuando ejecutes `dqn_experience_replay.py`, verás dos curvas de aprendizaje:

| Naive DQN | DQN + Replay |
|-----------|-------------|
| Muy irregular | Más suave |
| Fallos frecuentes (olvida todo) | Mejora más constante |
| Alta varianza | Menor varianza |

La versión con búfer de repetición suele:
- Alcanzar buenas puntuaciones de forma más fiable.
- No caer de 500 a 30 con tanta frecuencia.
- Mostrar un progreso de aprendizaje más estable.

---

## El Búfer de Repetición en código

```
ReplayBuffer:
  - capacidad: 10,000 recuerdos (los más viejos se olvidan cuando está lleno)
  - push(estado, accion, recompensa, siguiente_estado, terminado)
  - sample(batch_size=64) → lote aleatorio
```

Piensa en él como un cuaderno con 10,000 líneas. Cuando está lleno, borras la línea más antigua y escribes la más nueva. ¡Siempre estudias de una página al azar!

---

## Vocabulario Clave

| Palabra | Significado |
|------|---------|
| **Experience Replay** | Almacenar y reutilizar aleatoriamente experiencias pasadas para el entrenamiento |
| **Replay Buffer** | La caja de memoria que almacena tuplas de (estado, acción, recompensa, siguiente_estado) pasadas |
| **Actualizaciones correlacionadas** | Cuando los datos de entrenamiento dependen de sí mismos (¡malo para el aprendizaje!) |
| **Mini-lote (Mini-batch)** | Una pequeña muestra aleatoria de recuerdos utilizada para un paso de actualización |
| **Decorrelación** | Romper las conexiones entre experiencias consecutivas |

---

## ¿Qué falta todavía?

Incluso con un búfer de repetición, hay otro problema: el **objetivo móvil**.

Cada vez que actualizamos la red, los valores Q cambian. Pero esos valores Q actualizados TAMBIÉN se utilizan para calcular el objetivo de la PRÓXIMA actualización. ¡Es un círculo de confusión!

Esto se resuelve mediante la **Red Objetivo (Target Network)** — una copia congelada de la red que solo se desarrolla cada 100 pasos. ¡Eso hace que el "centro de la diana" se mantenga quieto durante un tiempo para que el robot pueda apuntar de forma fiable!
