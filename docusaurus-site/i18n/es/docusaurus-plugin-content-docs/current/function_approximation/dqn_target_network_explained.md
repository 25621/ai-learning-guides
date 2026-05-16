# Red Objetivo (Target Network): Estabilizando el centro de la diana 🎯

## El problema de la meta móvil

Imagina que intentas dar en el centro de una diana con un arco y una flecha. Disparas, miras dónde ha aterrizado tu flecha y ajustas tu puntería para la próxima vez. Sencillo, ¿verdad?

¡Ahora imagina que la diana se MUEVE cada vez que disparas! Cada flecha que disparas cambia ligeramente el lugar donde estará la diana para el siguiente disparo. Nunca mejorarías — estarías persiguiendo un objetivo que siempre se escapa.

¡Ese es exactamente el problema de DQN sin una red objetivo!

---

## Por qué los objetivos Q (Q-Targets) se siguen moviendo

En DQN, el objetivo para cada actualización es:
> objetivo = recompensa + γ × max(Q(siguiente_estado))

Aquí **γ (gamma)** es el **factor de descuento** — un número entre 0 y 1 (normalmente 0.99) que controla cuánto le importan al agente las recompensas *futuras* frente a las *inmediatas*.

**Ejemplo de la vida real:** Imagina que alguien te ofrece una galleta ahora, o dos galletas mañana. Si realmente quieres galletas ahora, tu γ es bajo (descuentas mucho el futuro). Si eres paciente y feliz de esperar, tu γ es alto (las recompensas futuras importan casi tanto como las de ahora). En RL, γ = 0.99 significa "una recompensa en el siguiente paso vale el 99% de una recompensa ahora mismo".

Los valores Q del lado derecho provienen de... ¡la misma red que estamos entrenando!

Así que cada vez que actualizamos la red (para mejorar los valores Q), también cambiamos los objetivos. Es un bucle de retroalimentación:

1. Actualizar la red → los valores Q cambian.
2. Los valores Q cambian → los objetivos cambian.
3. Los objetivos cambian → actualizar la red de forma diferente.
4. Repetir para siempre — ¡inestable!

**Ejemplo de la vida real:** Intentar pesarte en una báscula que cambia sus lecturas cada vez que te subes a ella. ¡Nunca sabrías tu peso real!

---

## La Solución: ¡Congelar la diana! ❄️

La **Red Objetivo (Target Network)** es una COPIA de la red Q principal que se queda congelada.

- **Red online** (`qnet`): Se actualiza en cada paso del entrenamiento — aprende rápido.
- **Red objetivo** (`target_net`): Copia congelada — solo se actualiza cada 100 pasos.

Utilizamos el objetivo CONGELADO para calcular los objetivos:
> objetivo = recompensa + γ × max(Q_TARGET(siguiente_estado))

¡El objetivo se mantiene quieto durante 100 pasos! Eso le da a la red online una meta estable a la que apuntar. Luego copiamos los pesos de la online a la objetivo, volvemos a congelar y repetimos.

**Ejemplo de la vida real:** Piensa en un estudiante y un profesor. El profesor da deberes (el objetivo). El estudiante aprende y mejora. Después de 100 lecciones, el profesor ACTUALIZA los deberes para que sean más difíciles. El profesor no cambia cada minuto — ¡eso sería demasiado caótico!

---

## La receta completa de DQN 🍕

El algoritmo DQN completo (búfer de repetición + red objetivo) es:

```
1. Inicializar la red online Q y la red objetivo Q_target (mismos pesos)
2. Crear el búfer de repetición (caja de memoria)

Cada paso del entorno:
  a. Elegir acción usando ε-greedy con Q
  b. Guardar (estado, acción, recompensa, siguiente_estado) en el búfer

Cada 4 pasos:
  c. Muestrear un mini-lote aleatorio del búfer
  d. Calcular objetivos usando Q_TARGET (¡congelado!)
  e. Actualizar Q para minimizar la pérdida (loss)

Cada 100 pasos:
  f. Copiar pesos de Q → Q_TARGET (sincronizar objetivo)
```

¡Este es el algoritmo exacto del artículo de DQN de DeepMind (2015)!

---

## Qué muestra la comparación

Cuando ejecutes `dqn_target_network.py`, verás:

**Sin red objetivo (solo DQN + replay):**
- El entrenamiento puede estar "bien" pero con colapsos periódicos.
- Los valores Q pueden divergir (explotar u oscilar).
- El aprendizaje es menos predecible.

**DQN completo (replay + red objetivo):**
- Aprendizaje ascendente más constante.
- Los valores Q se mantienen en un rango razonable.
- Convergencia más rápida al umbral de resolución (195+ en CartPole).

---

## La "Tríada Mortal" ☠️

En el aprendizaje por refuerzo, la combinación de tres cosas crea inestabilidad:

1. **Aproximación de funciones** (red neuronal en lugar de tabla) ← usamos esto.
2. **Bootstrapping** (usar valores Q para estimar valores Q) ← usamos esto.
3. **Aprendizaje fuera de la política (off-policy)** (Q-learning usa max, no la política real) ← usamos esto.

Las tres juntas = la "tríada mortal". DQN doma esto con:
- Búfer de repetición → rompe las correlaciones.
- Red objetivo → rompe el bucle de retroalimentación.

No resuelve completamente el problema, ¡pero lo hace manejable!

---

## Vocabulario Clave

| Palabra | Significado |
|------|---------|
| **Red Objetivo (Target Network)** | Una copia congelada de la red Q utilizada solo para calcular los objetivos |
| **Red Online** | La red Q que se está entrenando activamente |
| **Sincronización (Sync)** | Copiar los pesos de la red online a la red objetivo |
| **Bucle de retroalimentación** | Cuando la salida de un sistema se retroalimenta para cambiar la entrada (puede causar inestabilidad) |
| **Tríada Mortal** | La combinación de aproximación de funciones + bootstrapping + off-policy que causa inestabilidad |

---

## Impacto en el mundo real

En 2015, DeepMind publicó su artículo sobre DQN mostrando una IA que podía jugar a 49 juegos de Atari a nivel sobrehumano — utilizando SOLO estos dos trucos (búfer de repetición + red objetivo).

Antes de esto, la gente pensaba que no se podían entrenar redes neuronales con RL debido a la inestabilidad. ¡DeepMind demostró que estaban equivocados e inició la revolución del RL profundo!

A continuación, aplicaremos esta receta completa de DQN a Atari Pong — ¡un videojuego real con píxeles brutos como entrada!
