# A2C: El Actor y el Crítico trabajan juntos

## La Idea Principal

REINFORCE espera hasta que el juego termina por completo antes de actualizar. Eso es como un entrenador que observa todo un partido de fútbol en silencio y luego da todos los comentarios al final.

**A2C (Advantage Actor-Critic)** da retroalimentación DURANTE el juego — cada pocos pasos, el entrenador hace una pausa para decir: "¡Ese pase fue genial! ¡Esa entrada fue mala!".

Esto es mucho más rápido y eficiente.

---

## Conoce a los Dos Personajes

> **¿Qué es LunarLander?** A lo largo de este documento utilizamos el entorno **LunarLander** — una simulación física en la que controlas una pequeña nave espacial y debes aterrizarla suavemente en una plataforma de aterrizaje en la luna utilizando tres motores (izquierdo, principal y derecho). Es un estándar de referencia en el aprendizaje por refuerzo, disponible en la biblioteca Gymnasium.

### El Actor 🎭
El **Actor** es la política — decide qué acción tomar.

> "Estoy en este estado. ¿Debo encender el motor izquierdo o el derecho?"

**Ejemplo de la vida real:** El *conductor* de un coche que gira el volante y pisa los pedales.

### El Crítico 🎬
El **Crítico** estima qué tan buena es la situación actual — el valor V(s).

> "Estar en ESTE estado vale unos +150 puntos de recompensa futura total."

**Ejemplo de la vida real:** El *navegador* sentado junto al conductor, diciendo "Estamos en una buena carretera — espera llegar en 30 minutos." o "Nos dirigimos al tráfico — esto va a ser lento."

### Comparten un Cerebro
En nuestra implementación, ambos utilizan la **misma base de red neuronal**:

```
          Estado (8 números para LunarLander)
                       ↓
          ┌─────────────────────────┐
          │  Capas Compartidas      │
          │  [256 neuronas] → ReLU   │
          │  [256 neuronas] → ReLU   │
          └────────┬────────┬───────┘
                   ↓        ↓
          Cabezal del Actor  Cabezal del Crítico
          [4 salidas]        [1 salida]
          (probs de acción)  (V(s))
```

- **ReLU** (Rectified Linear Unit): una función de activación aplicada después de cada capa — devuelve `max(0, x)`, manteniendo los valores positivos y anulando los negativos. Esto permite a la red aprender patrones no lineales.
- **probs de acción**: la probabilidad de tomar cada una de las 4 acciones. El Actor toma muestras de esta distribución para elegir una acción en cada paso.

**Ejemplo de la vida real:** Un cerebro, dos trabajos — como un taxista que conduce (actor) Y sabe si la ruta es buena (crítico). ¡Compartir el cerebro significa aprender más rápido!

---

## La Ventaja: ¿Fue esto mejor de lo esperado?

Al igual que REINFORCE con línea de base, A2C calcula la **Ventaja** (Advantage):

> A(s, a) = "Resultado real" − "Lo que esperábamos"

Pero aquí, el "resultado real" proviene del **bootstrap de n-pasos** del Crítico (**bootstrapping** = usar la propia predicción del Crítico V(s) para aproximar el valor de los pasos futuros, en lugar de esperar a que termine el episodio real — como estimar tu nota del examen final a mitad del semestre usando tu calificación actual):

```
Retorno TD real: r_t + γ · r_{t+1} + γ² · r_{t+2} + ... + γⁿ · V(s_{t+n})
Ventaja A_t = Retorno TD - V(s_t)
```

**Ejemplo de la vida real:** Esperas marcar 3 goles en este partido (V(s)). Si marcas 5 goles, tu ventaja es +2. Si marcas 1 gol, tu ventaja es -2.

Ventaja positiva → "esa acción ayudó más de lo esperado → ¡hazla más!"
Ventaja negativa → "esa acción ayudó menos de lo esperado → ¡hazla menos!"

---

## ¿Por qué usar múltiples entornos paralelos?

¡Nuestro A2C utiliza **8 copias** de LunarLander ejecutándose al mismo tiempo!

**¿Por qué?** Porque las experiencias de un solo entorno están **correlacionadas** — un paso sigue de cerca al paso anterior. Esta correlación engaña a la red neuronal haciéndole creer que los patrones son más fiables de lo que son.

Con 8 entornos, cada paso da 8 experiencias independientes de situaciones muy diferentes. Esto rompe la correlación y estabiliza el entrenamiento drásticamente.

**Ejemplo de la vida real:** Para aprender sobre el clima, ¿qué es mejor?:
- Observar una ciudad durante 8 horas consecutivas (correlacionado — si hacía sol a las 2 p.m., probablemente hará sol a las 3 p.m.)
- Observar 8 ciudades simultáneamente (decorrelacionado — diferentes patrones climáticos, ¡más información!)

```
Entorno 1: [aterrizó en la luna, fuego izquierda, choque, reinicio...]
Entorno 2: [cayendo demasiado rápido, fuego ambos, flotar, aterrizar...]
Entorno 3: [inclinándose a la derecha, fuego derecha, estabilizar, aterrizar...]
...
Entorno 8: [deriva a la izquierda, fuego izquierda, estable, ...]
```

Los 8 actualizan la red simultáneamente — ¡8 veces más experiencia diversa por actualización!

---

## Actualizaciones de N-Pasos: No esperes a que termine el juego

REINFORCE espera un episodio completo (¡podrían ser 1000 pasos!).

A2C se extiende cada **n_steps = 128 pasos**:

```
Juega 128 pasos en 8 entornos
    → Obtén 128 × 8 = 1024 tuplas de experiencia
    → Calcula ventajas y retornos
    → Actualiza el Actor y el Crítico
    → Juega 128 pasos más...
```

**Ejemplo de la vida real:** Un estudiante que estudia para un examen.
- Estilo REINFORCE: Lee todo el libro de texto, LUEGO haz los exámenes de práctica.
- Estilo A2C: Lee 10 páginas, haz un cuestionario, lee 10 páginas más, haz un cuestionario...

¡Comentarios más frecuentes = aprendizaje más rápido!

---

## Tres Pérdidas Combinadas

A2C entrena con tres términos de pérdida simultáneamente:

Una **pérdida** (loss) es el número que el optimizador intenta minimizar. Una pérdida menor significa que el comportamiento actual de la red está más cerca del objetivo del entrenamiento.

### 1. Pérdida del Actor (Gradiente de la Política)
Hace que las acciones ventajosas sean más probables:
```
L_actor = -E[log π(a|s) · A(s,a)]
```
Si A > 0: aumenta la probabilidad de esa acción
Si A < 0: disminuye la probabilidad de esa acción

### 2. Pérdida del Crítico (MSE de la Función de Valor)
Hace que las predicciones de valor sean más precisas (**MSE** = Error Cuadrático Medio: eleva al cuadrado el error de predicción y saca el promedio — elevar al cuadrado penaliza más los errores grandes que los pequeños):
```
L_critic = E[(V(s) - retorno)²]
```
Como entrenar cualquier modelo de **regresión** (regresión = predecir un número continuo, aquí el retorno esperado V(s)) — minimiza el error de predicción.

### 3. Bono de Entropía (Exploración)
Evita que la política se vuelva demasiado confiada demasiado rápido:
```
L_entropy = -H[π(·|s)] = E[log π(a|s)]
```
Alta entropía = opciones de acción diversas = exploración
Baja entropía = opciones estrechas y seguras = explotación

**Ejemplo de la vida real:** El bono de entropía es como un profesor que dice "¡No te limites a adivinar la A en cada pregunta de opción múltiple! Prueba diferentes respuestas para aprender qué funciona".

```
Pérdida total = L_actor + 0.5 × L_critic - 0.01 × entropía
```

---

## LunarLander: Un Desafío más Difícil

**LunarLander-v3** es un entorno de Gymnasium (anteriormente OpenAI Gym) — "v3" es el número de versión que indica la tercera revisión de este entorno. El agente controla una pequeña nave espacial que debe aterrizar de forma segura en una plataforma designada en la luna. Es mucho más difícil que CartPole:
- Espacio de estados de 8 dimensiones (posición, velocidad, ángulo, contacto de las patas, combustible)
- 4 acciones discretas (no hacer nada, fuego izquierda, fuego principal, fuego derecha)
- Recompensa: +100 por aterrizar, -100 por chocar, pequeñas penalizaciones por combustible

La curva de entrenamiento muestra una mejora gradual desde recompensas altamente negativas hacia positivas. A2C en LunarLander requiere una experiencia significativa antes de que el aterrizador aprenda la estabilidad básica.

---

## Ecuaciones Clave

```
retorno de n-pasos:  G_t = r_t + γ·r_{t+1} + ... + γⁿ·V(s_{t+n})
Ventaja:             A_t = G_t - V(s_t)
Actualización Actor: θ_π ← θ_π - α ∇ L_actor
Actualización Crítico: θ_V ← θ_V - α ∇ L_critic
```

---

## Conclusiones Clave

| Concepto | En lenguaje sencillo |
|----------|---------------|
| **Actor** | La política — decide qué hacer |
| **Crítico** | La función de valor — juzga qué tan buena es la situación |
| **Ventaja** | "¿Fue esto mejor de lo esperado?" (real - esperado) |
| **Retorno de n-pasos** | Mira n pasos hacia el futuro antes de hacer bootstrapping con V(s) |
| **Entornos paralelos** | Múltiples entornos para una experiencia diversa y decorrelacionada |
| **Bono de entropía** | Anima al actor a seguir probando cosas nuevas |

---

## ¿Qué sigue?

A2C es genial pero tiene una debilidad importante: a veces actualiza la política de forma demasiado agresiva. Una sola actualización mala puede destruir todo el buen aprendizaje de una actualización anterior.

**PPO (Proximal Policy Optimization)** soluciona esto con un inteligente "clip de seguridad" que evita que cualquier actualización individual cambie demasiado la política.

¡Mira `ppo_scratch.py` para la implementación de PPO!
