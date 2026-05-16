# Tareas de Horizonte Largo (Long-Horizon Tasks)

## La Gran Idea: Cuando la recompensa está muy lejos

Imagina que eres un chef que intenta aprender una nueva receta puramente probando el plato final. Sigues 40 pasos — picar, saltear, sazonar, hervir, emplatar — pero solo recibes retroalimentación al final: "Demasiado salado". ¿Cuál de los 40 pasos causó el problema? No tienes ni idea.

Este es el **problema del horizonte largo**: cuando la señal de recompensa está separada de las decisiones que la causaron por docenas (o cientos) de pasos, el aprendizaje se vuelve muy difícil.

---

## Por qué los agentes planos tienen dificultades

Un agente de RL plano (como los agentes DQN de la Fase 3) intenta aprender el valor de cada paso individual todo a la vez. En tareas cortas — equilibrar un poste, evitar una pared — esto funciona bien. La recompensa llega rápido y el agente puede conectar causa y efecto.

Pero en una tarea larga — recoger una llave, luego usarla para abrir una puerta, luego salir del laberinto — el agente debe:

1. Tropezar con la llave (¡suerte!).
2. Recordar que recoger llaves es útil.
3. Tropezar con la puerta (¡suerte otra vez!).
4. Conectar toda la secuencia con la única recompensa a la salida.

Con la exploración aleatoria, la probabilidad de completar accidentalmente toda esta secuencia se reduce exponencialmente con cada nuevo paso requerido. El DQN plano esencialmente necesita tener suerte muchas, muchas veces antes de ver una sola recompensa positiva de la que aprender.

---

## La Solución Jerárquica: Divide y Vencerás

El RL jerárquico divide la tarea larga en una **estructura de dos niveles**:

| Nivel | Nombre | Función |
|-------|--------|-----|
| Alto | **Gestor (Manager)** | Elige el siguiente subobjetivo |
| Bajo | **Trabajador (Worker)** | Navega hasta ese subobjetivo |

Así es exactamente como los humanos abordan las tareas complejas. No planeas tu viaje por carretera giro a giro antes de salir. En su lugar:

- **Gestor (tú, en casa)**: "Primera parada: la gasolinera. Siguiente parada: la entrada de la autopista. Luego: salida 42".
- **Trabajador (tú, conduciendo)**: Se encarga de todas las decisiones individuales del volante para llegar a cada parada.

El gestor piensa en *puntos de control*. El trabajador piensa en el *volante*.

---

## Por qué esto supera al aprendizaje plano en tareas largas

El trabajador solo necesita alcanzar el *siguiente subobjetivo* — una tarea corta con una recompensa clara y cercana. Recibe retroalimentación rápidamente y aprende de manera eficiente.

El gestor solo necesita decidir el *orden de los subobjetivos* — un problema mucho más simple que planificar cada paso individual.

Juntos, los dos niveles dividen el difícil problema de horizonte largo en dos problemas fáciles de horizonte corto.

---

## El experimento de la cuadrícula Llave-Puerta

Nuestro script prueba ambos enfoques en una **cuadrícula abierta de 9x9** con dos objetos:

- Una **LLAVE** en una esquina (debe recogerse primero).
- Una **PUERTA** en la esquina opuesta (solo cuenta si tienes la llave).

La única recompensa real es +1 cuando el agente llega a la puerta *después* de recoger la llave. Esa única recompensa requiere que dos subtareas secuenciales se encadenen correctamente.

Dos agentes compiten:

**DQN Plano**: Debe tropezar con ambas subtareas en el orden correcto por accidente, y luego propagar una señal a través de ambas. Debido a que el éxito requiere dos hallazgos afortunados en un episodio, el DQN rara vez aprende algo útil.

**Agente Jerárquico**:
- Regla del gestor: "Ve primero a la llave, luego ve a la puerta".
- El trabajador recibe **+1 cada vez que alcanza un subobjetivo**, ya sea la llave o la puerta.
- Dos tareas cortas separadas, cada una con una recompensa clara y cercana.

---

## Qué muestran los gráficos

![Resultados de Tareas de Horizonte Largo](outputs/long_horizon_tasks.png)

**Izquierda — Tasa de éxito a lo largo del tiempo**: El agente jerárquico (azul) aprende a resolver el laberinto mucho antes que el DQN plano (rojo). El agente plano puede acabar aprendiendo también — con suficientes episodios —, pero el agente jerárquico llega más rápido porque su señal de aprendizaje es densa y local.

**Derecha — Rendimiento final**: El gráfico de barras muestra la tasa de éxito promediada sobre los últimos 500 episodios. La ventaja del agente jerárquico es clara: dividir el problema en subobjetivos lo hace abordable.

---

## Dónde aparece el pensamiento de horizonte largo

| Dominio | Ejemplo de horizonte largo |
|--------|---------------------|
| Robótica | Montar un dispositivo con 30 piezas en orden |
| Juegos | Ganar una partida de ajedrez (muchos movimientos, un ganador) |
| Lenguaje | Escribir un artículo de investigación completo (muchas decisiones de escritura, una puntuación de calidad) |
| Ciencia | Realizar un experimento de varios meses y evaluar los resultados |

Esta es exactamente la razón por la que se inventaron las Redes Feudales (una arquitectura donde los gestores establecen objetivos direccionales para los trabajadores de nivel inferior) y HIRO (RL Jerárquico con subobjetivos) — a medida que el RL plano chocaba con muros en estos problemas, la descomposición jerárquica se convirtió en la estrategia dominante.

---

## La conexión con las políticas condicionadas a objetivos

Observa que el **trabajador** en nuestro agente jerárquico es esencialmente una **política condicionada a objetivos** — recibe un subobjetivo y navega hacia él. Este es el diseño estándar en HIRO y artículos relacionados: el gestor establece objetivos, el trabajador es una política condicionada a objetivos que los persigue.

Las dos ideas —políticas condicionadas a objetivos y estructura jerárquica— son, por tanto, las dos caras de la misma moneda, por lo que aparecen juntas en este módulo.

---

## Resumen de una frase

> **Las tareas de horizonte largo son difíciles porque la recompensa llega demasiado tarde para enseñar decisiones individuales — el RL jerárquico resuelve esto insertando subobjetivos cercanos que permiten al trabajador aprender rápido mientras el gestor maneja la secuencia general.**
