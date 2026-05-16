# DPO: Saltarse al juez y acudir directamente a la fuente

## La Gran Idea

El RLHF clásico tiene dos etapas: primero entrenar un modelo de recompensa y luego usar PPO para perseguir sus puntuaciones. DPO (Direct Preference Optimization) plantea una pregunta ingeniosa:

*Si el modelo de recompensa es solo un paso intermedio, ¿podemos saltárnoslo?*

Resulta que sí. DPO entrena el modelo de lenguaje directamente a partir de pares de preferencias, sin un juez separado, sin un bucle de muestreo PPO y sin un coeficiente KL que ajustar. Utiliza una fórmula elegante y se comporta como el aprendizaje supervisado.

Esto hace que DPO sea más sencillo de ejecutar, más estable y más rápido, razón por la cual se ha convertido rápidamente en la opción predeterminada para muchos modelos de código abierto alineados.

## Una analogía de la vida real

Supongamos que estás entrenando a un estudiante para escribir ensayos.

El enfoque PPO es: contratar a un profesor para que califique los ensayos, luego hacer que el estudiante escriba ensayo tras ensayo y ajustarlos en función de las notas del profesor.

El enfoque DPO es: mostrar al estudiante dos ensayos a la vez y decirle: "este es mejor; inclínate por escribir como este y aléjate de aquel". Sin profesor de por medio. El estudiante se ajusta directamente a partir de las comparaciones.

Ambos enfoques pueden funcionar. DPO suele terminar más rápido porque nadie tiene que entrenar y mantener a un profesor separado.

## Cómo funciona el aprendizaje (solo intuición)

DPO utiliza los mismos pares de preferencias que el modelado de recompensas: prompt (instrucción), elegido (chosen) y rechazado (rejected). Para cada par, se plantea dos preguntas:

1. ¿Se ha vuelto el modelo **más propenso** a generar la respuesta elegida de lo que lo habría sido el modelo de referencia?
2. ¿Se ha vuelto el modelo **menos propenso** a generar la respuesta rechazada de lo que lo habría sido el modelo de referencia?

El entrenamiento empuja ambos números en la dirección correcta a la vez. Crucialmente, el modelo de referencia siempre está presente en la comparación — desempeña el mismo papel que la penalización KL en PPO. Se permite que el modelo cambie, pero los cambios son siempre *relativos al* punto de partida.

Un resultado sutil y hermoso del artículo de DPO es que esta función de pérdida única es matemáticamente equivalente a "entrenar un modelo de recompensa y luego ejecutar PPO con una penalización KL". Mismo destino, viaje más sencillo.

## Qué muestra el experimento

Entrenamos una política directamente en 2,000 pares de preferencias durante 300 épocas.

![Entrenamiento DPO](outputs/dpo_implementation.png)

- **Izquierda**: la pérdida de DPO cae a medida que el modelo aprende a preferir las respuestas elegidas sobre las rechazadas.
- **Centro**: la precisión de preferencia (con qué frecuencia la política asigna una recompensa implícita más alta a la respuesta elegida) sube hasta aproximadamente el 99%.
- **Derecha**: el margen de recompensa implícita crece. DPO nunca nombra una "recompensa", pero la brecha entre las log-probabilidades de lo elegido frente a lo rechazado, escalada por beta, puede interpretarse como tal. Se ensancha constantemente, lo que significa que el modelo adquiere más confianza en sus preferencias.

Observa lo limpio que se ve esto en comparación con PPO. No hay bucle de muestreo, no hay ruido de exploración y no hay un modelo de recompensa separado ejecutándose. Cada época es una actualización pura de estilo supervisado sobre el conjunto de datos de preferencias.

## Dónde encaja esto en el pipeline de RLHF

DPO es una *alternativa* al paso dos del pipeline clásico:

- **Clásico**: preferencias → modelo de recompensa → PPO → modelo alineado.
- **DPO**: preferencias → modelo alineado. (Listo).

La pega es que DPO se entrena sobre un conjunto de datos de preferencias fijo. PPO, debido a que toma muestras de respuestas frescas en cada ronda, puede, en principio, explorar más allá. En la práctica, DPO gana para la mayoría de los casos de uso de "alineación sobre un conjunto de datos de preferencias curado".

## Por qué esto importa fuera del laboratorio

El patrón de "saltarse la medición intermedia" está en todas partes:

- Un entrenador que corrige la forma de un nadador mediante demostraciones comparativas en lugar de cronometrar cada vuelta.
- un fotógrafo que edita dos fotos a la vez, eligiendo la mejor, en lugar de crear una rúbrica de puntuación para una "buena foto".
- Un responsable de contratación que compara dos currículums en lugar de puntuar cada uno con una lista de verificación de 30 puntos.

Cuando solo necesitas *clasificar*, no necesitas una escala absoluta. DPO es esa idea aplicada a los modelos de lenguaje.

## Resumen de una frase

**DPO convierte los pares de preferencias directamente en un modelo mejor, sin un modelo de recompensa de por medio: más sencillo que PPO y, a menudo, igual de bueno.**
