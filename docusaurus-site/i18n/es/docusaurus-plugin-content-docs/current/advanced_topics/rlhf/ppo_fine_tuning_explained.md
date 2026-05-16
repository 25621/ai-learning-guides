# Ajuste Fino con PPO: Pulir un modelo sin romperlo

## La Gran Idea

Una vez que tenemos un modelo de recompensa que puntúa las respuestas, queremos que nuestro modelo de lenguaje produzca respuestas con puntuaciones más altas. PPO (Proximal Policy Optimization) hace exactamente esto, pero añade un cinturón de seguridad para que el modelo no persiga la puntuación y olvide cómo escribir texto normal.

Piénsalo como un paso de pulido. El modelo ya habla con fluidez; simplemente le damos un empujón para que hable de una manera que el modelo de recompensa premie, manteniendo su voz reconocible.

## Una analogía de la vida real

Imagina a un chef que ya cocina bien pero que ahora está aprendiendo a complacer a un crítico gastronómico específico.

Después de cada plato, el crítico da una puntuación. El chef tiene dos presiones:

1. **Obtener una puntuación más alta.** Cocinar de la manera que le gusta al crítico.
2. **No volverse irreconocible.** Si el chef abandona su propio estilo por completo — echando sal a tazas solo por perseguir una puntuación — la comida se vuelve extraña. Los clientes dejan de venir.

PPO captura ambas presiones:

- La parte de la **recompensa** empuja al modelo hacia las respuestas que le gustan al juez.
- La parte de la **penalización KL** tira del modelo hacia atrás, hacia cómo hablaba antes de que comenzara el entrenamiento. KL es solo una forma de medir "qué tan diferente es el nuevo comportamiento del antiguo".

Juntos dicen: *mejora, pero sigue siendo tú mismo*.

## Cómo funciona el aprendizaje (solo intuición)

Cada ronda de entrenamiento se parece a esto:

1. Tomar algunas instrucciones (prompts). Dejar que el modelo actual produzca respuestas.
2. Puntuar las respuestas con el modelo de recompensa.
3. Comparar con el **modelo de referencia** — una copia congelada del modelo de antes del entrenamiento. Si las nuevas respuestas son radicalmente diferentes, restar una penalización KL de la recompensa.
4. Empujar al modelo hacia las respuestas que obtuvieron buena puntuación.

Lo de "Proximal" en PPO significa *no dar grandes saltos*. Cada actualización es un paso pequeño y cuidadoso. Los grandes saltos en el entrenamiento de políticas causan fallos, razón por la cual los métodos anteriores, como el gradiente de política vainilla, eran tan inestables.

## Qué muestra el experimento

Comenzamos con una política nueva y un modelo de recompensa entrenado. PPO se ejecuta durante 150 iteraciones, tomando muestras de lotes de respuestas y actualizando la política.

![Entrenamiento PPO](outputs/ppo_fine_tuning.png)

- **Izquierda**: la puntuación media del modelo de recompensa sube constantemente. La política está aprendiendo a producir respuestas que le gustan al juez.
- **Centro**: la divergencia KL con respecto al modelo de referencia crece. La política se está alejando de donde empezó. Esto es lo esperado, pero si creciera sin control, el modelo derivaría hacia el sinsentido.
- **Derecha**: la recompensa moldeada (recompensa bruta menos la penalización KL) sigue de cerca a la recompensa bruta al principio, y luego se queda atrás a medida que el KL sube. La penalización está haciendo su trabajo: haciendo que el modelo "pague" por alejarse demasiado.

En un sistema RLHF real, ajustas el coeficiente KL hasta que la puntuación siga subiendo pero el modelo se mantenga coherente. Si la penalización es demasiado pequeña, el modelo hackea la recompensa emitiendo frases repetitivas extrañas. Si es demasiado grande, el modelo nunca mejora.

## Dónde encaja esto en el pipeline de RLHF

Este es el paso dos de la receta clásica de RLHF:

1. Entrenar un modelo de recompensa a partir de las preferencias.
2. **Ajustar el modelo de lenguaje con PPO usando ese modelo de recompensa.**
3. (Opcional) Saltarse el paso 2 con DPO si se desea un camino más sencillo.

PPO es la herramienta de trabajo que empresas como OpenAI y Anthropic utilizaron para la primera ola de modelos alineados, incluidos InstructGPT y el ChatGPT original.

## Por qué esto importa fuera del laboratorio

El patrón de "mejorar, pero no desviarse" aparece en todas partes:

- Un pianista que practica una pieza difícil no cambia toda su técnica para clavar un pasaje; eso arruinaría el resto del recital.
- Una empresa que retoca un sitio web para aumentar los registros tiene que mantener la marca reconocible para los usuarios actuales.
- Una fábrica que ajusta un mando en un proceso mantiene los otros cerca de los ajustes que se sabe que funcionan.

PPO es solo una versión cuidadosa de esta idea universal, escrita en matemáticas.

## Resumen de una frase

**El ajuste fino con PPO empuja a un modelo hacia una recompensa más alta, mientras que una penalización KL lo mantiene cerca de su comportamiento original: mejora, pero sigue siendo tú mismo.**
