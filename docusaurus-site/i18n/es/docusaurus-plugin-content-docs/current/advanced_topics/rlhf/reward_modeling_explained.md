# Modelado de Recompensas: Enseñar a un ordenador qué prefieren las personas

## La Gran Idea

Un modelo de recompensa es un pequeño juez. Le muestras dos respuestas a la misma pregunta, le dices cuál le gustó más a una persona y, con el tiempo, aprende a dar una puntuación más alta a las respuestas que la gente preferiría.

¿Por qué necesitamos a este juez? Porque la mayor parte de lo que queremos de un modelo de lenguaje es difícil de escribir como una fórmula matemática. No existe una única ecuación para lo "útil", lo "educado" o lo "bien escrito". Pero la gente casi siempre puede señalar cuál es la mejor de dos opciones. El modelo de recompensa convierte esos simples votos de "esta es mejor" en una puntuación que un algoritmo de aprendizaje puede utilizar.

## Una analogía de la vida real

Imagina que le enseñas a un amigo a hornear brownies.

No le entregas un libro de reglas de 50 páginas sobre "qué hace que un brownie sea bueno". En su lugar, pruebas dos tandas y dices:

"Esta es mejor".

Después de unas cuantas rondas de esto, tu amigo empieza a notar patrones. Quizás el más meloso siempre gana. Quizás el que está demasiado horneado siempre pierde. Tu amigo construye un sistema de puntuación mental a partir de tus comparaciones.

Un modelo de recompensa hace exactamente esto, pero con números. No necesita saber *por qué* la respuesta elegida es mejor. Solo necesita muchos ejemplos de "esto supera a aquello" y gradualmente aprende una puntuación que se alinea con las preferencias.

## Cómo funciona el aprendizaje (solo intuición)

Cada ejemplo es un trío: una instrucción (prompt), una respuesta **elegida** (chosen) y una respuesta **rechazada** (rejected). Queremos que el modelo dé una puntuación más alta a la elegida que a la rechazada — por cualquier margen.

El impulso del entrenamiento es simple en esencia:

- ¿La puntuación de la elegida es demasiado baja? Súbela.
- ¿La puntuación de la rechazada es demasiado alta? Bájala.
- ¿Ya están en el orden correcto con una brecha clara? Déjalas en paz.

Ese impulso se llama pérdida de Bradley-Terry, y es la receta estándar en los sistemas de RLHF modernos.

## Qué muestra el experimento

Entrenamos un modelo de recompensa en 2,000 pares de preferencias sintéticas. El gráfico de abajo muestra tres vistas de la misma ejecución de entrenamiento.

![Entrenamiento del modelo de recompensa](outputs/reward_modeling.png)

- **Izquierda**: la pérdida cae rápidamente. El modelo adquiere más confianza en sus clasificaciones.
- **Centro**: la precisión de las preferencias sube hasta casi el 100%. En casi todos los pares, la respuesta elegida obtiene una puntuación más alta que la rechazada.
- **Derecha**: las distribuciones de puntuación para las respuestas elegidas frente a las rechazadas se separan. Al principio se solapaban; tras el entrenamiento, las respuestas elegidas se sitúan claramente a la derecha.

Esa separación es el objetivo principal. Un paso posterior (PPO o DPO) puede ahora utilizar esta puntuación como una meta hacia la cual optimizar.

## Dónde encaja esto en el pipeline de RLHF

La hoja de ruta describe el RLHF como "alinear modelos con las preferencias humanas". El modelo de recompensa es el paso uno de tres:

1. **Modelo de recompensa (este archivo)**: convertir los votos de preferencia en una puntuación.
2. **Ajuste fino con PPO**: empujar el modelo de lenguaje hacia puntuaciones más altas manteniéndose cerca de su comportamiento original.
3. **DPO**: un atajo más reciente que omite el modelo de recompensa por completo.

Así pues, el modelado de recompensas es el puente entre el *juicio humano* y la *optimización de la máquina*. Si este puente sale mal, todos los pasos posteriores se desviarán de su curso.

## Por qué esto importa fuera del laboratorio

La misma idea aparece en muchos lugares:

- Los **sistemas de recomendación** aprenden lo que te gusta a partir de los clics, los saltos y el tiempo que pasas mirando.
- Los **motores de búsqueda** aprenden la clasificación a partir de "en qué resultado hiciste clic".
- Los **restaurantes** aprenden cuáles son los platos populares por los pedidos repetidos, no por clientes que escriben ensayos sobre lo que les gustó.

Siempre que sea más fácil *comparar* que *puntuar*, un modelo de recompensa es la herramienta adecuada.

## Resumen de una frase

**Un modelo de recompensa es un juez aprendido que convierte las preferencias de "esta es mejor" en una puntuación numérica que el resto del RLHF puede optimizar.**
