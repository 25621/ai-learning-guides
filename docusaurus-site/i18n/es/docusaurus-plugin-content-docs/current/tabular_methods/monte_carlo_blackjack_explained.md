# Control de Monte Carlo para Blackjack 🃏

## ¿Qué es esto?

¿Alguna vez has jugado a un juego de cartas en el que tienes que decidir: **"¿Pido otra carta o estoy contento con lo que tengo?"**?

¡El **Blackjack** (también llamado "21") es exactamente eso! Quieres que tus cartas sumen lo más cerca posible de 21, sin pasarte. ¡Si te pasas de 21, te has "pasado" (bust) y pierdes!

El **control de Monte Carlo** es cómo un robot aprende a jugar al Blackjack — jugando *miles de partidas completas* y recordando qué funcionó y qué no.

---

## La Gran Idea: Aprender de historias completas

La palabra "Monte Carlo" proviene del famoso casino de Mónaco. En matemáticas, significa: **usar experimentos aleatorios para aprender algo**.

Así es como funciona:

1. **Juega una partida completa** (un episodio completo) utilizando cualquier estrategia que tengas.
2. **Mira lo que pasó**: ¿Ganaste? ¿Perdiste? ¿Empataste?
3. **Trabaja hacia atrás**: ¿Fue buena idea pedir carta (hit) con 17? ¿Y con 14?
4. **Actualiza tu memoria**: Recuerda si cada decisión te llevó a ganar o a perder.

¡Haz esto durante **500,000 partidas** y te volverás muy bueno!

**Ejemplo de la vida real:** Imagina aprender a cocinar preparando 500,000 comidas. Cada vez, recuerdas exactamente lo que hiciste — y si la comida sabía bien. Después de suficientes intentos, sabes: "Añadir demasiada sal en este paso siempre lo estropeaba". ¡Monte Carlo funciona de la misma manera!

---

## Diferencia clave con SARSA y Q-Learning

SARSA y Q-Learning actualizan sus conocimientos **después de cada paso** (incluso a mitad del episodio). Monte Carlo espera hasta que el **episodio completo ha terminado**, y luego mira todo lo ocurrido.

| Método | ¿Cuándo actualiza? | ¿Necesita el episodio completo? |
|--------|---------------|------------------------|
| **TD (SARSA, Q-Learning)** | Después de cada paso | No |
| **Monte Carlo** | Después de cada episodio completo | Sí |

Esto hace que Monte Carlo sea más sencillo de entender, pero no puede aprender hasta que termina cada episodio.

---

## El estado en el Blackjack

El robot mira 3 cosas en cada turno:
1. **Mi total de cartas** (12 a 21).
2. **¿Qué carta muestra el crupier?** (As hasta 10).
3. **¿Tengo un As utilizable?** (Un As puede contar como 1 u 11).

A partir de estas 3 piezas de información, decide: **¿Pedir carta (Hit) o Plantarse (Stick)?**

---

## Qué encontró nuestro código

Después de **500,000 partidas** de Blackjack:

| Resultado | Porcentaje |
|---------|------------|
| **Victorias** | **43.1%** |
| **Empates** | 8.9% |
| **Derrotas** | 48.0% |

¡Esto está cerca de la "estrategia básica" matemáticamente óptima (alrededor del 42-43% de victorias)! El robot aprendió cuándo pedir y cuándo plantarse, simplemente jugando partidas y recordando.

La política aprendida muestra:
- **Pedir carta** (Hit) cuando tu total es bajo (es poco probable que te pases).
- **Plantarse** (Stick) cuando tu total es alto (podrías pasarte si pides otra carta).
- Tener un **As utilizable** te permite ser más agresivo (puede cambiar de 11 a 1 si es necesario).

---

## Ejemplos de la vida real

- **Previsión meteorológica**: Las simulaciones de Monte Carlo ejecutan miles de escenarios de "qué pasaría si" para predecir el tiempo de mañana.
- **Modelado del mercado de valores**: Los analistas simulan miles de futuros posibles para estimar el riesgo.
- **Aprender a jugar al ajedrez**: Un jugador revisa partidas enteras (no solo movimientos individuales) para entender qué estrategia le llevó a ganar.

---

## Palabras Clave para Recordar

- **Episodio**: Una partida completa de principio a fin.
- **Retorno (G)**: Recompensa total recogida desde un punto del juego hasta el final.
- **MC de cada visita (Every-visit MC)**: Actualizar la puntuación de un estado cada vez que lo visitas en un episodio.
- **Sin bootstrapping**: Monte Carlo no utiliza estimaciones de valores futuros — ¡espera al resultado real!
- **Política ε-soft** (ε = epsilon): Normalmente hace la mejor acción conocida, pero a veces explora al azar.

La gran idea: **¡Monte Carlo aprende jugando muchas partidas completas. Es como aprender de la experiencia — recuerdas todo lo que pasó y descubres qué te llevó a ganar!**
