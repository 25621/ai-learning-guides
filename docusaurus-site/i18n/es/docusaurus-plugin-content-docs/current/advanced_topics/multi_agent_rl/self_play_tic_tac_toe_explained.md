# Auto-juego (Self-Play): Enseñar a un agente dejando que juegue contra sí mismo ♟️

## ¿Qué es el auto-juego?

Imagina a una niña que quiere volverse muy buena al ajedrez pero no tiene con quién jugar. Así que juega contra sí misma. Mano izquierda contra mano derecha. En cada partida, *ambos* bandos intentan ganar. En cada partida, *ambos* bandos aprenden qué funcionó.

Eso es el **auto-juego**: un único agente actúa como ambos jugadores, y cada movimiento se convierte en una lección para quien mueva a continuación. Sin profesor, sin oponente experto. Solo un aprendiz que es también su propia escalera.

El auto-juego suena a truco — ¿seguro que necesitas un oponente real? —, pero es el motor que hay detrás de los hitos más famosos de RL de la última década: **AlphaGo Zero**, **AlphaZero**, **MuZero**, **OpenAI Five**. Todos utilizan el auto-juego. La razón es simple: a medida que tú mejoras, tu oponente mejora en la misma medida. El desafío siempre está a la altura de tu habilidad.

---

## Por qué funciona

Tres cosas hacen que el auto-juego sea especial:

1. **Oponentes infinitos.** Nunca te quedas sin partidas. El oponente siempre está presente y es gratis.
2. **Un plan de estudios que crece contigo.** Un principiante solo puede jugar contra otros principiantes. A medida que mejoras, también lo hace tu sombra, automáticamente.
3. **Simetría.** En un juego de suma cero (la victoria de un jugador es la pérdida del otro), un único conjunto de valores Q describe a ambos bandos; simplemente inviertes el signo cuando es el turno del otro jugador. Así, una *única* tabla Q puede enseñarse a sí misma.

El tres en raya (Tic-tac-toe) es el banco de pruebas perfecto: lo suficientemente pequeño como para caber en un diccionario, pero lo suficientemente complejo como para que elegir movimientos al azar casi siempre lleve a una derrota contra un jugador estratégico.

---

## Una analogía de la vida real

- **Practicar tenis contra una pared.** No puedes perder contra una pared, pero puedes practicar tus saques. El auto-juego es hacer esto en ambos extremos: tú eres la pared *y* el jugador, y vas alternando.
- **Un club de debate que argumenta en ambos bandos.** Surgen mejores debatientes al defender siempre el punto de vista opuesto a lo que creen personalmente. Cada argumento entrena tanto el ataque como la defensa.
- **AlphaGo Zero.** Aprendió a partir de cero partidas humanas. Partiendo de movimientos aleatorios, jugó millones de partidas contra sí mismo; en pocos días era mejor que cualquier programa de Go anterior, incluido el que venció a Lee Sedol.

---

## Qué hace nuestro código

Aprendemos una tabla Q para el *jugador al que le toca mover*:

```
Q[(tablero, jugador_que_mueve)][accion] = retorno esperado para ese jugador
```

El bucle de entrenamiento es:

1. Comenzar con un tablero vacío. `jugador = X`.
2. Ambos jugadores actúan con el **mismo agente**, utilizando ε-greedy.
3. Después de cada partida, recorrer hacia atrás cada trío (tablero, jugador, acción) de la historia y aplicar la actualización de Q-learning.
4. La recompensa invierte su signo entre turnos: si X gana, cada movimiento que hizo X obtiene +1 (o obtiene el valor de un estado ganador futuro); cada movimiento que hizo O obtiene -1.
5. Reducimos lentamente (decay) nuestra tasa de exploración (ε) de 0.2 → 0.02, para que el agente se comprometa con su mejor juego al final del entrenamiento en lugar de probar movimientos aleatorios.

Cada 2,500 episodios evaluamos al agente contra un **oponente aleatorio** (congelamos el proceso de aprendizaje para que no se realicen nuevas actualizaciones en la tabla Q durante la evaluación, y ambos bandos juegan de forma codiciosa). El agente debería ganar o empatar cerca del 100% de esas partidas tras suficiente auto-juego.

### Qué deberías ver

Tras 50,000 episodios de auto-juego:

| Enfrentamiento | Resultado esperado |
|----------|-----------------|
| Agente entrenado vs Oponente aleatorio (1000 partidas) | **~95-99% victorias o empates**, virtualmente 0% derrotas |
| Agente entrenado vs Sí mismo (200 partidas codiciosas) | **Las 200 empates**. El tres en raya es un juego que siempre termina en tablas (empate) si ambos jugadores juegan perfectamente. El hecho de que el auto-juego empate todas las partidas es una señal de convergencia. |

El gráfico `outputs/self_play_tic_tac_toe.png` muestra las fracciones de victoria/empate/derrota del agente contra un oponente aleatorio a lo largo del tiempo:
- La tasa de victorias empieza en ~60% (cuando ambos jugadores juegan al azar, el primer jugador tiene una ventaja inherente porque llega a colocar más fichas en el tablero, lo que lleva a una tasa de victorias base de aproximadamente el 60% para el jugador X).
- Sube a >90%.
- La tasa de derrotas cae a casi el 0%.

El script también imprime una partida de ejemplo movimiento a movimiento al final para que puedas ver al agente jugar.

---

## Ojo con estas sutilezas

- **Los cambios de signo importan.** Un error común: olvidar que "el oponente maximizando su valor" significa *minimizar el nuestro* en el objetivo del bootstrap. La actualización en nuestro código utiliza `objetivo = recompensa - gamma * max(Q[siguiente, oponente])`.
- **La simetría no se explota aquí.** Una implementación real canonicalizaría los tableros (es decir, rotaría o reflejaría cualquier estado del tablero a una 'forma normal' estándar y única para que el agente reconozca situaciones de tablero idénticas) para compartir los valores Q a través de 8 simetrías. Nosotros omitimos esto — el espacio de estados es lo suficientemente pequeño como para usar la fuerza bruta.
- **La tabla Q crece.** Tras 50k partidas de auto-juego verás unos pocos miles de claves estado-jugador. Eso está bien aquí; para el ajedrez o el Go necesitarías una red neuronal en su lugar, razón por la cual **AlphaZero sustituye la tabla por una CNN + MCTS**.

---

## Donde el auto-juego falla

- **Juegos que no son de suma cero.** Que "ambas partes estén contentas" es incompatible con el juego simétrico; no puedes simplemente invertir un signo.
- **Roles asimétricos.** Si el "atacante" y el "defensor" tienen espacios de acción diferentes, necesitas dos redes separadas.
- **Ciclos de estrategia.** El auto-juego puro puede quedarse atascado en ciclos tipo piedra-papel-tijera. AlphaStar solucionó esto manteniendo un gran *fondo* (pool o "liga") de versiones pasadas guardadas del agente y eligiendo oponentes de ese fondo al azar, para que el agente aprenda a vencer a muchos estilos de juego diferentes en lugar de solo al actual.
- **Hackeo de recompensas (Reward hacking).** El auto-juego hace que ambos bandos sean más inteligentes, pero solo en el juego *tal y como tú lo has definido*. Si tu sistema de recompensas tiene lagunas no deseadas (como recompensar a un jugador solo por sobrevivir más tiempo en lugar de por ganar), ambos bandos explotarán mutuamente la laguna, lo que llevará a comportamientos extraños e inútiles en lugar de dominar el juego real.

---

## Palabras Clave para Recordar

| Palabra | Significado |
|------|---------|
| **Auto-juego (Self-play)** | El mismo agente juega ambos bandos de un juego. |
| **Suma cero** | La ganancia de un jugador = la pérdida del otro. |
| **Simetría** | Una tabla Q puede servir a ambos bandos si inviertes los signos. |
| **Juego de población** | Auto-juego con *muchas* versiones pasadas de ti mismo como oponentes (AlphaStar). |
| **Plan de estudios (Curriculum)** | Una progresión natural de la dificultad — el auto-juego la obtiene gratis. |
| **MCTS** | Búsqueda en Árbol de Monte-Carlo — el algoritmo de planificación que AlphaZero empareja con el auto-juego. |

---

## Resumen de una frase

> **El auto-juego convierte la mejora en su propia escalera: cada vez que tú te vuelves mejor, tu oponente también lo hace — automáticamente.**

Esta idea, escalada con **redes neuronales** (funciones matemáticas inspiradas en el cerebro que aprenden patrones a partir de datos) y búsqueda en árbol, venció a los mejores humanos en Go, ajedrez, shogi, Dota 2 y StarCraft.
