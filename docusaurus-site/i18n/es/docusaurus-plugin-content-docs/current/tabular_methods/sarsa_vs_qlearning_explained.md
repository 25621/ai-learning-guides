# SARSA vs Q-Learning: Caminos seguros frente a óptimos 🐢 vs 🐇

## ¿Qué es esto?

Dos robots necesitan caminar a lo largo del **borde de un precipicio** para llegar a la meta. Ambos robots todavía están *aprendiendo* y a veces realizan movimientos aleatorios (¡uups!).

- 🐢 **Robot SARSA**: "Sé que a veces me tambaleo... así que caminaré lejos del precipicio para estar seguro, aunque tarde más".
- 🐇 **Robot Q-Learning**: "El camino más corto bordea el precipicio — ¡allá vamos! (Se cae a veces mientras aprende, pero acaba aprendiendo la mejor ruta)".

Ambos robots son inteligentes, pero hacen una **elección diferente**: seguro-pero-más-lento frente a óptimo-pero-arriesgado-mientras-se-aprende.

---

## La diferencia clave: ¿Qué "Acción Siguiente" utilizas?

Al actualizar las puntuaciones después de cada paso, ambos algoritmos preguntan:
> "¿Cuál es el valor del *siguiente estado*?"

| Algoritmo | Utiliza la siguiente acción... | ¿Dentro de la política (On-policy)? |
|-----------|------------------------|------------|
| **SARSA** | ...que *realmente tomaré* (¡quizás sea aleatoria!) | Sí |
| **Q-Learning** | ...que es *teóricamente la mejor* (siempre codiciosa) | No |

**Ejemplo de la vida real:** Dos niños aprendiendo a montar en bicicleta.

- **Niño SARSA**: Se mantiene cerca del césped porque *sabe* que a veces se tambalea al azar. Está planeando para su yo real tambaleante.
- **Niño Q-Learning**: Practica en medio del camino porque se imagina a un yo futuro perfecto que nunca se tambalea. Se cae algunas veces ahora, pero aprende el mejor camino más rápido.

Ambos niños acaban aprendiendo, ¡pero durante el entrenamiento el niño SARSA se cae menos!

---

## Qué encontró nuestro código

Ambos algoritmos se ejecutaron durante **500 episodios** en Cliff Walking con ε=0.1 (ε = épsilon; aquí significa un 10% de probabilidad de realizar un movimiento aleatorio):

| Métrica | SARSA | Q-Learning |
|--------|-------|------------|
| Recompensa media en entrenamiento (últimos 50 ep) | **-19.7** | **-51.0** |
| Evaluación codiciosa (sin exploración) | -17 | **-13** |

- **Durante el entrenamiento**: SARSA obtiene **recompensas mucho mejores** porque evita el precipicio (teniendo en cuenta sus propios movimientos aleatorios).
- **Después del entrenamiento** (puro codicioso): ¡Q-Learning encuentra el **camino óptimo más corto** (-13)!

A medida que ε se reduce hacia 0, ambos algoritmos convergen hacia la misma política óptima.

---

## Resumen visual

```
Camino SARSA (en entrenamiento):    Camino Q-Learning (greedy, tras entrenamiento):
[ ][→][→][→][→][→][→][→][→][→][→][↓]   [ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][↓]
[↑][→][→][→][→][→][→][→][→][→][→][↓]   [ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][↓]
[ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][↓]   [ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][↓]
[S][P][P][P][P][P][P][P][P][P][P][G]   [S][→][→][→][→][→][→][→][→][→][→][G]
    (rodeo seguro, filas sup.)              (óptimo, pega al precipicio)
```

---

## Ejemplos de la vida real

- **Cirujano novel frente a cirujano experimentado**: El cirujano novel (SARSA) se aleja de las técnicas arriesgadas mientras aprende. El cirujano experimentado (Q-Learning codicioso) utiliza la técnica más eficiente tras haberla dominado.
- **Conducir por la ciudad frente a ruta por autopista**: Una planificación tipo SARSA toma calles residenciales más seguras; el Q-Learning encuentra la autopista óptima pero estrecha.
- **Estudiante estudiando**: El estudiante-SARSA se ciñe a temas bien comprendidos durante la práctica. El estudiante-Q-Learning se lanza a los problemas más difíciles (falla más) pero aprende la estrategia óptima.

---

## Palabras Clave para Recordar

- **Dentro de la política (On-policy)** (SARSA): Aprende sobre lo que *realmente haces*, incluida la exploración aleatoria.
- **Fuera de la política (Off-policy)** (Q-Learning): Aprende sobre el *mejor comportamiento posible* de forma independiente a lo que realmente haces.
- **Camino seguro**: Ruta más larga que evita el peligro, utilizada cuando la exploración causa accidentes.
- **Camino óptimo**: Ruta más corta/de mayor recompensa, encontrada cuando no hay exploración.
- **Dilema exploración-explotación**: El equilibrio entre probar cosas nuevas y usar lo que ya sabes.

La gran idea: **SARSA es más seguro durante el entrenamiento (dentro de la política), Q-Learning encuentra el camino óptimo más rápido (fuera de la política). ¡Cuál es mejor depende de si importa caerse por el precipicio!**
