# Comparación de Estrategias de Exploración 🔦

## El Problema en una Frase

Un agente de RL tiene que hacer dos cosas que tiran en direcciones opuestas:

- **Explotar**: hacer lo que mejor ha funcionado hasta ahora.
- **Explorar**: probar algo nuevo, por si acaso es aún mejor.

Inclínate demasiado hacia la explotación y te conformarás felizmente con una rutina mediocre para siempre. Inclínate demasiado hacia la exploración y nunca cobrarás los beneficios. *Cómo* exploras — no solo *si* lo haces — es lo que separa a un agente que resuelve Montezuma's Revenge de uno que obtiene una puntuación de cero.

Este script pone frente a frente **cinco** estrategias de exploración en las mismas dos tareas difíciles, para que puedas ver sus personalidades.

## Analogía de la Vida Real: Elegir un Lugar para Almorzar

Te acabas de mudar a una ciudad nueva con 200 restaurantes.

- **ε-greedy** = "Ir a mi favorito actual, pero una vez cada diez días lanzar un dado y elegir un restaurante *totalmente al azar*." Probarás de todo, pero *sin rumbo* — y volverás a lugares que ya odiabas.
- **Inicialización optimista** = "Asumir que *cada* restaurante que no he probado es el mejor de la ciudad hasta que se demuestre lo contrario." Irás probando metódicamente los 200, descartando cada uno a medida que la realidad te decepcione — y encontrarás los verdaderamente geniales rápidamente.
- **UCB (Upper Confidence Bound)** = "Preferir mi favorito, pero dar un *bono* a los lugares que apenas he probado — cuanto menos sepa de él, mayor será el bono." Esto es inteligente sobre *qué* lugar desconocido probar hoy, pero cada decisión es local: elige la opción que mejor se ve *ahora mismo* sin planear una ruta a través de barrios enteros sin explorar. No pensará "debería cruzar la ciudad hacia el lado este, porque hay veinte lugares sin probar agrupados allí" — cada restaurante se evalúa de forma aislada, paso a paso.
- **Bono de recompensa basado en el recuento** = como UCB, pero también *disfrutas de la novedad en sí misma* — una comida en un lugar nuevo es intrínsecamente satisfactoria, y esa satisfacción da forma a tu plan a largo plazo de en qué barrios aventurarte.
- **Bono de recompensa por error de predicción** = "Siento una emoción con una comida que me *sorprendió* — algo que no podría haber predicho." ¿Un lugar nuevo que resulta ser exactamente como esperabas? Bah. ¿Uno que es radicalmente diferente de tu modelo mental? Fascinante, y actualizas tu plan para buscar más como ese.

## Las Cinco Estrategias (todas en `compare_exploration.py`)

### 1. ε-greedy — la predeterminada, y es *vacilación*, no exploración

Actúa de forma codiciosa (greedy), pero con probabilidad ε toma una acción uniformemente aleatoria. Es el estándar en DQN y similares. Su fallo fatal en tareas difíciles: **cada paso es un lanzamiento de moneda independiente.** Para tropezar con una cadena de `N` movimientos correctos, necesitas que la moneda salga bien `N` veces seguidas — eso es exponencialmente improbable. ε-greedy es *sacudirse*, no *explorar*.

### 2. Inicialización optimista — "inocente hasta que se demuestre que es aburrido"

Comienza *cada* valor Q en el mayor retorno que sea posible, `R_max / (1 − γ)`. Ahora, una acción que el agente nunca ha probado parece lo mejor del mundo, por lo que la política **codiciosa** se ve obligada a ir a probarla; solo después de visitarla el valor cae hacia la verdad. El optimismo sobre las regiones *no* probadas se **propaga automáticamente a través de la función de valor** (vía el bootstrap de Q-learning), por lo que el agente es arrastrado, paso a paso, hacia las partes del mundo que no ha visto. Casi gratis, sin contabilidad extra — y, como verás, el explorador *profundo* más fuerte en un mundo tabular pequeño.

### 3. Selección de acciones estilo UCB — bono en la *elección*, no en la *recompensa*

Elige `argmax_a [ Q(s,a) + c·√(ln t / N(s,a)) ]`: prefiere acciones de alto valor, pero infla las que raramente has probado. Famoso en los bandidos multibrazo (multi-armed bandits). El truco: el bono vive solo en la **regla de selección de acciones**, nunca en la recompensa — por lo que *no* fluye a través de la función de valor. UCB es genial para "asegurarme de haber probado cada acción en *este* estado" pero débil para "planear una ruta hacia una región inexplorada lejana".

### 4. Bono de **recompensa** basado en el recuento — curiosidad, la versión clásica

Añade `1/√(N(s,a))` a la **recompensa** (con un peso `beta` que decae). Debido a que está en la recompensa, Q-learning *sí* lo propaga: los estados que conducen hacia regiones novedosas se vuelven valiosos. Esta es la idea de MBIE-EB / "bono de exploración" clásico.

### 5. Bono de **recompensa** por error de predicción — curiosidad, la versión ICM/RND

Añade `−log P(s'|s,a)` de un pequeño modelo predictivo aprendido a la recompensa (de nuevo con `beta` decreciente). La señal de novedad más aguda de las cinco: en un mundo determinista, la sorpresa de una transición cae a ~0 en el momento en que la has visto una vez, en lugar de desvanecerse lentamente como `1/√N`. El primo tabular de ICM / RND.

## Las Dos Tareas de Prueba

- **Tarea A — MiniMontezuma**: un mundo de cuadrícula (gridworld) llave→puerta→tesoro, recompensa solo en el tesoro (~15 movimientos perfectos de distancia). Prueba "¿puedes sobrevivir a una cadena larga de recompensa escasa?".
- **Tarea B — DeepSea(N)**: la cadena de exploración profunda de libro de texto, ejecutada en longitudes `N = 5, 8, 11, 14`. La recompensa se esconde tras `N` movimientos correctos, cada uno con un pequeño coste inmediato — por lo que un agente miope aprende a evitar el coste y nunca encuentra el premio. Prueba "¿sigue funcionando tu estrategia a medida que la cadena se hace más *larga*?".

## Qué Sucede Realmente (ejecútalo y verás)

**Tarea A — MiniMontezuma:**

| Estrategia | Primer tesoro | Tasa de resolución final |
|----------|---------------:|-----------------:|
| ε-greedy | nunca | 0.00 |
| inicialización optimista | ~episodio 1 | 1.00 |
| selección de acción UCB | ~episodio 3 | ~0.95 |
| bono recompensa recuento | ~episodio 82 | ~0.41 |
| bono recompensa predicción | ~episodio 23 | 1.00 |

**Tarea B — DeepSea, fracción de semillas que encontraron la recompensa:**

| Estrategia | N=5 | N=8 | N=11 | N=14 |
|----------|----:|----:|-----:|-----:|
| ε-greedy | 0 | 0 | 0 | 0 |
| inicialización optimista | 1.0 | 1.0 | 1.0 | 1.0 |
| selección de acción UCB | 1.0 | 1.0 | 0.0 | 0.0 |
| bono recompensa recuento | 1.0 | 1.0 | ~0.1 | 0.0 |
| bono recompensa predicción | ~0.9 | ~0.8 | ~0.9 | ~0.2 |

*(Los números oscilan un poco con las semillas aleatorias, pero la forma es sólida como una roca).*

## Las Lecciones

1. **ε-greedy no es exploración.** Nunca resuelve *ninguna* de las dos tareas difíciles. El titubeo aleatorio simplemente no logra ensartar secuencias correctas largas. (Sin embargo, sigue siendo el valor predeterminado en mucho código — porque en tareas *fáciles* es suficiente y muy simple).

2. **La verdadera exploración significa ser optimista sobre lo desconocido — de una forma u otra.** Ya sea que integres el optimismo en los *valores iniciales* (estrategia 2), en la *elección de la acción* (estrategia 3) o en una *recompensa autogenerada* (estrategias 4–5), el hilo común es: *hacer que lo inexplorado parezca atractivo*, y luego dejar que el aprendizaje te lleve allí.

3. **En una cuadrícula de recompensa escasa, las cuatro estrategias "reales" funcionan — y el bono por error de predicción llega más rápido**, porque produce la señal más nítida de "esto es nuevo".

4. **En una cadena *profunda*, donde el optimismo tiene que viajar un largo camino, el campeón indiscutible es la inicialización optimista.** Propaga el optimismo a través de la función de valor de forma gratuita. UCB se desmorona primero (su bono nunca entra en la función de valor, por lo que no puede *planear* profundamente). Los bonos de recompensa lo hacen mejor — *sí* se propagan — pero el Q-learning tabular simple es lento para empujar ese optimismo a lo largo de una cadena larga antes de que el bono decaiga.

5. **Ese último punto es exactamente por qué escalar la exploración profunda a píxeles necesitó potencia extra** — DQN con bootstrap, RND con una red neuronal real (para que el optimismo se *generalice* a través de estados similares en lugar de propagarse celda por celda), Go-Explore (literalmente recordar y volver a estados prometedores). Los juguetes tabulares aquí te muestran los *principios*; los sistemas reales son estos mismos principios más una red que generaliza.

## Palabras Clave para Recordar

| Palabra | Significado |
|------|---------|
| **Dilema exploración–explotación** | Probar cosas nuevas vs. cobrar lo que ya sabes — la tensión central en RL |
| **Dithering (Titubeo/Vacilación)** | "Exploración" añadiendo ruido aleatorio a las acciones (ε-greedy, ruido Gaussiano) — débil en tareas difíciles |
| **Optimismo ante la incertidumbre** | El principio general: tratar lo desconocido como si fuera genial hasta que lo hayas comprobado |
| **Inicialización optimista** | Implementar ese principio iniciando todos los valores en el retorno máximo posible |
| **UCB** | Upper Confidence Bound: elegir `argmax (valor + bono que se reduce con el recuento de visitas)` |
| **Exploración profunda** | Exploración que requiere una secuencia *coherente* y larga de acciones "inusuales", no solo una |
| **Recocido (annealing) de `beta`** | Reducir el peso de la curiosidad con el tiempo para que el agente deje de explorar y explote |

## Resumen de una frase

> **ε-greedy es solo ruido; toda estrategia de exploración real funciona haciendo que lo inexplorado parezca atractivo — a través de valores optimistas, un bono en la elección de acciones o una recompensa de novedad autogenerada — y la elección correcta depende de si tu recompensa es simplemente *escasa* (como encontrar un premio oculto en un campo llano) o genuinamente *profunda* (como una cerradura de combinación que requiere una secuencia larga y precisa de elecciones específicas para abrirse).**
