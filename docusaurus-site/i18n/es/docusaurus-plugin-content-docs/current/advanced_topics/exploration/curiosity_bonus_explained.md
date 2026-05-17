# Bono de Curiosidad (Motivación Intrínseca) 🧭

## ¿Qué es?

Imagina a un niño pequeño dejado en una habitación nueva. Nadie le paga, nadie le aplaude — sin embargo, se dirige directamente al armario que no ha abierto, al botón que no ha pulsado, al juguete ruidoso en la esquina. Están funcionando con una **recompensa interna**: *"Eso parece nuevo. Ve a echar un vistazo".*

Un **bono de curiosidad** le da a un agente de aprendizaje por refuerzo el mismo impulso interno. La recompensa real del entorno (la recompensa "extrínseca" — puntos, dinero, ganar el juego) se deja exactamente como está. Simplemente añadimos una segunda recompensa autogenerada por visitar cosas que el agente encuentra *novedosas* o *sorprendentes*, y entrenamos sobre la suma:

```
recompensa de la que aprende el agente = recompensa real + beta * bono de curiosidad
```

`beta` es una perilla que empieza siendo grande (¡sé curioso!) y se reduce con el tiempo (deja de perder el tiempo, ve a cobrar lo que has aprendido).

## ¿Por qué molestarse? El problema de la Recompensa Escasa

Los agentes de RL normales aprenden de las recompensas que reciben realmente. Eso funciona de maravilla cuando las recompensas están en todas partes ("+1 en cada paso que te mantengas erguido" en CartPole). Se desmorona cuando la recompensa es **escasa** (sparse) — cero, cero, cero, ... , cero, y luego finalmente un +1 después de una secuencia larga y muy específica de acciones correctas.

Ejemplos reales de recompensa escasa:

- **Montezuma's Revenge** (el juego de Atari): tu primer punto llega solo después de unos 100 movimientos precisos — bajar una escalera, esquivar una calavera, subir, tomar una llave. Hasta entonces, la puntuación es un cero absoluto.
- **Una cerradura de combinación.** 9,999 códigos erróneos no te dan nada; uno te da el premio.
- **Descubrimiento de fármacos / experimentos científicos.** Miles de ensayos fallidos, y luego uno que funciona.
- **Escribir una prueba larga o un programa.** No hay crédito parcial hasta que todo el conjunto sea correcto.

Un agente que solo se basa en recompensas en estos entornos es como un estudiante que se niega a estudiar a menos que le paguen por cada respuesta correcta en el examen final — nunca empieza. La curiosidad es el bono que dice *"explorar es su propia recompensa"*, por lo que el agente sigue buscando hasta que tropieza con el premio real.

## Dos Sabores de Curiosidad (ambos implementados en `curiosity_bonus.py`)

### 1. Novedad basada en el recuento: "Apenas he estado aquí"

La señal de novedad más simple posible. Mantén un recuento `N(s, a)` de cuántas veces has tomado la acción `a` en el estado `s`, y dante un bono que se reduce a medida que ese recuento crece:

```
bono de curiosidad = 1 / sqrt( N(s, a) + 1 )
```

La primera vez que pruebas algo: bono = 1.0. Después de 100 intentos: bono = 0.1. Después de 10,000 intentos: 0.01. El agente es recompensado por ir a donde no ha estado, y el atractivo se desvanece naturalmente en los terrenos ya conocidos.

**Analogía de la vida real:** un turista con una lista de "lugares que no he visitado". ¿Un barrio nuevo? Prioridad máxima. ¿El café en el que has estado cincuenta veces? Ya no es emocionante.

Esta es la idea más vieja de todas (MBIE-EB, UCB). Su debilidad: en un mundo enorme o continuo, nunca visitas el *exacto* mismo estado dos veces, por lo que el recuento bruto siempre es 1 — razón por la cual existe el siguiente sabor.

### 2. Novedad por error de predicción: "No me esperaba *eso*"

Esta es la idea detrás del famoso **ICM** (Intrinsic Curiosity Module, Pathak et al. 2017) y su primo **RND** (Random Network Distillation, Burda et al. 2018). En lugar de contar, el agente mantiene un pequeño **modelo que intenta predecir qué sucede después** — "si estoy aquí y hago esto, ¿dónde termino?" — y se recompensa por **lo equivocado que estaba el modelo**:

```
bono de curiosidad = sorpresa = -log P( el estado que realmente alcancé | dónde estaba, qué hice )
```

- Una situación que el modelo nunca ha visto → predice mal → gran sorpresa → gran bono → "¡ve a explorar allí!"
- Una situación que el modelo ha visto cien veces → predice perfectamente → cero sorpresa → cero bono → "ya estuve allí, lo entiendo, a otra cosa".

**Analogía de la vida real:** un niño aprendiendo cómo funciona el mundo jugando. Empujar un vaso de la mesa la *primera* vez es fascinante (¡se rompió!). La centésima vez, ya sabías que se rompería — no es interesante. Curiosidad = la brecha entre lo que esperabas y lo que sucedió.

En nuestro código tabular, el "modelo" es solo una tabla de recuentos de transición, y "lo equivocado que estaba" es la sorpresa `-log P`. El ICM/RND real utiliza redes neuronales para que la misma idea funcione sobre píxeles brutos — pero el principio es idéntico.

> **¿Por qué dos versiones?** La basada en recuentos es extremadamente simple y un gran punto de referencia. La de error de predicción escala a mundos grandes que nunca se repiten y da una señal más *nítida*: en un entorno determinista, una vez que has visto una transición una vez, la sorpresa cae instantáneamente a ~0, mientras que un bono de recuento se desvanece lentamente como `1/sqrt(N)`. En nuestros experimentos, el agente de error de predicción resuelve MiniMontezuma en un par de docenas de episodios; el agente de recuento también llega allí, solo que más lentamente y de forma menos fiable.

## Qué hace nuestro código

`curiosity_bonus.py` entrena a un **Q-learner tabular** simple en `MiniMontezumaEnv` — un pequeño mundo de cuadrícula de dos habitaciones donde debes caminar hacia una **llave**, recogerla (ahora se abre la **puerta**), atravesarla y llegar al **tesoro**. La recompensa (+1) aparece *solo* en el tesoro, después de unos 15 movimientos perfectos. Ejecuta tres agentes y los grafica:

| Agente | Qué hace en MiniMontezuma |
|-------|-------------------------------|
| **epsilon-greedy (sin curiosidad)** | Deambula cerca del inicio, *nunca* llega a la llave, la puntuación se mantiene en 0 para siempre. |
| **bono basado en recuento** | Encuentra la llave de forma fiable; completa toda la cadena hasta el tesoro en un ~40% de los episodios. Funciona, solo que es un poco ruidoso. |
| **bono por error de predicción** | Llega por primera vez a la llave *y* al tesoro en unos 20–25 episodios; a medida que `beta` decae, converge a resolverlo en cada episodio. |

La figura muestra:
- una curva de aprendizaje: *P(llegar al tesoro)* durante el entrenamiento,
- una segunda curva para el hito intermedio *P(recoger la llave)*,
- y **mapas de calor de visitas de estados** de la cuadrícula — el agente sin curiosidad es una mancha apretada cerca del inicio; los agentes curiosos inundan *ambas* habitaciones.

## El mecanismo en una imagen

```
            sin curiosidad                     con bono de curiosidad
   recompensa: 0 0 0 0 0 0 0 0 ... 0 (+1?)     0 0 0 0 0 0 0 0 ... 0 (+1!)
               └──── nada de lo que aprender ──┘ └ + 0.4 0.3 0.9 0.2 ... ┘ (autogenerada,
                                                  densa, apunta "hacia lo nuevo")
   resultado:  camino aleatorio, nunca encuentra +1  barre el mundo sistemáticamente,
                                                  tropieza con +1, luego el bono se desvanece
```

El bono de curiosidad convierte el *"no he visto esto"* en recompensa, por lo que el agente **empuja deliberadamente hacia territorio inexplorado** en lugar de agitarse al azar. Y como el bono se reduce a medida que las cosas se vuelven familiares (y `beta` decae), una vez que el agente ha encontrado la recompensa real, deja de perder el tiempo de forma natural y empieza a explotar lo aprendido.

## Algunas advertencias honestas

- **El problema de la "televisión ruidosa" (noisy-TV problem).** Un agente de error de predicción puede quedar hipnotizado por una fuente de aleatoriedad pura (una televisión que muestra estática, dados rodando) — *nunca* puede predecirlo, por lo que la sorpresa nunca se desvanece. El truco real del ICM es predecir en un *espacio de características aprendido* que ignora las cosas que el agente no puede controlar; RND lo esquiva de forma diferente. Nuestro mundo de cuadrícula determinista no tiene televisión ruidosa, por lo que no nos enfrentamos a esto.
- **La curiosidad es un medio, no un fin.** Por eso `beta` decae. Un agente que se mantiene al máximo de curiosidad para siempre nunca se asienta para ganar realmente.
- **Escalar la exploración profunda sigue siendo difícil.** Un bono en la recompensa ayuda, pero el Q-learning tabular simple es lento para propagar el optimismo resultante a lo largo de una cadena larga (ver `compare_exploration.py`). Descifrar el Montezuma de píxeles necesitó potencia extra — RND con una red neuronal, DQN con bootstrap, Go-Explore.

## Palabras Clave para Recordar

| Palabra | Significado |
|------|---------|
| **Recompensa intrínseca** | Una recompensa que el agente genera para sí mismo, independiente de la recompensa del entorno |
| **Recompensa extrínseca** | La recompensa real del entorno (puntos, ganar/perder) |
| **Recompensa escasa** | La recompensa es cero en casi todas partes; solo la obtienes después de una secuencia correcta larga |
| **Novedad / sorpresa** | Qué tan nuevo o inesperado es un estado (o transición) — lo que recompensa la curiosidad |
| **Bono basado en recuento** | Novedad ≈ `1/sqrt(recuento de visitas)` — el bono de exploración clásico |
| **ICM** | Intrinsic Curiosity Module: novedad = error de predicción de un modelo hacia adelante (en un espacio de características aprendido) |
| **`beta`** | El peso del bono de curiosidad; generalmente se reduce hacia 0 para que el agente termine explotando |

## Resumen de una frase

> **Un bono de curiosidad es una recompensa autogenerada por la novedad — fabrica una señal densa de "ve a explorar allí" que arrastra al agente a través de mundos de recompensa escasa que de otro modo nunca resolvería, y luego se desvanece cortésmente una vez que todo es familiar.**
