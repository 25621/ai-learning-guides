# Funciones de Valor de Estado 🗺️

## ¿Qué es un "Estado"?

Piensa en jugar a un juego de mesa. En cualquier momento, estás parado en *un* cuadrado del tablero. Ese cuadrado es tu **estado** — es donde estás ahora mismo.

En nuestro juego de cuadrícula de 4×4, hay 16 cuadrados (estados). Cada cuadrado es un lugar en el que el agente puede estar.

---

## ¿Qué es un "Valor"?

Aquí viene la pregunta mágica: **"Si estoy parado en este cuadrado ahora mismo, ¿cuánto tesoro puedo esperar recoger antes de que termine el juego?"**

¡Esa respuesta es el **valor** de ese estado!

Un cuadrado con un **valor alto** significa: "Este es un lugar genial — ¡probablemente recogeré mucho tesoro desde aquí!".

Un cuadrado con un **valor bajo** significa: "Oh, oh — desde aquí, las cosas suelen salir mal".

**Ejemplo de la vida real:** Imagina que estás jugando al escondite. Si te escondes detrás de un árbol grande (un lugar genial), tu probabilidad de ganar es alta — ¡ese es un estado de alto valor! Si te escondes en medio de una habitación vacía, probablemente te encontrarán — ese es un estado de bajo valor.

---

## Nuestro Mundo de Cuadrícula (Grid World)

He aquí el tablero de juego que utilizamos:

```
S  .  .  .      S = Inicio (Start)
.  A  .  A      A = Agujero (recompensa -1, el juego termina)
.  .  .  A      M = Meta (recompensa +1, el juego termina)
A  .  .  M      . = Cuadrado seguro vacío
```

- Si llegas a **M** (Meta): obtienes **+1 punto** 🎉
- Si pisas un **A** (Agujero): obtienes **-1 punto** 😢
- Otros pasos: **0 puntos**

Utilizamos γ (gamma) = 0.99, lo que significa que las recompensas futuras cuentan casi tanto como las recompensas inmediatas. (¡Un caramelo mañana es casi tan bueno como un caramelo hoy!).

---

## Dos Planes Diferentes (Políticas)

Probamos dos políticas y calculamos el valor de cada cuadrado para cada una:

### Política 1: Aleatoria Uniforme
Elegir al azar arriba, abajo, izquierda o derecha con la misma probabilidad.

```
Valores (Política Aleatoria Uniforme):
-0.912  -0.932  -0.912  -0.942
-0.929   (A)   -0.898   (A)
-0.901  -0.801  -0.696   (A)
 (A)   -0.630  -0.104   (M)
```

Casi todas partes son **negativas** — ¡la política aleatoria cae en los agujeros tan a menudo que estar en cualquier lugar es bastante malo!

---

### Política 2: Sesgada hacia la Meta
Prefiere moverse a la derecha y hacia abajo (hacia la meta), pero a veces sigue yendo en otras direcciones.

```
Valores (Política Sesgada hacia la Meta):
-0.838  -0.895  -0.814  -0.961
-0.798   (A)   -0.665   (A)
-0.595  -0.143  -0.213   (A)
 (A)    0.254   0.673   (M)
```

¡Ahora los cuadrados cerca de la **Meta** tienen **valores positivos** (0.254 y 0.673)! La política inteligente hace que esos cuadrados sean buenos lugares para estar.

---

## Qué significan los colores en nuestra imagen

En nuestra visualización:
- **Cuadrados verdes** = valor alto (lugares geniales para estar)
- **Cuadrados rojos** = valor bajo (¡evita estos!)
- **Cuadrados amarillos** = algo intermedio

Puedes ver el **gradiente**: los valores se vuelven más verdes a medida que te acercas a la Meta y más rojos cerca de los Agujeros.

---

## ¿Por qué nos importan los valores?

¡Los valores son la *base* del aprendizaje por refuerzo! Una vez que conoces el valor de cada estado, puedes tomar decisiones inteligentes:

> "Estoy en el cuadrado A. Puedo ir al cuadrado B (valor = 0.5) o al cuadrado C (valor = -0.3). ¡Iré al B, tiene un valor más alto!".

Así es exactamente como muchos algoritmos de RL (como el Q-learning) aprenden a tomar buenas decisiones sin que se les digan las reglas.

**Ejemplo de la vida real:** Imagina que estás eligiendo en qué cola ponerte en el supermercado. Cada cola es un "estado". El valor de ese estado es la rapidez con la que saldrás por la caja. Miras las colas (observas los estados) y eliges la que tiene el valor más alto (espera más corta + menos artículos).

---

## Cómo calculamos los valores

Utilizamos la **Evaluación de Política Iterativa**, que funciona así:

1. Inicio: suponer que todos los valores son 0.
2. Actualización: para cada cuadrado, calcular cuál *debería* ser el valor basándose en adónde te lleva la política a continuación.
3. Repetir hasta que los valores dejen de cambiar (converjan).

Matemáticamente: **V(s) = Σ_a π(a|s) × [R(s,a) + γ × V(siguiente_estado)]**

En lenguaje sencillo: "El valor de este cuadrado = la recompensa media que obtendré ahora mismo + un poco del valor de donde sea que acabe".

---

## Palabras Clave para Recordar

- **Estado**: Donde estás ahora mismo (un cuadrado del tablero).
- **Valor V(s)**: Recompensa total esperada comenzando desde el estado s.
- **Política**: Tu plan sobre qué hacer en cada estado.
- **Factor de descuento γ**: Cuánto te importan las recompensas futuras (0.99 = ¡mucho!).
- **Evaluación de Política**: Calcular los valores para cada estado bajo una política dada.

La gran idea: **Algunos lugares son mejores que otros — ¡y la función de valor te dice exactamente qué tan bueno es cada lugar!**
