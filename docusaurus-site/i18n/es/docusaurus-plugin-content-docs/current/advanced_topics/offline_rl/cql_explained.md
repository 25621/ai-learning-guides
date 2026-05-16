# Q-Learning Conservador (CQL) 🛡️

## ¿Qué es?

Imagina que estás aprendiendo a invertir dinero leyendo un libro de contabilidad gigante de operaciones bursátiles pasadas realizadas por otras personas. El libro tiene compras, ventas y mantenimientos, pero **no hay registro de ninguna operación que nadie haya realizado realmente**.

Ahora imagina que un estudiante demasiado confiado mira el libro y dice:
*"¿Qué pasaría si alguien hubiera comprado boletos de lotería todos los lunes? ¡Esa habría sido una operación increíble!"*

El problema: **el libro no tiene datos sobre la compra de lotería los lunes**, por lo que el estudiante solo está alucinando. Sin embargo, esa operación alucinada se ve genial en el papel, por lo que la "política" del estudiante quiere seguir haciéndola.

Ese problema de la alucinación es el **desplazamiento de la distribución** (distribution shift): un aprendiz offline ama las acciones que el conjunto de datos nunca probó, porque no hay datos que contradigan el optimismo. CQL es la cura.

---

## Cómo el Q-Learning sale mal Offline

El objetivo (target) del Q-learning normal es:

```
target(s, a) = r + γ · max_{a'} Q(s', a')
```

Ese `max_{a'}` es el peligro. Cuando el conjunto de datos nunca registró la acción `a'` en el estado `s'`, la red simplemente *adivina* un valor Q, y las redes neuronales tienden a **sobreestimar** Q para entradas no vistas. El objetivo hereda la sobreestimación, la red aprende a predecir ese número mayor y, en el siguiente paso, **extrapolamos** (proyectamos incluso más allá de lo que los datos soportan) aún más alto. La política persigue un fantasma.

Si pudieras seguir recopilando más datos, esto se autocorregiría (la acción fantasma resulta ser mala en la realidad). Pero **en el RL offline no puedes recopilar más datos.** El fantasma es para siempre.

---

## El truco de CQL

CQL (Kumar et al., 2020) añade un término de penalización a la pérdida:

```
cql_loss(s)  =  log Σ_a exp Q(s, a)   -   Q(s, a_dataset)
```

Dos piezas:

1. **`log Σ_a exp Q(s, a)`** (léase: *"log-sum-exp sobre todas las acciones"*) es un **máximo suave** (soft maximum) sobre todas las acciones — una aproximación suave y diferenciable de `max` que considera todas las acciones a la vez en lugar de seleccionar rígidamente un ganador. Penalizarlo reduce los valores Q **en todos los ámbitos** (empujando todas las predicciones hacia abajo uniformemente), especialmente para las acciones con el Q *más alto*, que es exactamente donde viven las alucinaciones.
2. **`- Q(s, a_dataset)`** recompensa el Q alto en la acción que el conjunto de datos realmente registró, protegiendo los valores Q dentro de la distribución de la reducción anterior.

Efecto neto: **Q se reduce en las acciones no vistas, se aumenta en las acciones vistas.** El Q aprendido se convierte en un *límite inferior* del Q verdadero. La política **`argmax`** (la regla que simplemente elige la acción con el Q más alto) deja de perseguir fantasmas.

Pérdida completa:

```
L  =  Bellman_MSE   +   α · cql_loss
```

(Donde **`Bellman_MSE`** es el error estándar del Q-learning normal, que mide cuánto discrepa la suposición actual de la red con su propia suposición futura).

`α` es la perilla del conservadurismo. Demasiado pequeña → el desplazamiento de la distribución vuelve a aparecer. Demasiado grande → el agente es tan conservador que nunca mejora más allá de los datos.

---

## Ejemplos de la vida real

- **Entrenador de ajedrez conservador.** Solo puedes aprender de las partidas ya jugadas. Un entrenador imprudente dice "¡este movimiento hipotético sin precedentes podría ser brillante!". CQL es el entrenador que dice "no tenemos datos sobre eso, ciñámonos a los movimientos que los jugadores reales han probado".
- **Elecciones de menú de restaurante.** Las reseñas de Yelp nunca cubren los platos que no están en el menú. Una política ingenua recomendaría los platos fuera de carta basándose en calificaciones de cinco estrellas alucinadas. CQL recomienda solo lo que se ha pedido suficientes veces como para confiar.
- **Agarre robótico a partir de registros.** El robot tiene vídeos de agarres de tazas, botellas y libros, pero nunca de un cuchillo. CQL se niega a recomendar con confianza "agarra el cuchillo por la hoja".

---

## Qué hace nuestro código

El script `cql.py`:

1. **Carga los cuatro conjuntos de datos** creados por `d4rl_dataset.py`.
2. **Elige `medium-replay`** como conjunto de entrenamiento — es el más realista (calidad mixta) y el más dañino para los métodos ingenuos.
3. **Entrena a tres agentes puramente offline**, en condiciones idénticas excepto por `α`:
   - `α = 0`   →  DQN offline ingenuo (sin penalización — el estándar de referencia roto)
   - `α = 1.0` →  CQL suave
   - `α = 5.0` →  CQL fuerte
4. **Evalúa a cada uno cada 2,500 pasos de gradiente** ejecutándolos de forma codiciosa en el entorno real (10 episodios). Este es el *único* contacto con el entorno; el entrenamiento en sí nunca ve el entorno.
5. **Grafica las curvas de aprendizaje** en `outputs/cql.png`.

---

## Qué deberías ver

Una ejecución típica imprime algo como:

```
Retornos de evaluación finales (promedio sobre 10 episodios, greedy):
  DQN offline ingenuo (alpha=0)         ->  ~30-150  (inestable; a menudo falla)
  CQL (alpha=1.0)                     ->  ~300-450
  CQL (alpha=5.0)                     ->  ~450-500
```

En el gráfico de la curva de aprendizaje:

- La **curva roja** (`α = 0`) sube temprano y luego, a menudo, **cae por un acantilado** una vez que las alucinaciones de desplazamiento de la distribución infectan el **objetivo de Bellman** (el número que usamos como "respuesta correcta" al entrenar la red Q: `r + γ · max Q(s', ·)`). Cuando los valores Q fantasmales contaminan ese objetivo, cada paso de gradiente empeora las cosas. La **pérdida de Bellman** (el MSE entre la predicción de la red Q y el objetivo de Bellman) se ve bien — esa es la **traición** del problema: la red es perfectamente coherente con sus propias creencias erróneas, por lo que la pérdida no da ninguna advertencia.
- La **curva naranja** (`α = 1.0`) sube más lentamente pero **se mantiene arriba**.
- La **curva verde** (`α = 5.0`) es la más estable y, por lo general, la mejor.

El panel de pérdida de Bellman muestra otra señal: la pérdida de DQN ingenuo puede mantenerse pequeña mientras su política es terrible, porque la red es internamente coherente con sus propias alucinaciones.

---

## Dónde se sitúa CQL en el campo

CQL fue un *gran* avance porque dio una solución sencilla y con principios al desplazamiento de la distribución. El linaje:

```
DQN (online)
   │
   ▼
DQN offline ingenuo  ── se rompe debido al desplazamiento de la distribución
   │
   ▼
CQL (Kumar 2020)     ── añade una penalización conservadora: Q es un límite inferior
   │
   ▼
IQL (Kostrikov 2021) ── en primer lugar, nunca consulta Q sobre acciones no vistas
   │
   ▼
Decision Transformer (Chen 2021)  ── omite Q por completo, trata el RL como modelado de secuencias
                                      (predice la *siguiente acción* dados los estados pasados y
                                       un retorno total deseado, exactamente como un LLM
                                       predice la siguiente palabra)
```

Cada paso en este linaje es una respuesta diferente a la misma pregunta: **¿cómo evito preguntarle a mi red Q sobre cosas que nunca ha visto?**

---

## Palabras Clave para Recordar

| Palabra | Significado |
|------|---------|
| **Desplazamiento de la distribución** | La política entrenada quiere acciones fuera de los datos |
| **Fuera de la distribución (OOD)** | Un par (s, a) que el conjunto de datos nunca registró |
| **Q verdadero** | El retorno futuro real esperado por tomar la acción `a` en el estado `s`, si pudiéramos medirlo perfectamente |
| **Q conservador** | Una función Q aprendida que intenta mantenerse por debajo del Q verdadero en lugar de prometer demasiado |
| **Logsumexp** | Una aproximación suave y diferenciable de `max` |
| **Alpha (α)** | La perilla de conservadurismo de CQL — con qué fuerza empujar Q hacia abajo en acciones OOD |

---

## Resumen de una frase

> **CQL añade una "penalización de pesimismo" que castiga los valores Q altos en acciones que el conjunto de datos nunca probó, para que la política no pueda enamorarse de las alucinaciones.**
