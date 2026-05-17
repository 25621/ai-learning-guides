# PPO para Control Continuo: Haciendo que BipedalWalker camine

## Acciones Discretas vs. Continuas

Hasta ahora, todos los entornos que hemos resuelto tenían acciones **discretas**:
- CartPole: empujar IZQUIERDA o empujar DERECHA (2 opciones).
- LunarLander: no encender nada / izquierda / principal / derecha (4 opciones).

Pero los robots del mundo real necesitan acciones **continuas**:
- Un robot humanoide: "con qué fuerza empujar cada articulación" (cualquier valor de -1 a +1).
- Un coche: "exactamente cuánto girar el volante" (cualquier ángulo de -30° a +30°).
- Un brazo: "aplicar exactamente 2.3 Newtons en esta dirección".

**Ejemplo de la vida real:** Escribir en un teclado = discreto (pulsar A, B, C...). Escribir con un lápiz = continuo (mover la mano 2.3 cm a la derecha, presionar con 40g de fuerza...).

---

## La Política Gaussiana para Acciones Continuas

Para acciones continuas, en lugar de una distribución Categórica (elegir entre N categorías), utilizamos una **distribución Normal (Gaussiana)**:

```
Acción ~ Normal(μ, σ)
```

Donde:
- **μ (mu, media)**: El centro de la distribución — el valor de acción al que la red "apunta".
- **σ (sigma, desviación estándar)**: La dispersión — cuánta aleatoriedad / exploración añadir.

```
        Probabilidad
             │
        0.4 ─┤      ██████
             │    ████████████
        0.2 ─┤  ██████████████████
             │████████████████████████
             └──────────────────────── Valor de la acción
           -1  -0.5   0   0.5   1
                      ↑
                   media μ
```

**Ejemplo de la vida real:** Un arquero experto apunta al centro de la diana (μ). Sus flechas no aterrizan todas exactamente en el mismo punto — hay cierta dispersión (σ). A medida que practica, se vuelve más preciso (σ disminuye) mientras se mantiene centrado en el blanco.

---

## Nuestra red Actor-Critic Gaussiana

```
Estado (24 números) → [256 neuronas] → [256 neuronas] →
    ├── Actor: 4 valores de media (μ₁, μ₂, μ₃, μ₄)
    │          + 4 parámetros log_std (¡compartidos por todos los estados!)
    └── Crítico: 1 valor (V(s))
```

El `log_std` (logaritmo de la **desviación estándar** — una medida de dispersión o incertidumbre) es un **parámetro aprendible**, no depende del estado. Esto lo mantiene simple pero permite que la exploración cambie durante el entrenamiento.

**¿Por qué log_std en lugar de std?** La desviación estándar debe ser positiva. El uso de `log_std` permite que la red produzca cualquier número real (positivo o negativo), luego aplicamos `exp(log_std)` — la función exponencial, que es la inversa del logaritmo — para recuperar una std garantizada como positiva. Esto evita que la std sea alguna vez negativa o cero.

---

## Cálculo de la Log-Probabilidad para Acciones Continuas

Para acciones discretas: `log_prob = log(P(accion=IZQUIERDA))`

Para acciones continuas, la **distribución Normal** describe una curva suave en forma de campana alrededor de la media. Un único valor exacto tiene probabilidad cero en matemáticas continuas, por lo que usamos la altura de la curva en ese valor, llamada **pdf** (función de densidad de probabilidad):
```
log_prob = Σᵢ log[Normal(μᵢ, σᵢ).pdf(aᵢ)]
```

`log` significa logaritmo natural. Convierte valores de densidad diminutos en números estables que son más fáciles de optimizar para las redes neuronales. Sumamos todas las dimensiones de la acción (4 para BipedalWalker), porque la acción completa es un vector de 4 números.

**Ejemplo de la vida real:** ¿Cuál es la probabilidad de que mañana haga exactamente 5.732...°C? Para el clima continuo, mirarías la curva de la distribución Normal y verías qué tan alta es en ese punto exacto. Las temperaturas más probables (cerca de la media) tienen mayor probabilidad.

---

## BipedalWalker: Un desafío de caminata

BipedalWalker-v3 es un robot 2D que debe aprender a caminar sin caerse:

```
          O (cabeza)
         /│\
        / │ \
       /  │  \
      I   │   D   ← dos piernas, cada una con una articulación de rodilla
     / \  │  / \
    ●   ● │ ●   ●  ← 4 motores (cadera/rodilla para cada pierna)
```

**Espacio de estados (24 números):**
- Casco: ángulo, velocidad angular, velocidad horizontal, velocidad vertical (4 números).
- Articulaciones: 4 motores (2 caderas, 2 rodillas) cada uno proporcionando ángulo y velocidad, más 2 sensores de contacto con el suelo (uno para cada pierna) (10 números).
- 10 sensores de distancia LIDAR (lecturas de distancia que ven el suelo por delante) (10 números).

**Espacio de acciones (4 valores continuos, cada uno en [-1, 1]):**
Los valores de acción controlan el **par motor** (torque - la fuerza de rotación aplicada por los motores) para exactamente 4 articulaciones (no se aplican acciones directamente al casco):
- Par de cadera pierna 1, Par de rodilla pierna 1, Par de cadera pierna 2, Par de rodilla pierna 2.

**Recompensas:**
- +300 por llegar a la meta (lado derecho).
- -100 por caerse (tocar el suelo con el cuerpo).
- Pequeña recompensa por cada paso de progreso hacia adelante.
- Pequeña penalización por cada uso del motor (eficiencia de la recompensa).

**Resuelto cuando:** La recompensa media sea > 300 durante 100 episodios.

---

## Diferencia clave con el PPO discreto

Todo es igual EXCEPTO:

| | PPO Discreto | PPO Continuo |
|---|---|---|
| **Política** | Categorical(logits) | Normal(μ, σ) |
| **Muestreo** | accion = muestra de {0,1,...,N} | accion = μ + σ × ruido |
| **log_prob** | log P(accion=k) | Σ log Normal(μᵢ, σᵢ).pdf(aᵢ) |
| **Restricción (Clamp)** | No necesaria | Restringir acciones a [-1, 1] |

Los **logits** son puntuaciones brutas y no normalizadas para acciones discretas. Una política categórica las convierte en probabilidades con **softmax** — una función que toma cualquier conjunto de números y los aplasta en una distribución de probabilidad válida (todos los valores positivos, sumando 1). Por ejemplo, los logits [2.0, 1.0, 0.5] se convierten en probabilidades [0.59, 0.24, 0.17]. El PPO continuo **no** utiliza softmax para la acción en sí, porque la acción no se elige de un menú fijo. En su lugar, la política genera la media y la desviación estándar de una distribución Normal, y luego toma muestras de pares motores con valores reales a partir de ella.

**Clamp** significa forzar un valor dentro de un rango válido. El código utiliza `action.clamp(-1, 1)` para que el entorno nunca reciba una orden de motor fuera de sus límites permitidos.

**Clip** en PPO significa algo diferente: PPO recorta la relación de probabilidad dentro de la pérdida, como se explica en la [sección de recorte de PPO](./ppo_scratch_explained.md#el-truco-del-recorte). La restricción de acción (clamping) protege la interfaz del entorno; el recorte (clipping) de PPO protege la actualización de la política.

---

## Caminar desde cero: Lo que aprende el agente

**Entrenamiento temprano (recompensas negativas):** El robot se agita aleatoriamente, cae inmediatamente. Cada episodio termina en un choque en cuestión de segundos.

**Entrenamiento medio:** El robot descubre que mover las piernas alternativamente crea un progreso hacia adelante. Empieza a dar pasos pequeños y torpes — la recompensa se vuelve menos negativa.

**Entrenamiento tardío:** Surge una **marcha** (gait) suave y eficiente. Una marcha es un patrón de movimiento repetido, como alternar los pasos izquierdo y derecho. El robot se ajusta dinámicamente al terreno irregular utilizando sus sensores LIDAR para adaptar sus pasos en tiempo real.

**Ejemplo de la vida real:** Un bebé aprendiendo a caminar:
1. Se cae inmediatamente (recompensa negativa).
2. Da un paso, se cae (ligeramente menos negativa).
3. Da unos pocos pasos (pequeña recompensa positiva).
4. Camina por la habitación (¡gran recompensa positiva!).

---

## Por qué BipedalWalker necesita PPO (y no REINFORCE)

- Los **episodios de BipedalWalker** pueden durar hasta 1600 pasos (¡mucho más largos que los de CartPole!).
- Las **recompensas son escasas**: las recompensas por progreso hacia adelante son diminutas por paso.
- **REINFORCE necesitaría** miles de episodios completos para obtener una señal útil.

Las actualizaciones de n-pasos de PPO con [GAE (Generalized Advantage Estimation)](./ppo_scratch_explained.md#gae-smarter-advantage-estimates) permiten al robot aprender de episodios incompletos:
> "Aunque me haya caído después de 50 pasos, esos pasos mostraron ALGO de progreso hacia adelante. Permítanme usar una estimación de retorno de 50 pasos en lugar de esperar a que termine el episodio".

---

## Resultados

Después de 500 actualizaciones (≈ 1 millón de pasos de entorno):
- El robot realiza progresos visibles desde el agite aleatorio hacia algún movimiento hacia adelante.
- Mejora constante en la curva de aprendizaje.
- La convergencia total a una recompensa > 300 requiere más entrenamiento (5-10M de pasos).

La curva de aprendizaje muestra la característica "curva en S" del control continuo:
1. Progreso inicial lento (estabilidad del aprendizaje).
2. Mejora rápida (descubrimiento de la marcha).
3. Refinamiento gradual (optimización de la marcha).

---

## Conclusiones clave

| Concepto | En lenguaje sencillo |
|----------|---------------|
| **Política Gaussiana** | En lugar de elegir de un menú, lanza un dardo a un rango de valores |
| **μ (media)** | Adónde "apunta" la política |
| **σ (std)** | Cuánta aleatoriedad / exploración usa la política |
| **log_std como parámetro aprendible** | Una tasa de exploración global actualizada mediante optimización basada en gradientes (ascenso de gradiente en la recompensa, o equivalentemente descenso de gradiente en la pérdida de PPO) — igual que cualquier otro peso de la red |
| **Control continuo** | Controlar salidas de valor real (pares motores, fuerzas, ángulos) |

---

## ¿Qué sigue?

PPO tiene muchos **hiperparámetros** — configuraciones que eliges antes de que comience el entrenamiento (a diferencia de los *parámetros* como los pesos de la red, que se aprenden automáticamente). Algunos ejemplos son `clip_eps`, la tasa de aprendizaje, el número de épocas y el tamaño del lote.

¿Qué tan sensible es PPO a estas elecciones? ¡`ppo_hyperparams.py` realiza experimentos variando sistemáticamente cada hiperparámetro y muestra el efecto en la velocidad y estabilidad del aprendizaje!
