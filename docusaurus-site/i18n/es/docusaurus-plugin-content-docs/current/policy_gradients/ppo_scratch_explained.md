# PPO: Actualizaciones de política seguras y constantes

## El Problema con A2C

Imagina que estás aprendiendo a equilibrar un palo de escoba sobre tu dedo. Después de semanas de práctica, ¡puedes mantenerlo durante 30 segundos!

Ahora tu entrenador te da un consejo: "Inclina tu muñeca un poco más hacia la izquierda".

**Buen consejo → cambio cuidadoso → sigues equilibrándolo durante 30 segundos ✓**

Pero, ¿qué pasa si el entrenador reacciona exageradamente y dice: "¡INCLÍNATE HACIA LA IZQUIERDA INMEDIATAMENTE!"?
Corriges en exceso → el palo de escoba se cae → has perdido semanas de progreso.

Este es el problema de A2C: **las grandes actualizaciones de gradiente pueden destruir una buena política**.

**PPO (Proximal Policy Optimization)** es un sistema de seguridad que evita esto.

---

## La Idea Central: Mantenerse cerca de lo que funcionaba

La restricción clave de PPO:

> **"No cambies demasiado la política en una sola actualización".**

Antes de una actualización, tenemos la política "antigua" π_old.
Después de la actualización, tenemos la política "nueva" π_new.

PPO mide cuánto ha cambiado la política con la **relación de probabilidad** (probability ratio):

```
r(θ) = π_new(a|s) / π_old(a|s)
```

- r = 1.0: política sin cambios.
- r = 1.5: la nueva política tiene un 50% más de probabilidades de realizar esa acción.
- r = 0.5: la nueva política tiene un 50% menos de probabilidades de realizar esa acción.

**Ejemplo de la vida real:** Eres un chef ajustando una receta.
- r = 1.0: la misma cantidad de sal que antes.
- r = 2.0: el doble de sal — ¡demasiado extremo!
- r = 0.9: un 10% menos de sal — un cambio pequeño y seguro.

---

## El truco del Recorte (Clipping)

PPO recorta (clips) la relación para que se mantenga dentro de [1-ε, 1+ε] (normalmente ε = 0.2):

```
L_CLIP = E[min(r(θ) · A,  clip(r(θ), 1-ε, 1+ε) · A)]
```

Vamos a desglosarlo:

**Caso 1: La acción fue BUENA (A > 0)**

Queremos realizar esta acción más veces (r > 1). Pero limitamos cuánto aumentamos:
```
si r > 1.2: recortar a 1.2, no hay más incentivo para presionar más
```
Esto evita que nos columpiemos DEMASIADO en una dirección.

**Caso 2: La acción fue MALA (A < 0)**

Queremos realizar esta acción menos veces (r < 1). Pero, de nuevo, limitamos:
```
si r < 0.8: recortar a 0.8, no hay más penalización por ir más allá
```

**Visual:**
```
ε = 0.2, por lo que la ventana de relación segura es de 0.8 a 1.2.

Acción BUENA (A > 0): aumenta la probabilidad de la acción, pero deja de recompensarla después de 1.2
relación r:    0.6      0.8      1.0      1.2      1.4
incentivo:      ↑        ↑        ↑        ↑        -
significado: demasiado bajo  ok   antiguo   max   recortado

Acción MALA (A < 0): disminuye la probabilidad de la acción, pero deja de penalizarla por debajo de 0.8
relación r:    0.6      0.8      1.0      1.2      1.4
incentivo:      -        ↓        ↓        ↓        ↓
significado: recortado  max    antiguo    ok  demasiado alto
```

El `-` marca la región plana recortada. En esa región, hacer que la relación de probabilidad sea aún más extrema no mejora el objetivo, por lo que PPO no tiene ningún incentivo extra para presionar más allá.

**Ejemplo de la vida real:** El limitador de velocidad de un coche. Puedes acelerar, pero una vez que llegas a 120 km/h, el limitador entra en acción y no te deja ir más rápido. Te mantiene seguro sin impedirte moverte.

---

## Por qué esto evita actualizaciones catastróficas

Una **actualización catastrófica** es cuando un gran cambio en la política destruye por completo todo lo que el agente ha aprendido — horas de entrenamiento perdidas en un solo paso de gradiente.

Sin recorte: un gran paso de gradiente podría cambiar la política drásticamente.
Con recorte: el gradiente es cero fuera de [1-ε, 1+ε], por lo que la política solo puede moverse un poco por paso.

**Ejemplo de la vida real:** Un buen cirujano realiza cortes pequeños y precisos, no cortes grandes y amplios. PPO es el "cirujano cuidadoso" de los optimizadores de RL.

---

## GAE: Estimaciones de ventaja más inteligentes {#gae-smarter-advantage-estimates}

PPO utiliza **Estimación de Ventaja Generalizada** (Generalized Advantage Estimation, GAE) para calcular la ventaja:

```
δ_t = r_t + γ · V(s_{t+1}) - V(s_t)          (error TD)
A_t = δ_t + γλ · δ_{t+1} + (γλ)² · δ_{t+2} + ...
```

GAE tiene un parámetro λ (lambda):
- λ = 0: usar solo el error TD de un paso (baja varianza, alto sesgo).
- λ = 1: usar los retornos completos de Monte Carlo (alta varianza, bajo sesgo).
- λ = 0.95: ¡un buen equilibrio entre ambos!

**Ejemplo de la vida real:** Planificar un viaje por carretera.
- λ=0: solo mirar las próximas 5 millas (seguro, pero podría perderse un atajo más adelante).
- λ=1: considerar todo el viaje de 500 millas (más información, pero mucha incertidumbre).
- λ=0.95: mirar lejos pero dar más peso a las carreteras cercanas ← ¡el mejor equilibrio!

---

## Múltiples Épocas: Reutilización eficiente de los datos

Después de recopilar un lote de experiencia (rollout), REINFORCE lo descarta después de UNA actualización.

PPO reutiliza cada lote durante **K épocas** (normalmente de 4 a 10 pasadas por los mismos datos):

```
Recopilar 512 pasos × 4 entornos = 2048 transiciones
Época 1: 32 minilotes × actualizar cada uno
Época 2: barajar, 32 minilotes más × actualizar cada uno
Época 3: ...
Época 4: ...
```

**¿Qué es un "minilote" (minibatch)?** Actualizar con las 2048 transiciones a la vez es lento y consume mucha memoria; actualizar una transición cada vez es ruidoso. Un **minilote** es un trozo pequeño intermedio — aquí, 2048 ÷ 32 = **64 transiciones por minilote**. Calculamos un paso de gradiente por minilote, de modo que cada época realiza 32 actualizaciones pequeñas y estables en lugar de una enorme. (Esta es la misma idea de minilote que se usa en todo el aprendizaje profundo — ver [descenso de gradiente por minilotes](https://es.wikipedia.org/wiki/Descenso_de_gradiente_estoc%C3%A1stico#Descenso_de_gradiente_por_minilotes)).

El recorte garantiza que estas pasadas múltiples no se pasen de largo — ¡sin recorte, las múltiples épocas destruirían la política al empujarla demasiado lejos!

**Ejemplo de la vida real:** Un estudiante tiene 30 problemas de práctica.
- REINFORCE: hacer cada problema una vez, aprender un poco, tirarlos.
- PPO: hacer cada problema 4 veces (desde diferentes ángulos cada vez), recortar tus cambios para no memorizar patrones incorrectos.

---

## La pérdida (loss) completa de PPO

```
L = L_CLIP - c₁ · L_entropy + c₂ · L_critic

L_CLIP    = gradiente de política recortado
L_entropy = bono de entropía (fomenta la exploración)  
L_critic  = MSE entre V(s) y los retornos
```

Coeficientes típicos: c₁ = 0.01 (entropía), c₂ = 0.5 (crítico).

**Dos términos que vale la pena desglosar:**

- **Gradiente de política**: la mitad "actor" de la pérdida. Utiliza la señal del gradiente para empujar la política hacia acciones con mayor ventaja y alejarla de acciones con menor ventaja. Esta es la misma idea central introducida en REINFORCE — ver el [tutorial de REINFORCE](./reinforce_cartpole_explained.md#la-forma-antigua-frente-a-la-forma-nueva) para la intuición. PPO simplemente le añade el envoltorio de recorte.
- **MSE (Error Cuadrático Medio)**: la mitad "crítico" de la pérdida. El crítico V(s) predice el retorno esperado desde un estado; comparamos su predicción con el retorno real y elevamos al cuadrado la diferencia: `MSE = media((V(s) - retorno)²)`. Al elevar al cuadrado, se castigan más los errores grandes que los pequeños y se obtiene una señal suave y diferenciable para el entrenamiento. (Pérdida de regresión estándar — ver [error cuadrático medio](https://es.wikipedia.org/wiki/Error_cuadr%C3%A1tico_medio)).

---

## Los Resultados

```
Actualización 200 | Recompensa media: ~120
Actualización 400 | Recompensa media: ~280
Actualización 800 | Recompensa media: ~280-300
```

PPO en CartPole muestra una mejora constante pero tiende a estancarse alrededor de 280-300. (Una **meseta** significa que la curva de aprendizaje se aplana — la recompensa deja de mejorar incluso cuando el entrenamiento continúa. La política ha encontrado una estrategia localmente buena pero no está progresando más). Esto es realmente lo esperado — PPO está diseñado para entornos más difíciles y con episodios más largos.

Una observación interesante: **¡REINFORCE resolvió CartPole más rápido!** (500 de media frente a 300 de media).

¿Por qué? Los episodios de CartPole son cortos (≤500 pasos), por lo que los retornos exactos de REINFORCE son muy precisos. Las estimaciones por bootstrap de PPO añaden una complejidad innecesaria. PPO brilla de verdad en entornos donde esperar episodios completos es poco práctico (como BipedalWalker).

**¿Qué es "BipedalWalker"?** BipedalWalker (específicamente `BipedalWalker-v3` en [Gymnasium](https://gymnasium.farama.org/environments/box2d/bipedal_walker/)) es un entorno clásico de referencia de RL: un robot de 2 piernas que debe aprender a caminar hacia adelante a través de un terreno irregular sin caerse. A diferencia de las dos acciones discretas de CartPole (IZQUIERDA / DERECHA), BipedalWalker tiene **acciones continuas** — cuatro valores de par motor, uno para cada articulación de las piernas, cada uno un número real en [-1, 1]. Los episodios pueden durar miles de pasos, que es exactamente el régimen donde la eficiencia de datos y la estabilidad de PPO dan sus frutos.

---

## Ecuaciones Clave

```
Relación:       r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
Pérdida clip:   L_CLIP = E[min(r_t A_t, clip(r_t, 1-ε, 1+ε) · A_t)]
GAE:            A_t = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}
```

---

## Conclusiones clave

| Concepto | En lenguaje sencillo |
|----------|---------------|
| **Relación r(θ)** | Cuánto cambió la política en esta acción. |
| **Recorte ε** | El límite de seguridad — no cambies la política más que esto. |
| **GAE** | Una forma inteligente de estimar ventajas mirando varios pasos hacia adelante. |
| **Eficiencia de datos** | Cada rollout se recoge de varios entornos paralelos (experiencia decorrelacionada y estable) y luego se reutiliza para K épocas de actualizaciones de minilotes — el recorte mantiene seguras estas pasadas repetidas. |

---

## ¿Qué sigue?

Hasta ahora, todos nuestros entornos tienen acciones **discretas** (pulsar IZQUIERDA o DERECHA).

Los robots reales necesitan controlar acciones **continuas** — como "aplicar exactamente 0.73 Newtons de fuerza".

¡`ppo_continuous.py` extiende PPO a acciones continuas usando una **política Gaussiana**, y lo prueba en el entorno BipedalWalker-v3, que es mucho más difícil!
