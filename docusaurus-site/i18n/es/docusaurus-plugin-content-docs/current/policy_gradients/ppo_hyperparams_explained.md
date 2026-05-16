# Sensibilidad de los Hiperparámetros de PPO: ¿Qué es lo que más importa?

## Por qué importan los hiperparámetros

Imagina que estás horneando un pastel de chocolate. La receta pide:
- 2 huevos
- 200g de harina
- 1 cucharadita de levadura en polvo
- 35 minutos a 180°C

Si usas 10 huevos, el pastel explota. Si usas 0.1 cucharaditas de levadura, no sube. Si lo horneas a 300°C durante 10 minutos, se quema por fuera y se queda crudo por dentro.

**Los hiperparámetros en PPO son como los ingredientes y los ajustes del horno.** La combinación adecuada funciona de maravilla; los ajustes incorrectos pueden impedir el aprendizaje por completo.

Este script prueba sistemáticamente 3 hiperparámetros clave cambiando solo UNO a la vez, ejecutando cada ajuste con 3 semillas aleatorias diferentes y comparando los resultados.

---

## Los tres experimentos

### Experimento 1: Epsilon de recorte (Clip Epsilon, ε)

```
ε = 0.05   (muy conservador — solo se permiten cambios minúsculos en la política)
ε = 0.2    (estándar — equilibrio entre seguridad y velocidad)
ε = 0.4    (agresivo — permite grandes cambios en la política)
```

**¿Qué controla ε?**

ε es el tamaño de la "ventana de seguridad" alrededor de la antigua política:
```
la relación debe mantenerse en [1 - ε,  1 + ε]
ε=0.05: relación en [0.95, 1.05]  ← cambios minúsculos
ε=0.2:  relación en [0.80, 1.20]  ← estándar  
ε=0.4:  relación en [0.60, 1.40]  ← cambios grandes
```

**Ejemplo de la vida real:** Piensa en ε como "qué tanto se te permite girar el volante del coche en un solo movimiento".
- ε=0.05: Como conducir sobre hielo — solo ajustes minúsculos.
- ε=0.2:  Conducción normal — giros razonables.
- ε=0.4:  Piloto de carreras — giros agresivos, riesgo de **trompear** (perder el control porque el cambio es demasiado drástico, como un coche derrapando fuera de la carretera).

**Resultados esperados:**
- ε=0.05: Aprendizaje lento pero estable (demasiado cauteloso).
- ε=0.2:  Buen equilibrio (el **valor "Ricitos de Oro"** — ni demasiado pequeño, ni demasiado grande, justo el adecuado — llamado así por el cuento de hadas donde Ricitos de Oro elige la avena que no está ni demasiado caliente ni demasiado fría).
- ε=0.4:  Puede aprender rápido pero puede **pasarse y oscilar** (pasarse = superar la política óptima; oscilar = rebotar de un lado a otro a su alrededor sin asentarse, como un péndulo que oscila demasiado en ambas direcciones).

---

### Experimento 2: Tasa de aprendizaje (Learning Rate)

```
lr = 1e-4  (lento pero estable)
lr = 3e-4  (estándar)
lr = 1e-3  (rápido pero arriesgado)
```

**¿Qué controla la tasa de aprendizaje?**

La tasa de aprendizaje es como el "tamaño del paso" al subir una colina (cada paso = una actualización de los pesos de la red neuronal, moviéndola ligeramente en la dirección que mejora la recompensa):
- Demasiado pequeña: Tarda una eternidad en llegar a la cima (converge lentamente).
- Demasiado grande: Te pasas de la cima y caes por el otro lado (**diverge** — la recompensa del entrenamiento colapsa o fluctúa salvajemente en lugar de mejorar de forma constante).
- Justo la adecuada: Progreso constante hacia la cumbre.

**Ejemplo de la vida real:** Afinar la cuerda de una guitarra.
- lr=1e-4: Giros minúsculos de la **clavija** (el mando que giras para apretar o aflojar una cuerda) — tarda una eternidad pero es preciso.
- lr=3e-4: Afinación normal — encuentra el tono adecuado en unos pocos giros.
- lr=1e-3: Grandes **tirones** (tirones bruscos y repentinos) de la clavija — ¡podría **romper** la cuerda (romperla por completo, igual que las actualizaciones demasiado grandes pueden romper el entrenamiento de forma irreversible)!

**Resultados esperados:**
- lr=1e-4: Eventualmente bueno pero muy lento.
- lr=3e-4: Mejor rendimiento general.
- lr=1e-3: Progreso inicial rápido, luego inestabilidad.

---

### Experimento 3: Épocas de actualización (Update Epochs, K)

```
K = 3   (conservador — pocas pasadas por cada lote)
K = 10  (estándar)
K = 20  (agresivo — muchas pasadas por cada lote)
```

**¿Qué controlan las épocas de actualización?**

Después de recopilar un **rollout** (= jugar al juego durante un periodo de tiempo para reunir nueva experiencia — como un estudiante que hace una sesión de deberes antes de revisarlos), PPO empaqueta esa experiencia en un **lote** (batch = el conjunto completo de tuplas de estado, acción, recompensa de ese rollout). Luego realiza K **pasadas** (sweeps = recorridos completos por el lote, cada pasada actualizando la red una vez) sobre los mismos datos.
Más épocas = exprimir más aprendizaje de cada lote, pero riesgo de **sobreajuste a datos obsoletos** (= memorizar patrones que eran ciertos bajo la política antigua pero que ya no son válidos una vez que la política se ha actualizado, como un estudiante que memoriza el examen del año pasado y suspende el nuevo).

**Ejemplo de la vida real:** Un estudiante practicando con un conjunto de 20 problemas de matemáticas.
- K=3:  Hacer cada problema 3 veces → sigue aprendiendo, no se sobreajusta al conjunto de práctica.
- K=10: Hacer cada problema 10 veces → dominio sólido de estos problemas específicos.
- K=20: Hacer cada problema 20 veces → **memorizar soluciones sin entender realmente las matemáticas** (= el modelo se ajusta perfectamente al lote específico pero pierde la capacidad de generalizar).

> ⚠️ **"Pero los resultados para K=20 parecen estar bien —¿por qué debería importarme?"**
> El truco de recorte de PPO limita cuánto puede cambiar la política por pasada, por lo que K=20 no causará un colapso repentino.
> Sin embargo, el agente sigue sobreadaptándose silenciosamente a datos que ya no reflejan lo que la política actual experimentaría realmente.
> Esto **ralentiza el aprendizaje a largo plazo**: cada rollout enseña al agente menos de lo que debería, porque las pasadas posteriores reciclan información cada vez más obsoleta.
> El daño es gradual, no dramático — que es exactamente por lo que es fácil pasarlo por alto en experimentos cortos.

El recorte evita el sobreajuste catastrófico, pero demasiadas épocas pueden ralentizar el aprendizaje general.

**Resultados esperados:**
- K=3:  Menos eficiente (parte del potencial de aprendizaje desperdiciado por lote).
- K=10: Buen equilibrio.
- K=20: Riesgo de que la política se vuelva **demasiado confiada en datos obsoletos** (= las actualizaciones de la red se ven impulsadas por experiencias que ya no coinciden con lo que la política actual encontraría, erosionando silenciosamente la eficiencia de las muestras).

---

## Cómo leer los resultados

El gráfico muestra tres diagramas, cada uno variando un hiperparámetro:

```
Gráfico izquierdo:   Clip Epsilon —¿qué ε aprende más rápido?
Gráfico central:     Tasa de aprendizaje —¿qué lr es más estable?
Gráfico derecho:     Épocas de actualización —¿qué K encuentra la mejor política?
```

Cada línea es la **recompensa media sobre 3 semillas** (para reducir la aleatoriedad).

**Qué buscar:**
1. **Velocidad de aprendizaje**: ¿Qué línea alcanza antes una recompensa alta?
2. **Rendimiento final**: ¿Qué línea logra la recompensa final más alta?
3. **Estabilidad**: ¿Qué línea tiene menos oscilaciones?

¡Un buen hiperparámetro equilibra los tres!

---

## Metodología: Experimentación Científica

Este experimento utiliza un diseño de **estudio de ablación** (= un método en el que se elimina o varía un componente a la vez para medir su impacto individual — llamado así por la práctica científica de eliminar selectivamente tejido para estudiar su función):
1. Elegir valores por defecto: ε=0.2, lr=3e-4, K=10.
2. Cambiar UN parámetro a la vez.
3. Mantener todo lo demás fijo.
4. Comparar los resultados.

Esto nos indica el efecto de CADA parámetro por separado.

**Ejemplo de la vida real:** Probar si un nuevo fertilizante ayuda a las plantas:
- Cambiar el fertilizante, mantener todo lo demás igual (mismo suelo, agua, luz solar).
- Si las plantas crecen mejor → ¡el fertilizante ayudó!

---

## Hallazgos comunes en la práctica

| Hiperparámetro | Demasiado pequeño | Punto óptimo | Demasiado grande |
|----------------|-----------|------------|-----------|
| **ε (clip)** | Convergencia lenta | ε ≈ 0.2 | Inestabilidad |
| **lr** | Demasiado lento | 2.5e-4 a 3e-4 | Divergencia |
| **K (épocas)** | **Desperdicio de datos** (descartar el rollout antes de extraer toda la señal) | K = 4-10 | Sobreajuste a datos de rollout obsoletos |
| **n_steps** | Demasiado ruidoso | 128-2048 | **Errores de memoria OOM** (usa demasiada RAM) |
| **batch_size** | Demasiado ruidoso | 32-256 | **Errores de memoria OOM** (usa demasiada RAM) |

¡Estos "puntos óptimos" pueden variar según el entorno!

---

## La idea clave: PPO es relativamente robusto

En comparación con algoritmos anteriores (como DQN sin redes objetivo), PPO es relativamente robusto a la elección de hiperparámetros. El mecanismo de recorte proporciona una red de seguridad natural.

**Ejemplo de la vida real:** Un coche con frenos **ABS** (Anti-lock Braking System — un sistema de seguridad que evita que las ruedas se bloqueen durante una frenada brusca, manteniendo el control del conductor) frente a uno sin ellos:
- Sin ABS (DQN): Un giro equivocado (mal hiperparámetro) y trompeas.
- Con ABS (PPO): El coche se corrige a sí mismo — los hiperparámetros razonables funcionan todos más o menos bien.

¡Esta robustez es una de las razones principales por las que PPO es el algoritmo de RL más popular en la práctica!

---

## Conclusiones clave

| Concepto | En lenguaje sencillo |
|---------|---------------|
| **Estudio de ablación** | Cambiar una cosa a la vez para ver su efecto. |
| **Clip epsilon ε** | Límite de seguridad — 0.2 suele ser lo mejor. |
| **Tasa de aprendizaje** | **Tamaño del paso** — cuánto se ajustan los pesos de la red después de cada lote (piensa en ello como el tamaño de cada paso al caminar hacia una meta). **2.5e-4 a 3e-4** es notación científica para 0.00025 a 0.0003 — estos son multiplicadores adimensionales, no valores de tiempo. |
| **Épocas de actualización K** | Cuántas veces reutilizar cada lote — 4-10 es lo estándar. |
| **Semillas Aleatorias** | Cada experimento se repite con diferentes **semillas aleatorias** (= el número inicial que se introduce en el generador de números aleatorios, que controla todas las elecciones aleatorias en el entrenamiento). El uso de múltiples semillas revela si los resultados son consistentes o si simplemente se tuvo suerte. |

---

## Resumen: Métodos de Gradiente de Política de un vistazo

```
REINFORCE              A2C                    PPO
     │                  │                      │
Episodios completos  Act. de n-pasos        N-pasos + recorte
Simple pero ruidoso  Más rápido pero inest. Estable + eficiente
Mejor para entornos  Entornos de            Entornos difíciles
fáciles              dificultad media       (estándar industrial)
```

**Si solo aprendes UN algoritmo de esta fase, que sea PPO.** Es la base de:
- El entrenamiento de ChatGPT de OpenAI (RLHF usa PPO).
- Las continuaciones de AlphaGo de DeepMind.
- La mayoría de la investigación moderna en robótica.
- Las IAs que juegan a videojuegos.
