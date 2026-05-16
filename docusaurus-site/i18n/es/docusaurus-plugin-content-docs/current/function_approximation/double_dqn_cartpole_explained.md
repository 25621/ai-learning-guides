# Double DQN: Solucionando el problema del exceso de confianza 🤔

## El Problema: DQN cree que es mejor de lo que es

Imagina que te preguntan: "¿Cuál es el mejor restaurante de la ciudad?"

Podrías decir: "¡Pizza Palace es increíble, definitivamente es un 10/10!". Pero solo has estado allí dos veces. En realidad, no sabes si *realmente* es un 10/10. Podrías estar sobreestimando porque tuviste suerte con una buena pizza en esas dos visitas.

Este mismo problema ocurre con DQN: el agente **sobreestima los valores Q**.

---

## ¿Por qué sobreestima DQN?

Cuando DQN calcula el objetivo (target), hace:
> objetivo = recompensa + γ × **max** Q(siguiente_estado)

¡El `max` es el problema! Cuando eliges el máximo de varias estimaciones ruidosas, casi siempre eliges la que tiene el mayor error aleatorio (sesgo al alza).

**Ejemplo de la vida real:** Pides a 5 amigos que adivinen la altura de un edificio. Sus suposiciones son: 40m, 38m, 45m (¡suerte!), 39m, 41m. La altura real es 40m. Si usas el `max(suposiciones)` = 45m, ¡estás muy equivocado! El máximo de suposiciones ruidosas es casi siempre una sobreestimación.

A lo largo de miles de actualizaciones, DQN sigue entrenándose hacia estos objetivos sobreinflados, aprendiendo que las cosas son mejores de lo que realmente son. Esto puede ralentizar el aprendizaje o hacer que el agente tome decisiones pobres y demasiado confiadas.

---

## La solución Double DQN

**Double DQN** (Hasselt et al., 2016) divide el `max` en dos pasos:

**Paso 1 — ¿Qué acción?** Usa la **red online** para elegir la mejor acción:
> mejor_accion = argmax Q_online(siguiente_estado)

**Paso 2 — ¿Cuál es su valor?** Usa la **red objetivo** (target) para evaluar esa acción:
> objetivo = recompensa + γ × Q_target(siguiente_estado, mejor_accion)

```
DQN normal:    objetivo = r + γ × max_a Q_target(s', a)
                                 ↑ la misma red elige Y evalúa → sesgado

Double DQN:    mejor_a = argmax_a Q_online(s', a)   ← la online elige
               objetivo = r + γ × Q_target(s', mejor_a) ← la target evalúa
                                 ↑ diferentes redes → menos sesgado
```

**Ejemplo de la vida real:** En una entrevista de trabajo, no dejas que el solicitante de empleo califique su propio examen de desempeño (¡ese es el problema de DQN normal!). En su lugar, el candidato *presenta* su mejor trabajo y un examinador *diferente* lo evalúa. ¡Dos personas diferentes = evaluación más justa!

---

## ¿Por qué ayuda la separación?

Las dos redes (online y objetivo) tienen pesos diferentes porque la red objetivo se actualiza con menos frecuencia. Tienen "opiniones" diferentes sobre qué acción es la mejor.

Cuando no están de acuerdo:
- La online dice: "¡La acción A parece genial!"
- La objetivo dice: "En realidad, la acción A solo está bien — vale unos 7, no 10"

Al usar la estimación de VALOR de la red objetivo para la acción ELEGIDA por la red online, obtenemos un número más honesto y menos inflado.

---

## Diferencia en el código: ¡Solo una línea!

El único cambio en el código de DQN normal a double DQN está en el cálculo del objetivo:

```python
# DQN Normal:
q_next = target_net(s_next).max(dim=1).values

# Double DQN:
best_actions = online_net(s_next).argmax(dim=1, keepdim=True)   # elige con la online
q_next = target_net(s_next).gather(1, best_actions)              # evalúa con la target
```

Solo cambian dos líneas, ¡pero el impacto en la estabilidad y la precisión es significativo!

---

## Qué muestra la comparación

Cuando ejecutes `double_dqn_cartpole.py`, verás dos gráficos:

**Gráfico 1: Curvas de aprendizaje**
- Tanto DQN normal como double DQN deberían resolver CartPole.
- Double DQN a menudo converge más rápido y de forma más estable.
- CartPole es lo suficientemente simple como para que la diferencia sea modesta; es más dramática en Atari.

**Gráfico 2: Estimaciones de los valores Q**
- DQN normal: Los valores Q tienden al alza con el tiempo (sobreestimación).
- Double DQN: Los valores Q se mantienen más modestos y precisos.

El gráfico de sobreestimación de los valores Q es la clave — muestra cómo el DQN normal aprende valores inflados que acaban perjudicando el rendimiento.

---

## La familia de mejoras de DQN

Double DQN es solo una de las muchas mejoras al DQN original. El artículo "Rainbow" (2017) combinó 6 mejoras:

1. **Double DQN** (corrige la sobreestimación) ← ¡este script!
2. **Prioritized Replay** (aprende más de las experiencias sorprendentes).
3. **Dueling Networks** (separa "¿qué tan bueno es este estado?" de "¿cuál es la mejor acción?").
4. **Multi-step returns** (mira más hacia el futuro).
5. **Distributional RL** (aprende la distribución completa de los retornos, no solo el promedio).
6. **NoisyNets** (exploración aprendida en lugar de [ε-greedy](../foundations/multi_armed_bandit_explained.md#the-epsilon-greedy-strategy)).

¡Rainbow combinó TODAS ellas y logró el mejor rendimiento en Atari de su época!

---

## Vocabulario Clave

| Palabra | Significado |
|------|---------|
| **Sobreestimación** | Los valores Q son más altos que los valores reales (demasiado optimistas) |
| **Double DQN** | Usa la red online para la selección de acciones y la red objetivo para la evaluación |
| **Desacoplamiento** | Separar dos tareas que antes realizaba la misma red |
| **Sesgo (Bias)** | Un error sistemático en una dirección (siempre demasiado alto o siempre demasiado bajo) |
| **Rainbow** | Una variante de DQN que combina 6 mejoras para el máximo rendimiento |

---

## Resumen del viaje de la Fase 3

Ahora has completado la progresión completa de la Fase 3:

| Algoritmo | Qué añade | Por qué ayuda |
|-----------|-------------|-------------|
| Linear Q | Red neuronal → fórmula simple | Maneja estados continuos |
| Naive DQN | Red neuronal completa | Aprende patrones complejos |
| + Replay buffer | Muestreo de memoria aleatorio | Rompe las correlaciones |
| + Target network | Copia congelada para objetivos | Estabiliza el "centro de la diana" |
| Atari DQN | CNN + apilamiento de frames | ¡Aprende de los píxeles! |
| Double DQN | Separa elegir/evaluar | Reduce la sobreestimación |

Cada paso solucionó un problema específico. Así es como funciona la investigación real — ¡una mejora cuidadosa a la vez!
