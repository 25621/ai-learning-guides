# Arquitectura Option-Critic

## La Gran Idea: Trabajar por capítulos, no palabra por palabra

Imagina que estás escribiendo una novela. No planeas cada palabra antes de empezar. En su lugar, piensas en **capítulos**: "El capítulo 1 presenta al héroe. El capítulo 2 es la búsqueda. El capítulo 3 es el enfrentamiento final". Dentro de cada capítulo, vas resolviendo los detalles sobre la marcha.

Así es exactamente como la arquitectura Option-Critic piensa sobre las decisiones.

---

## ¿Qué es un agente "plano"?

Un agente de RL normal (como los de las Fases 3 y 4 del plan de estudios) decide una acción cada vez, en cada paso. Es como un GPS que recalcula toda la ruta desde cero cada vez que te mueves un metro. Funciona, pero es agotador y lento de aprender.

---

## ¿Qué es una "opción" (option)?

Una **opción** es una **habilidad con nombre** — una minipolítica que el agente puede ejecutar durante varios pasos seguidos antes de devolver el control.

Piénsalo como un gestor que delega en especialistas:

| Quién | Qué hace |
|-----|-------------|
| **Gestor (metapolítica)** | Decide *qué* especialista enviar a un trabajo |
| **Especialista A** | Experto en navegar por la habitación superior izquierda |
| **Especialista B** | Experto en cruzar umbrales de puertas |
| **Especialista C** | Experto en arremeter hacia la meta |
| **Especialista D** | Generalista de apoyo |

El gestor elige a un especialista. El especialista trabaja hasta que decide que ha terminado (esto se llama **terminación**). Entonces el gestor vuelve a elegir.

---

## Las tres partes móviles

Cada opción tiene tres componentes — piénsalos como la **descripción del trabajo** del especialista:

1. **Iniciación**: ¿Cuándo se puede llamar a este especialista? *(ej. "El especialista A solo se activa cerca de la habitación superior izquierda").*
2. **Política intra-opción**: ¿Qué hace el especialista mientras trabaja? *(ej. "Caminar hacia la esquina superior izquierda").*
3. **Terminación**: ¿Cuándo devuelve el especialista el control? *(ej. "Para cuando hayas llegado a una puerta").*

Lo hermoso de Option-Critic es que las tres se **aprenden automáticamente** — no diseñas a los especialistas a mano. El algoritmo descubre por sí solo que es útil tener una opción para cada habitación, o una para correr hacia la meta.

---

## Un día en la vida de un agente Option-Critic

1. El agente entra en una nueva habitación (estado).
2. El **Gestor** mira la habitación y elige una opción — digamos, la Opción 2.
3. El **especialista de la Opción 2** toma el mando: camina hacia la puerta, paso a paso.
4. En algún momento, la Opción 2 dice "he terminado aquí" (terminación).
5. El **Gestor** se despierta, elige una nueva opción para la nueva situación.
6. Repetir.

Compara esto con el agente plano: el agente plano se angustia por cada paso individual. Option-Critic delega tramos enteros de comportamiento, permitiendo que cada especialista se vuelva bueno en su tarea específica.

---

## ¿Por qué ayuda esto?

En un laberinto, el agente necesita alcanzar una meta que puede estar a 30-50 pasos de distancia. Con el aprendizaje plano, cada paso del camino es igualmente "invisible" hasta que la recompensa finalmente llega al final — esa señal tiene que viajar hacia atrás a través de docenas de pasos.

Con las opciones, el camino se divide en **subtareas**. Cada subtarea recibe su propia miniseñal de recompensa (llegar a la puerta, entrar en la siguiente habitación). El aprendizaje se propaga a través de segmentos más cortos. **El agente aprende más rápido en problemas que requieren muchos pasos.**

Esta es la idea central de todo el RL jerárquico — y Option-Critic es una de sus implementaciones más limpias.

---

## Qué hace nuestro código

El script `option_critic.py` pone a un agente Option-Critic en un **mundo de cuadrícula de 7x7** con una meta fija. El agente comienza en cualquier lugar de la cuadrícula y debe navegar hasta la celda de la meta.

El agente tiene cuatro opciones y debe aprender simultáneamente:

- Una política para cada opción (hacia dónde caminar)
- Cuándo terminar cada opción (condición de terminación)
- Una metapolítica para elegir entre opciones

La recompensa utiliza **moldeado basado en potencial** (potential-based shaping) — el agente recibe un pequeño bono cada paso que se acerca a la meta, además de +1 por alcanzarla. Esta retroalimentación densa hace que el aprendizaje sea lo suficientemente estable como para ver las opciones funcionando en 2,500 episodios.

Ningún humano le dice nunca qué debe hacer cada opción. El algoritmo descubre en qué áreas de la cuadrícula se especializa cada opción.

---

## Qué muestran los gráficos

![Curvas de aprendizaje de Option-Critic](outputs/option_critic.png)

**Izquierda — Retorno moldeado (Shaped Return)**: Un retorno más alto significa que el agente está llegando a la meta de forma más fiable Y tomando caminos más cortos (el moldeado da un bono por cada paso más cerca). La curva que sube y luego se estabiliza muestra a las opciones aprendiendo a coordinarse.

**Derecha — Pasos hasta la meta (Steps to Goal)**: Menos pasos significan que el agente encontró un camino más eficiente. La tendencia a la baja muestra que las opciones maduran en habilidades coherentes que guían al agente más directamente hacia la meta.

Las curvas suavizadas muestran la tendencia general a lo largo de ventanas de 100 episodios — algo de ruido es normal en RL, especialmente cuando varios componentes (opciones, terminación, metapolítica) están aprendiendo simultáneamente.

---

## Resumen de una frase

> **Option-Critic enseña a un agente a trabajar con habilidades en lugar de pasos individuales — un gestor elige qué especialista se ejecuta, cada especialista hace su trabajo y todo el sistema aprende unido a partir de la misma señal de recompensa.**
