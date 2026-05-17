# Iteración de Políticas para GridWorld 🗺️

## ¿Qué es esto?

Imagina que estás jugando a un juego de mesa en una **cuadrícula de 4×4** (como un tablero de ajedrez diminuto). Empiezas en una esquina y tienes que llegar a la esquina opuesta. Cada paso cuesta 1 punto (¡no quieres desperdiciar pasos!) y llegar a la meta no te da nada extra — solo quieres llegar allí lo más rápido posible.

La **Iteración de Políticas** (Policy Iteration) es como un equipo informático descubre los **mejores movimientos para cada cuadrado** del tablero — ¡todos a la vez!

---

## La Gran Idea: Dos pasos, una y otra vez

Piénsalo como si estuvieras limpiando tu habitación con un ayudante:

1. **Paso 1 — Averiguar qué tan bueno es cada cuadrado (Evaluación de la política)**
   Tu ayudante camina por cada cuadrado y anota: "Si sigo el plan actual, ¿cuántos pasos me llevará llegar a la salida desde aquí?". Hacen esto una y otra vez hasta que los números dejan de cambiar.

2. **Paso 2 — Mejorar el plan (Mejora de la política)**
   Ahora miras cada cuadrado y preguntas: "¿Hay una dirección mejor en la que podría ir desde aquí?". Si es así, ¡actualiza el plan!

Repite los pasos 1 y 2 hasta que el plan deje de cambiar — ¡esa es la **política óptima**!

**Ejemplo de la vida real:** Imagina encontrar la ruta más rápida a la escuela. Primero adivinas una ruta y cronometras el tiempo (Paso 1). Luego miras cada esquina de la calle y preguntas "¿hay un atajo desde aquí?" (Paso 2). ¡Actualizas tu ruta y repites hasta que no encuentres más atajos!

---

## Qué encontró nuestro código

Nuestro GridWorld de 4×4 tiene dos estados terminales (esquinas), y el agente paga -1 por paso. La iteración de la política convergió en solo **4 rondas** (139 barridos de evaluación totales):

```
Valores de Estado V(s):   Política Óptima:
 0.0  -1.0  -1.9  -2.7    T   ←   ←   ↓
-1.0  -1.9  -2.7  -1.9    ↑   ↑   ↑   ↓
-1.9  -2.7  -1.9  -1.0    ↑   ↑   ↓   ↓
-2.7  -1.9  -1.0   0.0    ↑   →   →   T
```

**¡Los valores tienen todo el sentido!** Los cuadrados junto a un terminal tienen el valor -1 (a un paso de distancia). Los cuadrados a dos pasos tienen el valor -1.9 (= -1 + 0.9 × -1), y así sucesivamente.

---

## Ejemplos de la vida real

- **Navegación GPS**: Calcular el mejor giro en *cada* intersección del mapa.
- **Control de ascensores**: ¿A qué piso debe ir el ascensor cuando tiene múltiples solicitudes?
- **Robot de fábrica**: Planificar el camino más eficiente por una cuadrícula de almacén.

---

## Palabras Clave para Recordar

- **Política (Policy)**: El plan — qué acción tomar en cada estado.
- **Función de Valor V(s)**: Qué tan bueno es estar en el estado s (más alto = más cerca de la meta).
- **Evaluación de la Política**: Calcular qué tan bueno es el plan actual.
- **Mejora de la Política**: Hacer el plan mejor usando la función de valor.
- **Política Óptima**: El mejor plan posible — no se puede mejorar más.

La gran idea: **¡No necesitas probar todos los planes posibles! Simplemente sigue mejorando el actual y encontrarás el mejor plan en muy pocas rondas.**
