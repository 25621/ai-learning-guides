# Deep Q-Network (DQN) desde cero 🧠

## El problema con lo lineal

¿Recuerdas nuestra fórmula lineal de antes?

> Puntuación = w₁ × posición_carro + w₂ × velocidad_carro + w₃ × ángulo_poste + w₄ × velocidad_punta_poste

Esto funciona bien para CartPole, pero ¿qué pasa con un videojuego donde ves miles de píxeles? ¡No puedes escribir una receta simple para eso!

Necesitamos algo que pueda mirar situaciones complicadas y deducir la mejor acción. Ese algo es una **red neuronal**.

---

## ¿Qué es una red neuronal?

Piensa en tu cerebro. Millones de pequeñas células llamadas neuronas hablan entre sí. Cuando tocas algo caliente, las neuronas envían señales: "¡CALIENTE! → ¡Retira la mano AHORA!". Cada neurona pasa información y, juntas, toman una decisión inteligente.

Una **red neuronal en un equipo informático** funciona de la misma manera:

```
Capa de Entrada    Capa Oculta 1     Capa Oculta 2     Capa de Salida
[pos carro]    →   [128 neuronas]  →  [128 neuronas]  →  [puntuación IZQUIERDA]
[vel carro]    →   [  ...       ]     [  ...       ]     [puntuación DERECHA]
[ángulo poste] →
[vel poste]    →
```

Cada flecha tiene un **peso** (qué tan fuerte es esa conexión). Hay miles de estos pesos — ¡y la red los aprende TODOS!

**Ejemplo de la vida real:** Un chef en un restaurante prueba tu comida y ajusta cientos de ingredientes a la vez. Cada papila gustativa es como una neurona, y juntas le dicen al chef "añade más sal" o "menos pimienta". Entrenar la red es como si el chef aprendiera a lo largo de miles de comidas.

---

## DQN = Deep Q-Network

**DQN** (Deep Q-Network) fue inventado por DeepMind en 2013. ¡Tomaron la antigua fórmula de Q-learning y cambiaron la tabla Q por una red neuronal!

En lugar de:
> Tabla-Q[estado][accion] = puntuación

Tenemos:
> Red-Q(estado) → [puntuación_izquierda, puntuación_derecha]

La red toma el estado como entrada y devuelve los valores Q para TODAS las acciones a la vez. ¡Esto es mucho más eficiente que calcularlos por separado!

---

## Este script: La versión "Ingenua" (Naive)

Este script muestra DQN **sin** ningún truco especial. Simplemente:
1. Ve el estado.
2. Pregunta a la red "¿qué tan buena es la izquierda? ¿qué tan buena es la derecha?".
3. Realiza la acción con la puntuación más alta.
4. Recibe una recompensa, actualiza la red.

**¡Esto es intencionadamente inestable!** Piensa en ello como un estudiante que olvida inmediatamente sus lecciones anteriores cada vez que aprende algo nuevo. La red se actualiza después de cada paso, lo que causa el caos.

**Ejemplo de la vida real:** Imagina aprender a cocinar cambiando toda tu receta después de cada bocado. Podrías pasar de "demasiado salado" a "nada de sal" y luego a "demasiado salado" de nuevo, sin llegar nunca a la cantidad adecuada. ¡Eso es lo que pasa aquí!

---

## Qué verás

Cuando ejecutes `dqn_cartpole.py`:
- Las puntuaciones pueden saltar mucho (aprendizaje inestable).
- A veces el agente se vuelve muy bueno y luego lo olvida todo.
- El gráfico de pérdida (loss) muestra oscilaciones salvajes.

**¡Esto es lo esperado!** Demuestra POR QUÉ necesitamos mejoras: el búfer de repetición (experience replay) y las redes objetivo (target networks). ¡Esos vienen a continuación!

---

## El truco ε-Greedy 🎲

El robot no siempre elige la mejor acción. ¡A veces elige al azar!

¿Por qué? Porque si siempre elige lo que parece mejor, puede que nunca descubra mejores opciones.

> Con probabilidad ε (epsilon): elige una acción ALEATORIA (¡explora!)
> Con probabilidad 1-ε: elige la MEJOR acción conocida (¡explota!)

Empezamos con ε = 1.0 (100% al azar) y disminuimos lentamente hasta ε = 0.01 (1% al azar). De esta manera, el robot explora mucho al principio y luego se centra en lo que ha aprendido.

**Ejemplo de la vida real:** Al visitar una ciudad nueva, puedes probar restaurantes al azar al principio (explorar). Después de un tiempo, vuelves a tus favoritos (explotar). ¡Pero de vez en cuando sigues probando algo nuevo por si acaso hay una joya escondida!

---

## Vocabulario Clave

| Palabra | Significado |
|------|---------|
| **Red Neuronal** | Capas de neuronas matemáticas conectadas que aprenden de los datos |
| **Deep (Profundo)** | Más de una capa oculta (de ahí el "aprendizaje profundo") |
| **DQN** | Deep Q-Network — utiliza una red neuronal en lugar de una tabla Q |
| **ε-Greedy** | Estrategia: explorar al azar a veces, explotar el mejor conocimiento otras veces |
| **Inestabilidad** | La red sigue "olvidando" porque las actualizaciones interfieren entre sí |

---

## Qué falta (y por qué importa)

Este DQN ingenuo tiene dos grandes problemas:

1. **Actualizaciones correlacionadas**: Cada experiencia viene en orden (paso 1, paso 2, paso 3...). Si el paso 5 fue malo, TODAS las actualizaciones cercanas se confunden.
   
2. **Objetivo móvil**: Después de cada actualización, la red cambia. Pero la siguiente actualización utiliza la MISMA red para calcular cuál debería ser el objetivo. ¡Es como disparar a una diana móvil!

Estos problemas se resuelven mediante el **Búfer de Repetición (Experience Replay)** y las **Redes Objetivo (Target Networks)** en los siguientes scripts. ¡Juntos, convierten a DQN de un principiante tambaleante en un campeón de los videojuegos!
