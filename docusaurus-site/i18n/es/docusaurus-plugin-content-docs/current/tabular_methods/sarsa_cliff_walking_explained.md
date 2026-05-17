# SARSA para Cliff Walking 🏔️

## ¿Qué es esto?

Imagina un **pasillo muy largo** con un **terrible precipicio** a lo largo de un borde. ¡Si te caes por el precipicio, tienes que volver al principio! Tu objetivo es caminar de un extremo a otro lo más rápido posible, sin caerte.

**SARSA** es un robot que aprende a recorrer este pasillo practicando. Aprende a tomar un **camino seguro** que evita el precipicio — aunque sea un poco más largo — ¡porque sabe que podría resbalar accidentalmente cerca del borde al explorar!

---

## La Gran Idea: Aprender de lo que realmente haces

SARSA son las siglas de: **S**tate (Estado) → **A**ction (Acción) → **R**eward (Recompensa) → **S**tate (Estado) → **A**ction (Acción).

Estas son las cinco piezas de información que SARSA utiliza para aprender:

1. **S** — ¿Dónde estoy ahora mismo? (estado actual).
2. **A** — ¿Qué acción he tomado realmente?
3. **R** — ¿Qué recompensa he obtenido?
4. **S** — ¿Dónde he acabado?
5. **A** — ¿Qué acción voy a tomar *realmente* a continuación?

¡La última "A" es lo que hace especial a SARSA! Actualiza sus conocimientos utilizando la acción que *realmente tomará a continuación* (incluso si se trata de un movimiento exploratorio aleatorio), no la acción ideal perfecta.

**Ejemplo de la vida real:** Piensa en aprender a montar en bicicleta. Si sabes que a veces te tambaleas al azar (exploración), te mantienes un poco más alejado de los automóviles estacionados — ¡porque sabes que tu yo tambaleante podría desviarse! SARSA hace esto: aprende un camino seguro porque tiene en cuenta sus propios errores aleatorios.

---

## El Mapa de Cliff Walking

```
[ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]
[ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]
[ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]
[S][P][P][P][P][P][P][P][P][P][P][G]
   ← ← ← ← PRECIPICIO ← ← ← ← ←
```

- **S** = Inicio (Start - abajo a la izquierda).
- **G** = Meta (Goal - abajo a la derecha).
- **P** = Precipicio (Cliff) — pisar aquí = recompensa de -100, ¡reiniciar!.
- Cualquier otro paso = recompensa de -1.

---

## Qué encontró nuestro código

Después de entrenar a SARSA durante 500 episodios:

| Resultado | Valor |
|--------|-------|
| Recompensa media final de 50 episodios | **-21.6** |
| Recompensa del camino óptimo (arriesgado) | -13 |

¡La política aprendida por SARSA va **por la parte superior de la cuadrícula** — un rodeo seguro! Cuesta unos pocos pasos extra (-21 en lugar de -13), pero casi nunca se cae por el precipicio durante el entrenamiento.

---

## Ejemplos de la vida real

- **Enfermera administrando medicación**: Sigue el protocolo seguro probado (camino seguro) aunque exista un método ligeramente más rápido, porque los pequeños errores (exploración) podrían ser peligrosos.
- **Pilotos de aerolíneas**: Siguen listas de comprobación estrictas (caminos seguros) incluso cuando los atajos parecen más rápidos, teniendo en cuenta el error humano.
- **Aprender a cocinar**: Empezar con recetas bien probadas (seguras), no con atajos arriesgados.

---

## Palabras Clave para Recordar

- **Dentro de la política (On-policy)**: Aprende sobre la política que está usando realmente (incluyendo sus errores aleatorios).
- **Actualización SARSA**: Utiliza la acción siguiente *real*, no la teóricamente mejor.
- **Camino seguro**: Un camino más largo que evita el peligro, teniendo en cuenta los errores de exploración.
- **Control TD (Diferencia Temporal)**: Actualizar los valores después de cada paso (sin esperar al episodio completo).

La gran idea: **¡SARSA es honesto — aprende de lo que realmente hace, no de lo que desearía hacer. Esto lo hace cauteloso y seguro cerca del peligro!**
