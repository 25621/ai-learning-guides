# El Problema del Bandido Multibrazo (Multi-Armed Bandit) 🎰

## ¿Qué es?

Imagina que estás en una fiesta de cumpleaños y hay **10 tarros de caramelos diferentes**. Cada tarro tiene caramelos dentro, pero algunos tarros tienen caramelos *ricos* y otros tienen caramelos *no tan ricos*. No sabes qué tarro es el mejor, ¡tienes que probarlos!

Cada vez que metes la mano en un tarro, sacas un caramelo. Tu trabajo es:

> **¡Conseguir tantos caramelos ricos como sea posible!**

¡Ese es el problema del Bandido Multibrazo! En lugar de tarros de caramelos, los científicos los llaman "brazos" (como los brazos de una máquina tragaperras). Cada brazo te da un premio, pero los premios son diferentes cada vez.

---

## La Gran Pregunta: ¿Debo probar tarros nuevos o quedarme con mi favorito?

¡Esta es la parte más difícil! Digamos que has probado el Tarro #3 y estaba bastante bien. Ahora tienes que elegir:

- **Explotar (Exploit)**: Seguir eligiendo el Tarro #3 porque ya sabes que es bueno.
- **Explorar (Explore)**: Probar un tarro nuevo — ¡quizás el Tarro #7 sea incluso *mejor*!

Si solo eliges el primer tarro que te gustó, podrías perderte el tarro súper rico. Pero si *siempre* pruebas tarros nuevos, ¡nunca usarás lo que ya has aprendido!

**Ejemplo de la vida real:** Piensa en tu restaurante favorito. Siempre pides nuggets de pollo (¡explotar!), pero quizás la pizza sea incluso mejor. Si nunca pruebas nada nuevo, ¡nunca lo sabrás!

---

## La Estrategia Epsilon-Greedy {#the-epsilon-greedy-strategy}

Una forma inteligente de resolver esto se llama **epsilon-greedy** (épsilon es solo la letra griega ε):

1. **La mayor parte del tiempo (digamos el 90%)**: Elige el tarro que *crees* que es el mejor.
2. **A veces (digamos el 10%)**: ¡Elige un tarro *al azar* para explorar!

Los viajes de exploración del 10% te ayudan a descubrir tarros mejores. Los viajes de explotación del 90% te permiten usar lo que ya has aprendido.

---

## Qué encontró nuestro código

Probamos 10 brazos (tarros de caramelos) con 200 niños diferentes, 1000 elecciones cada uno:

| Estrategia | % de veces que se elige el mejor tarro |
|----------|----------------------------------|
| **Nunca explorar (ε=0)** | 14.5% — se quedó atascado pronto, ¡nunca encontró el mejor! |
| **Explorar el 1% de las veces (ε=0.01)** | 37.6% — encontró el mejor tarro lentamente |
| **Explorar el 10% de las veces (ε=0.10)** | **74.2%** — ¡aprendió rápido, eligió el mejor la mayoría de las veces! |

**Lección**: ¡Un poco de exploración ayuda mucho!

---

## Ejemplos de la vida real

- **Recomendaciones de Netflix**: ¿Debería Netflix mostrarte una película que probablemente te guste (explotar) o sugerirte algo nuevo (explorar)?
- **Médico eligiendo un tratamiento**: ¿Usar el tratamiento que suele funcionar (explotar) o probar uno nuevo que podría ser incluso mejor (explorar)?
- **Una abeja buscando flores**: ¿Debería seguir visitando las flores que sabe que tienen néctar o volar a un campo nuevo?

---

## Palabras Clave para Recordar

- **Brazo (Arm)**: Una de las opciones (como un tarro de caramelos).
- **Recompensa (Reward)**: Lo que obtienes cuando eliges un brazo (como un caramelo).
- **Explotar (Exploit)**: Usar lo que ya sabes que es bueno.
- **Explorar (Explore)**: Probar algo nuevo para aprender más.
- **Epsilon (ε)**: La probabilidad de que explores en lugar de explotar.

La gran idea: **¡Tienes que equilibrar el probar cosas nuevas con el uso de lo que ya sabes!**
