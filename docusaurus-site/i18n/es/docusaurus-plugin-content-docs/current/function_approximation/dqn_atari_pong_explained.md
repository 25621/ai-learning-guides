# DQN en Atari Pong 🏓

## ¿Qué es Atari Pong?

Pong es uno de los videojuegos más antiguos jamás creados: ¡es como el tenis de mesa digital! Dos paletas hacen rebotar una pelota de un lado a otro. Ganas un punto si el oponente no alcanza la pelota. El juego termina cuando alguien llega a 21 puntos.

En nuestra versión, la IA controla una paleta. El oponente (computadora) controla la otra. El juego siempre comienza con una puntuación de −21 (el peor caso posible). Un buen agente llega a 0 o +21.

---

## ¿Por qué Pong es difícil para una IA?

En CartPole, el robot podía VER los números directamente (ángulo del poste, velocidad del carro...). En Pong, ¡todo lo que ve son **píxeles brutos**, miles de pequeños puntos de colores en una pantalla!

```
Entrada de CartPole: [0.02, −0.14, 0.01, −0.23]   ← 4 números, ¡fácil!
Entrada de Pong:     [cuadrícula de píxeles: 210×160×3] ← 100,800 números, ¡MUCHO más difícil!
```

El robot tiene que deducir a partir de los píxeles:
- ¿Dónde está mi paleta?
- ¿Dónde está la pelota?
- ¿Se mueve la pelota hacia la izquierda o hacia la derecha?
- ¿A qué velocidad?

Los humanos hacemos esto automáticamente (¡tenemos una visión increíble!). Para una IA, este es un desafío enorme.

---

## Ver el movimiento: Apilamiento de frames (Frame Stacking) 🎬

Un solo frame (captura de pantalla) no te dice si la pelota se mueve hacia la izquierda o hacia la derecha. Necesitas ver MÚLTIPLES frames para entender el movimiento, igual que un libro animado solo funciona cuando pasas muchas páginas.

**Apilamiento de frames:** Introducir los últimos 4 frames en la red simultáneamente.

```
Frame 1: pelota en la posición 40
Frame 2: pelota en la posición 43    → Apilar estos 4 frames → ¡La red ve el MOVIMIENTO!
Frame 3: pelota en la posición 46
Frame 4: pelota en la posición 49
```

La red ahora puede inferir: "la pelota se mueve hacia la derecha a velocidad 3".

**Ejemplo de la vida real:** Ver una película frente a mirar un solo frame. Un solo frame de una carrera de coches es solo una imagen borrosa. ¡Mira 4 frames y podrás decir qué coche es más rápido!

---

## Ver con una CNN 🔍

Para las entradas de píxeles, utilizamos una red neuronal especial llamada **Red Neuronal Convolucional (CNN)**. En lugar de mirar todos los píxeles a la vez, una CNN utiliza ventanas deslizantes para detectar patrones, como ojos escaneando una imagen.

```
Píxeles brutos (84×84×4 frames)
       ↓
Capa Conv 1 (filtro 8×8, stride 4) → encuentra bordes y formas
       ↓
Capa Conv 2 (filtro 4×4, stride 2) → encuentra objetos (paletas, pelota)
       ↓
Capa Conv 3 (filtro 3×3, stride 1) → encuentra relaciones
       ↓
Aplanar → 512 neuronas → Valores Q (uno por acción)
```

**Ejemplo de la vida real:** Cuando buscas a tu amigo en una multitud, tu cerebro nota primero las formas (una persona), luego las características (color de pelo) y después los detalles (su cara). ¡Las CNN funcionan de la misma manera, desde patrones simples a otros complejos!

---

## Preprocesamiento: Encoger el mundo

Los frames de Pong son de 210×160 píxeles en color. ¡Eso es demasiado grande! Preprocesamos cada frame:

1. **Escala de grises**: el color no importa en Pong (la pelota siempre es blanca de todos modos).
2. **Redimensionar a 84×84**: más pequeño = entrenamiento más rápido, pero todavía lo suficientemente claro como para verlo.
3. **Normalizar a [0,1]**: dividir los valores de los píxeles por 255 para que sean números pequeños.

**Ejemplo de la vida real:** Como hacer una fotocopia al 50% de tamaño. Los detalles importantes (pelota, paletas) siguen siendo visibles, solo que más pequeños. ¡A la fotocopiadora tampoco le importan los colores!

---

## Recorte de recompensas (Reward Clipping): Tratar todos los juegos por igual ✂️

En Pong, obtienes +1 por anotar, −1 si te anotan a ti. ¡En algunos otros juegos de Atari, las puntuaciones pueden ser de miles!

**Recortamos las recompensas** a [−1, +1] para que a la red no le importe la escala de las recompensas. Este mismo código puede entrenar en CUALQUIER juego de Atari sin ajustar las escalas de recompensa.

---

## ¿Cuánto tiempo lleva el entrenamiento?

| Duración del entrenamiento | Qué aprende el agente |
|---|---|
| 100K pasos | Principalmente aleatorio, apenas reacciona |
| 1M pasos | Empieza a moverse hacia la pelota a veces |
| 5M pasos | Devuelve algunos tiros |
| 10M pasos | Juego competitivo, puede ganar algunos |
| 20M+ pasos | A menudo vence al oponente de la computadora |

Nuestra demostración ejecuta **300K pasos**, lo suficiente para ver que la arquitectura de entrenamiento funciona y observar el aprendizaje temprano, pero no lo suficiente para dominar el juego.

**Ejemplo de la vida real:** Aprender a tocar el piano lleva meses. Una sesión de práctica de 10 minutos te muestra que lo estás haciendo bien, ¡pero no esperes dar conciertos todavía!

---

## Qué encontró nuestro código

Después de 300K pasos en Pong:
- El agente comienza con puntuaciones alrededor de −20 (apenas devuelve la pelota).
- Al final, normalmente mejora a unos −15 o −10.
- La curva de aprendizaje muestra una mejora gradual desde el juego aleatorio.

Para ver un rendimiento de Pong realmente competitivo, necesitarías ejecutar unos 10M+ de pasos con una GPU. La implementación es completa y correcta, ¡solo necesita más tiempo!

---

## Vocabulario Clave

| Palabra | Significado |
|------|---------|
| **CNN** | Red Neuronal Convolucional — especializada para entradas de imágenes |
| **Apilamiento de frames** | Alimentar múltiples frames consecutivos para capturar el movimiento |
| **Preprocesamiento** | Transformar los frames brutos (escala de grises, redimensionar, normalizar) antes de introducirlos en la red |
| **Recorte de recompensas** | Limitar las recompensas a [−1, +1] para que funcione en diferentes juegos |
| **ALE** | Arcade Learning Environment — la biblioteca que ejecuta los juegos de Atari |

---

## El logro histórico

Cuando DeepMind publicó DQN en 2015, el mundo quedó asombrado. ¡Un ÚNICO algoritmo, con la MISMA arquitectura, aprendió a jugar 49 juegos de Atari diferentes — muchos a nivel sobrehumano — solo a partir de píxeles brutos y una puntuación!

Antes de DQN, la gente pensaba que era necesario codificar a mano la estrategia de cada juego. DQN demostró que un aprendiz de propósito general podía resolverlo todo por sí mismo. ¡Fue un momento histórico en la IA!
