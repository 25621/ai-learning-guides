# Explorando los entornos PettingZoo 🦓

## ¿Qué es PettingZoo?

Si has hecho RL de un solo agente, probablemente habrás usado **Gymnasium** (el sucesor de OpenAI Gym). Todos los entornos se ven iguales: `env.reset()`, `env.step(action) → obs, reward, done, info` — una nueva *observación* del mundo, una señal de *recompensa* escalar, un flag *done* que dice "fin del juego" y un diccionario *info* para extras de depuración. Esa uniformidad es lo que hace que las bibliotecas de RL funcionen.

**PettingZoo** es exactamente la misma idea pero para *múltiples agentes*. Es un "zoo" de entornos multiagente, todos tras una API bien definida:
- **Problemas de juguete clásicos**: entornos simples como Piedra-Papel-Tijera para probar algoritmos básicos.
- **Mundos de cuadrícula cooperativos**: agentes que navegan por una cuadrícula para lograr un objetivo compartido.
- **Atari multijugador**: juegos competitivos clásicos como Pong.
- **MPE (Multi-Particle Environment)**: entornos físicos de espacio continuo para coordinación y competición complejas.

Si puedes escribir código que funcione en un entorno PettingZoo, puedes conectarlo a cualquier otro con casi ningún cambio.

---

## Los dos estilos de API

Los entornos multiagente son más complicados que los de un solo agente porque dos agentes pueden actuar al mismo tiempo, o por turnos, o incluso en órdenes arbitrarios. PettingZoo soluciona esto con dos APIs paralelas:

### 1) AEC (Agent-Environment-Cycle)

Actúa un agente a la vez. El entorno recorre los agentes en algún orden, y cada uno obtiene:
- una **observación**: lo que ven *ahora mismo*,
- una **recompensa**: el pago obtenido por la acción *conjunta* en la última ronda completa (es decir, lo que sucedió como resultado de que *todos* los agentes actuaran, no solo tú; en una partida de ajedrez, por ejemplo, tu recompensa refleja el estado del tablero tras el último movimiento de tu oponente, no solo el tuyo),
- un **flag de terminación**: `True` cuando el episodio termina *naturalmente* (ej. jaque mate, alguien gana),
- un **flag de truncamiento**: `True` cuando el episodio se *corta* por un límite de tiempo antes de llegar a un final natural.

Esto es natural para **juegos por turnos** como el ajedrez, el Go o el póker.

```python
env.reset()
for agent in env.agent_iter():
    obs, reward, term, trunc, info = env.last()
    if term or trunc:
        env.step(None)
        continue
    action = mi_politica(obs, agent)
    env.step(action)
```

### 2) Parallel (Paralelo)

Todos los agentes observan y actúan simultáneamente en cada paso. `step()` recibe un *diccionario* de acciones y devuelve diccionarios de observaciones y recompensas.

Esto es natural para **juegos en tiempo real** como MPE (Multi-Particle Environments, donde todos los agentes-punto se mueven simultáneamente) o mundos de cuadrícula multiagente.

```python
obs, info = env.reset()
while env.agents:
    actions = {a: mi_politica(obs[a]) for a in env.agents}
    obs, rewards, terms, truncs, info = env.step(actions)
```

Los dos estilos son **isomórficos** — estructuralmente equivalentes e interconvertibles: cualquier entorno AEC puede envolverse automáticamente para que parezca uno Paralelo, y viceversa. PettingZoo incluye envoltorios (wrappers) de conversión para que solo tengas que escribir código para un estilo.

---

## Analogía de la vida real

- **AEC = una noche de juegos de mesa**. "Turno de Alice. Ahora Bob. Ahora Carol. De vuelta a Alice". Quien mueve a continuación ve el estado más reciente del tablero.
- **Parallel = un videojuego multijugador**. Los cuatro jugadores están pulsando botones simultáneamente; el juego actualiza el mundo 60 veces por segundo.
- **Por qué importan las APIs uniformes**. Imagina que cada videojuego multijugador necesitara su propio joystick. PettingZoo es el "joystick universal" del MARL.

---

## Qué hace nuestro código

Construimos un entorno **estilo PettingZoo** desde cero: el **Juego de Coordinación Iterado**. Dos agentes eligen repetidamente el canal `0` o `1`:

- Misma elección → ambos obtienen +1
- Diferente elección → ambos obtienen -1

La **observación** que recibe cada agente es la *acción conjunta* anterior — lo que ambos agentes eligieron la ronda pasada, empaquetado en un único entero. Concretamente: la última acción de cada agente es una de `{inicio, 0, 1}` (3 estados), por lo que el par se codifica como `3 × estado_agente_1 + estado_agente_2`, resultando en 9 enteros posibles (0 – 8). El entero 0 es el estado de "inicio" — señala que aún no se ha tomado ninguna acción (el comienzo mismo de un episodio). Un episodio dura 25 pasos, por lo que el retorno total máximo es +25 por agente y el mínimo es -25. **El juego aleatorio puntúa ≈ 0** porque en cada paso dos agentes aleatorios independientes eligen cada uno 0 o 1 con igual probabilidad: coinciden el 50 % de las veces (+1) y difieren el 50 % de las veces (-1), dando una recompensa esperada por paso de 0.5 × (+1) + 0.5 × (-1) = **0**. Sumado a lo largo de 25 pasos, el retorno esperado del episodio también es 0.

Luego nosotros:

1. **Demostramos la interfaz AEC** con una ejecución aleatoria — esto confirma el bucle AEC básico: `agent_iter()` devuelve el agente al que le toca el turno, `last()` lee la observación actual y la recompensa acumulada de ese agente, y `step()` devuelve su acción elegida al entorno.
2. **Entrenamos a dos aprendices Q independientes a través de la interfaz Parallel**. Cada agente mantiene su propia tabla Q indexada por la **observación de acción conjunta** (el único entero que codifica lo que *ambos* agentes hicieron la ronda pasada), para que pueda aprender "cuando ambos elegimos 0 la última vez, debería volver a elegir 0".
3. **Intentamos importar la biblioteca `pettingzoo` real** y ejecutar uno de sus entornos integrados (Piedra-Papel-Tijera) con una política aleatoria. Si PettingZoo no está instalado, nos saltamos este paso con un mensaje amigable.

### Qué deberías ver

| Etapa | Esperado |
|-------|----------|
| Ejecución aleatoria (AEC) | Retorno medio del episodio cerca de **0** — los agentes aleatorios eligen canales de forma independiente, coincidiendo y fallando en medidas aproximadamente iguales. |
| Aprendices Q independientes (Parallel) — primeros 100 eps | Aproximadamente **0** — todavía mayoritariamente aleatorio mientras los agentes exploran. |
| Aprendices Q independientes — últimos 100 eps | Fuertemente positivo, **+20 a +25** — **ha surgido la coordinación**: ambos agentes han aprendido a elegir de forma fiable el mismo canal en cada ronda. |

El gráfico `outputs/pettingzoo_coordination.png` muestra los retornos de los episodios individuales (gris) y una curva de la **Media** móvil (azul). La media suaviza los episodios ruidosos para que puedas ver la tendencia: los agentes pasan de un juego aleatorio descoordinado cerca de ~0 hacia una **coordinación** estable cerca de ~+25. La línea verde discontinua marca el techo de coordinación perfecta.

Si `pettingzoo` está instalado, el script también ejecuta `pettingzoo.classic.rps_v2` para demostrar que el script funciona contra la biblioteca real exactamente de la misma manera que funciona en nuestro entorno artesanal. Para habilitar esa sección:

```bash
source ../../venv/bin/activate
pip install "pettingzoo[classic]"
```

---

## ¿Por qué construir primero un entorno personalizado?

Porque **la API es la lección**. (Entender cómo estructurar la interacción entre múltiples agentes y el entorno es más importante que las reglas específicas del juego). El RL multiagente tiene muchos sabores (por turnos, tiempo real, cooperativo, competitivo, mixto), y todos encajan en el patrón AEC / Parallel. Una vez que has implementado esos dos bucles, cada entorno de PettingZoo es solo cuestión de conectar un constructor de entorno diferente — el código de entrenamiento se mantiene igual.

Así es exactamente como Gymnasium cambió el RL de un solo agente: convirtiendo el entorno en una caja negra tras una interfaz uniforme.

---

## Donde el Q-learning independiente ayuda y perjudica

Los juegos de coordinación son *agradecidos* — los agentes comparten el signo de la recompensa, por lo que sus intereses están alineados. Los aprendices independientes pueden resolver esto felizmente porque cualquier mejora de un agente ayuda al otro.

En los juegos **adversarios** (RPS), el Q-learning independiente oscila para siempre (a medida que un agente se adapta, el otro cambia su estrategia para contrarrestarlo, lo que lleva a una persecución sin fin). En los juegos **parcialmente observables**, no puede aprender en absoluto porque la "observación" es solo una pieza del estado (un agente podría ser penalizado por una buena acción simplemente porque no podía ver lo que el otro agente estaba haciendo). PettingZoo incluye ambos tipos de entorno para que puedas ver estos modos de fallo por ti mismo.

---

## Palabras Clave para Recordar

| Palabra | Significado |
|------|---------|
| **PettingZoo** | El Gymnasium del RL multiagente — una biblioteca de entornos MARL estandarizados |
| **AEC** | Agent-Environment-Cycle: un agente actúa por paso (por turnos) |
| **Parallel API** | Todos los agentes actúan simultáneamente en cada paso |
| **MPE** | Multi-Particle Environment, un banco de pruebas cooperativo/competitivo popular incluido en PettingZoo (a menudo implica puntos que se mueven navegando por tareas basadas en la física). |
| **CTDE** | Centralised Training, Decentralised Execution — entrenar con una visión global (acceso a todos los estados), desplegar con solo observaciones locales (cada agente actúa sobre su propia visión limitada). |
| **Independent Q-learning** | Cada agente ejecuta un Q-learning vainilla (el algoritmo estándar, sin modificar), ignorando que existen otros aprendices. |

---

## Resumen de una frase

> **PettingZoo da a cada entorno multiagente la misma forma — así que el código que escribes hoy seguirá funcionando mañana en un juego totalmente diferente.**

Una vez que los dos estilos de API sean algo natural, podrás pasar a MADDPG (crítico centralizado para agentes de control continuo), QMIX (mezcla de valores para equipos cooperativos), MAPPO (PPO multiagente) o cualquier otro algoritmo MARL moderno — la parte del entorno de tu código nunca tendrá que cambiar.
