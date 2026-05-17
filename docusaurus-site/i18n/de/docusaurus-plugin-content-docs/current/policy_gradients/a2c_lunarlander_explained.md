# A2C: Actor und Critic arbeiten zusammen

## Die Kernidee

REINFORCE wartet, bis ein Spiel komplett beendet ist, bevor ein Update durchgeführt wird. Das ist wie ein Trainer, der ein ganzes Fußballspiel schweigend beobachtet und erst am Ende Feedback gibt.

**A2C (Advantage Actor-Critic)** gibt Feedback WÄHREND des Spiels – alle paar Schritte hält der Trainer inne und sagt: „Dieser Pass war großartig! Dieser Tackle war schlecht!“

Das ist viel schneller und effizienter.

---

## Die zwei Hauptakteure

> **Was ist LunarLander?** In diesem Dokument verwenden wir die **LunarLander**-Umgebung – eine Physiksimulation, in der Sie ein kleines Raumschiff steuern und es sanft auf einer Landefläche auf dem Mond landen müssen, wobei Sie drei Triebwerke (links, Haupttriebwerk und rechts) verwenden. Es ist ein Standard-Benchmark im Reinforcement Learning, verfügbar in der Gymnasium-Bibliothek.

### Der Actor (Akteur) 🎭
Der **Actor** ist die Policy (Strategie) – er entscheidet, welche Aktion ausgeführt werden soll.

> „Ich bin in diesem Zustand. Soll ich das linke oder das rechte Triebwerk zünden?“

**Beispiel aus dem echten Leben:** Der *Fahrer* eines Autos, der das Lenkrad dreht und die Pedale drückt.

### Der Critic (Kritiker) 🎬
Der **Critic** schätzt ein, wie gut die aktuelle Situation ist – den Wert V(s).

> „In DIESEM Zustand zu sein, ist etwa +150 Punkte an zukünftiger Gesamtbelohnung wert.“

**Beispiel aus dem echten Leben:** Der *Beifahrer/Navigator*, der neben dem Fahrer sitzt und sagt: „Wir sind auf einer guten Straße – wir werden voraussichtlich in 30 Minuten ankommen“ oder „Wir geraten in einen Stau – das wird langsam vorangehen.“

### Sie teilen sich ein Gehirn
In unserer Implementierung verwenden beide das **gleiche neuronale Netzwerk-Backbone**:

```
          Zustand (8 Zahlen für LunarLander)
                       ↓
          ┌─────────────────────────┐
          │  Gemeinsame Schichten   │
          │  [256 Neuronen] → ReLU  │
          │  [256 Neuronen] → ReLU  │
          └────────┬────────┬───────┘
                   ↓        ↓
              Actor-Head  Critic-Head
             [4 Ausgänge] [1 Ausgang]
           (Aktions-Wsk.)   (V(s))
```

- **ReLU** (Rectified Linear Unit): eine Aktivierungsfunktion, die nach jeder Schicht angewendet wird – sie gibt `max(0, x)` aus, behält also positive Werte bei und setzt negative Werte auf Null. Dies ermöglicht es dem Netzwerk, nicht-lineare Muster zu lernen.
- **Aktions-Wsk.**: die Wahrscheinlichkeit für jede der 4 Aktionen. Der Actor zieht aus dieser Verteilung Stichproben, um in jedem Schritt eine Aktion auszuwählen.

**Beispiel aus dem echten Leben:** Ein Gehirn, zwei Aufgaben – wie ein Taxifahrer, der sowohl fährt (Actor) ALS AUCH weiß, ob die Route gut ist (Critic). Das Teilen des Gehirns bedeutet schnelleres Lernen!

---

## Der Advantage: War das besser als erwartet?

Genau wie REINFORCE mit Baseline berechnet A2C den **Advantage** (Vorteil):

> A(s, a) = „Tatsächliches Ergebnis“ − „Was wir erwartet haben“

Aber hier kommt das „tatsächliche Ergebnis“ aus dem **n-Step Bootstrap** des Critics (**Bootstrapping** = die eigene Vorhersage V(s) des Critics verwenden, um den Wert zukünftiger Schritte zu approximieren, anstatt zu warten, bis die Episode tatsächlich endet – etwa so, als würde man seine Abschlussnote in der Mitte des Semesters anhand der aktuellen Noten schätzen):

```
Tatsächlicher TD-Return: r_t + γ · r_{t+1} + γ² · r_{t+2} + ... + γⁿ · V(s_{t+n})
Advantage A_t = TD-Return - V(s_t)
```

**Beispiel aus dem echten Leben:** Sie erwarten, in diesem Spiel 3 Tore zu schießen (V(s)). Wenn Sie 5 Tore schießen, ist Ihr Advantage +2. Wenn Sie 1 Tor schießen, ist Ihr Advantage -2.

Positiver Advantage → „Diese Aktion hat mehr geholfen als erwartet → mach sie öfter!“
Negativer Advantage → „Diese Aktion hat weniger geholfen als erwartet → mach sie seltener!“

---

## Warum mehrere parallele Umgebungen verwenden?

Unser A2C verwendet **8 Kopien** von LunarLander, die gleichzeitig laufen!

**Warum?** Weil Erfahrungen aus einer einzelnen Umgebung **korreliert** sind – ein Schritt folgt eng auf den vorherigen Schritt. Diese Korrelation täuscht dem neuronalen Netzwerk vor, dass Muster zuverlässiger sind, als sie es tatsächlich sind.

Mit 8 Umgebungen liefert jeder Schritt 8 unabhängige Erfahrungen aus sehr unterschiedlichen Situationen. Dies bricht die Korrelation auf und stabilisiert das Training massiv.

**Beispiel aus dem echten Leben:** Um etwas über das Wetter zu lernen, was ist besser:
- Eine Stadt 8 aufeinanderfolgende Stunden lang beobachten (korreliert – wenn es um 14 Uhr sonnig war, ist es wahrscheinlich auch um 15 Uhr sonnig).
- 8 Städte gleichzeitig beobachten (dekorreliert – unterschiedliche Wettermuster, mehr Informationen!).

```
Umgebung 1: [auf dem Mond gelandet, links zünden, Absturz, Reset...]
Umgebung 2: [falle zu schnell, beide zünden, schweben, landen...]
Umgebung 3: [neige mich nach rechts, rechts zünden, stabilisieren, landen...]
...
Umgebung 8: [drifte nach links, links zünden, stabil, ...]
```

Alle 8 aktualisieren das Netzwerk gleichzeitig – 8-mal mehr vielfältige Erfahrung pro Update!

---

## N-Step-Updates: Warten Sie nicht auf das Spielende

REINFORCE wartet eine volle Episode ab (könnte 1000 Schritte sein!).

A2C aktualisiert alle **n_steps = 128 Schritte**:

```
Spiele 128 Schritte in 8 Umgebungen
    → Erhalte 128 × 8 = 1024 Erfahrungs-Tupel
    → Berechne Advantages und Returns
    → Aktualisiere Actor und Critic
    → Spiele weitere 128 Schritte...
```

**Beispiel aus dem echten Leben:** Ein Student, der für eine Prüfung lernt.
- REINFORCE-Stil: Das gesamte Lehrbuch lesen, DANN Übungstests machen.
- A2C-Stil: 10 Seiten lesen, ein Quiz machen, 10 weitere Seiten lesen, ein Quiz machen...

Häufigeres Feedback = schnelleres Lernen!

---

## Drei kombinierte Verluste (Losses)

A2C trainiert mit drei Verlusttermen gleichzeitig:

Ein **Loss** ist der Wert, den der Optimizer zu minimieren versucht. Ein kleinerer Loss bedeutet, dass das aktuelle Verhalten des Netzwerks näher am Trainingsziel liegt.

### 1. Actor-Loss (Policy-Gradient)
Macht vorteilhafte Aktionen wahrscheinlicher:
```
L_actor = -E[log π(a|s) · A(s,a)]
```
Wenn A > 0: Wahrscheinlichkeit dieser Aktion erhöhen
Wenn A < 0: Wahrscheinlichkeit dieser Aktion verringern

### 2. Critic-Loss (Value-Function MSE)
Macht die Wertvorhersagen genauer (**MSE** = Mean Squared Error: den Vorhersagefehler quadrieren und den Durchschnitt bilden – das Quadrieren bestraft große Fehler stärker als kleine):
```
L_critic = E[(V(s) - return)²]
```
Wie das Training eines beliebigen **Regressionsmodells** (Regression = Vorhersage einer kontinuierlichen Zahl, hier der erwartete Return V(s)) – minimiere den Vorhersagefehler.

### 3. Entropie-Bonus (Exploration)
Verhindert, dass die Policy zu schnell zu sicher wird:
```
L_entropy = -H[π(·|s)] = E[log π(a|s)]
```
Hohe Entropie = vielfältige Aktionswahl = Exploration
Niedrige Entropie = sichere, enge Auswahl = Exploitation

**Beispiel aus dem echten Leben:** Der Entropie-Bonus ist wie ein Lehrer, der sagt: „Rate nicht einfach A bei jeder Multiple-Choice-Frage! Probiere verschiedene Antworten aus, damit du lernst, was funktioniert.“

```
Gesamtverlust = L_actor + 0,5 × L_critic - 0,01 × Entropie
```

---

## LunarLander: Eine größere Herausforderung

**LunarLander-v3** ist eine Gymnasium-Umgebung (ehemals OpenAI Gym) – „v3“ ist die Versionsnummer, die die dritte Revision dieser Umgebung angibt. Der Agent steuert ein kleines Raumschiff, das sicher auf einer dafür vorgesehenen Fläche auf dem Mond landen muss. Es ist viel schwieriger als CartPole:
- 8-dimensionaler Zustandsraum (Position, Geschwindigkeit, Winkel, Beinkontakt, Treibstoff)
- 4 diskrete Aktionen (nichts tun, links zünden, Haupttriebwerk zünden, rechts zünden)
- Belohnung: +100 für die Landung, -100 für den Absturz, kleine Treibstoffstrafen

Die Trainingskurve zeigt eine allmähliche Verbesserung von stark negativen Belohnungen hin zu positiven.
A2C auf LunarLander erfordert erhebliche Erfahrung, bevor der Lander die grundlegende Stabilität lernt.

---

## Wichtige Gleichungen

```
n-Step Return:  G_t = r_t + γ·r_{t+1} + ... + γⁿ·V(s_{t+n})
Advantage:      A_t = G_t - V(s_t)
Actor-Update:   θ_π ← θ_π - α ∇ L_actor
Critic-Update:  θ_V ← θ_V - α ∇ L_critic
```

---

## Wichtige Erkenntnisse

| Konzept | Einfache Erklärung |
|---------|---------------|
| **Actor** | Die Policy – entscheidet, was zu tun ist |
| **Critic** | Die Value-Funktion – beurteilt, wie gut die Situation ist |
| **Advantage** | „War das besser als erwartet?“ (tatsächlich - erwartet) |
| **N-Step Return** | Blicke n Schritte in die Zukunft, bevor Bootstrapping mit V(s) erfolgt |
| **Parallele Envs** | Mehrere Umgebungen für dekorrelierte, vielfältige Erfahrung |
| **Entropie-Bonus** | Ermutigt den Actor, weiterhin neue Dinge auszuprobieren |

---

## Was kommt als Nächstes?

A2C ist großartig, hat aber eine große Schwäche: Es aktualisiert die Policy manchmal zu aggressiv.
Ein einziges schlechtes Update kann all das gute Lernen aus einem vorherigen Update zerstören.

**PPO (Proximal Policy Optimization)** behebt dies mit einem cleveren „Sicherheits-Clip“, der verhindert, dass ein einzelnes Update die Policy zu stark verändert.

Siehe `ppo_scratch.py` für die PPO-Implementierung!
