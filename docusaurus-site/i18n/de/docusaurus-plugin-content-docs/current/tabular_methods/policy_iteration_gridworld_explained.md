# Policy Iteration für GridWorld 🗺️

## Was ist das?

Stellen Sie sich vor, Sie spielen ein Brettspiel auf einem **4×4-Gitter** (wie ein winziges Schachbrett). Sie beginnen in einer Ecke und müssen die andere Ecke erreichen. Jeder Schritt kostet 1 Punkt (Sie wollen keine Schritte verschwenden!), und das Erreichen des Ziels bringt Ihnen keine Extrapunkte ein – Sie wollen einfach so schnell wie möglich dorthin gelangen.

**Policy Iteration** ist die Methode, mit der ein Computer die **besten Züge für jedes Quadrat** auf dem Brett herausfindet – und zwar für alle gleichzeitig!

---

## Die Kernidee: Zwei Schritte, immer wieder

Stellen Sie es sich wie das Aufräumen Ihres Zimmers mit einem Helfer vor:

1. **Schritt 1 — Herausfinden, wie gut jedes Quadrat ist (Policy Evaluation)**
   Ihr Helfer geht jedes Quadrat ab und notiert: „Wenn ich dem aktuellen Plan folge, wie viele Schritte brauche ich von hier aus bis zum Ausgang?“ Er macht das so lange, bis sich die Zahlen nicht mehr ändern.

2. **Schritt 2 — Den Plan verbessern (Policy Improvement)**
   Nun schauen Sie sich jedes Quadrat an und fragen: „Gibt es eine bessere Richtung, in die ich von hier aus gehen könnte?“ Wenn ja, aktualisieren Sie den Plan!

Wiederholen Sie die Schritte 1 und 2, bis sich der Plan nicht mehr ändert – das ist die **optimale Policy** (Strategie)!

**Beispiel aus dem echten Leben:** Stellen Sie sich vor, Sie suchen den schnellsten Weg zur Schule. Zuerst raten Sie eine Route und stoppen die Zeit (Schritt 1). Dann schauen Sie sich jede Straßenecke an und fragen: „Gibt es von hier aus eine Abkürzung?“ (Schritt 2). Sie aktualisieren Ihre Route und wiederholen das Ganze, bis Sie keine weiteren Abkürzungen mehr finden können!

---

## Was unser Code herausgefunden hat

Unsere 4×4 GridWorld hat zwei Zielzustände (Ecken), und der Agent zahlt -1 pro Schritt. Die Policy Iteration konvergierte in nur **4 Runden** (insgesamt 139 Evaluationsdurchläufe):

```
Zustandswerte V(s):       Optimale Policy:
 0.0  -1.0  -1.9  -2.7    T   ←   ←   ↓
-1.0  -1.9  -2.7  -1.9    ↑   ↑   ↑   ↓
-1.9  -2.7  -1.9  -1.0    ↑   ↑   ↓   ↓
-2.7  -1.9  -1.0   0.0    ↑   →   →   T
```

**Die Werte ergeben absolut Sinn!** Quadrate direkt neben einem Ziel haben den Wert -1 (ein Schritt entfernt). Quadrate, die zwei Schritte entfernt sind, haben den Wert -1,9 (= -1 + 0,9 × -1) und so weiter.

---

## Beispiele aus dem echten Leben

- **GPS-Navigation**: Den besten Abzweig an *jeder* Kreuzung auf der Karte berechnen.
- **Aufzugsteuerung**: In welches Stockwerk sollte der Aufzug fahren, wenn mehrere Anforderungen vorliegen?
- **Fabrikroboter**: Planung des effizientesten Weges durch ein Lagerhaus-Gitter.

---

## Wichtige Begriffe zum Merken

- **Policy**: Der Plan — welche Aktion in jedem Zustand ausgeführt werden soll.
- **Value Function V(s)**: Wie gut es ist, im Zustand s zu sein (höher = näher am Ziel).
- **Policy Evaluation**: Berechnung, wie gut der aktuelle Plan ist.
- **Policy Improvement**: Den Plan mithilfe der Value Function verbessern.
- **Optimal Policy**: Der bestmögliche Plan — kann nicht weiter verbessert werden.

Die Kernbotschaft: **Sie müssen nicht jeden möglichen Plan ausprobieren! Verbessern Sie einfach den aktuellen Plan immer weiter, und Sie werden den besten Plan in sehr wenigen Runden finden.**
