# Deep Q-Network (DQN) von Grund auf 🧠

## Das Problem mit linearen Modellen

Erinnern Sie sich an unsere lineare Formel von vorhin?

> Score = w₁ × Wagenposition + w₂ × Wagengeschwindigkeit + w₃ × Stabwinkel + w₄ × Stabgeschwindigkeit

Das funktioniert für CartPole ganz gut, aber was ist mit einem Videospiel, bei dem man tausende Pixel sieht? Dafür kann man kein einfaches Rezept schreiben!

Wir brauchen etwas, das komplizierte Situationen betrachten und die beste Aktion herausfinden kann. Dieses Etwas ist ein **neuronales Netzwerk**.

---

## Was ist ein neuronales Netzwerk?

Denken Sie an Ihr Gehirn. Millionen winziger Zellen, sogenannte Neuronen, kommunizieren miteinander. Wenn Sie etwas Heißes berühren, senden Neuronen Signale: „HEISS! → Hand SOFORT wegziehen!“ Jedes Neuron gibt Informationen weiter, und zusammen treffen sie eine kluge Entscheidung.

Ein **neuronales Netzwerk auf einem Computer** funktioniert genauso:

```
Eingabeschicht      Versteckte Schicht 1   Versteckte Schicht 2   Ausgabeschicht
[Wagen-Pos]    →    [128 Neuronen]    →    [128 Neuronen]    →    [Score LINKS]
[Wagen-Geschw] →    [  ...       ]         [  ...       ]         [Score RECHTS]
[Stab-Winkel]  →
[Stab-Geschw]  →
```

Jeder Pfeil hat ein **Gewicht** (wie stark diese Verbindung ist). Es gibt tausende dieser Gewichte – und das Netzwerk lernt sie ALLE!

**Beispiel aus dem echten Leben:** Ein Koch in einem Restaurant probiert Ihr Essen und passt hunderte Zutaten gleichzeitig an. Jede Geschmacksknospe ist wie ein Neuron, und zusammen sagen sie dem Koch: „Mehr Salz“ oder „Weniger Pfeffer“. Das Training des Netzwerks ist wie ein Koch, der über tausende Mahlzeiten hinweg lernt.

---

## DQN = Deep Q-Network

**DQN** (Deep Q-Network) wurde 2013 von DeepMind erfunden. Sie nahmen die alte Q-Learning-Formel und ersetzten die Q-Tabelle durch ein neuronales Netzwerk!

Anstatt:
> Q-Tabelle[Zustand][Aktion] = Score

Haben wir:
> Q-Netzwerk(Zustand) → [Score_für_Links, Score_für_Rechts]

Das Netzwerk nimmt den Zustand als Eingabe und gibt Q-Werte für ALLE Aktionen gleichzeitig aus. Das ist viel effizienter, als sie einzeln zu berechnen!

---

## Dieses Skript: Die „naive“ Version

Dieses Skript zeigt DQN **ohne** spezielle Tricks. Es macht einfach Folgendes:
1. Den Zustand betrachten
2. Das Netzwerk fragen: „Wie gut ist Links? Wie gut ist Rechts?“
3. Die Aktion mit dem höheren Score ausführen
4. Eine Belohnung erhalten, das Netzwerk aktualisieren

**Dies ist absichtlich instabil!** Stellen Sie es sich wie einen Schüler vor, der jedes Mal, wenn er etwas Neues lernt, seine bisherigen Lektionen sofort wieder vergisst. Das Netzwerk aktualisiert sich nach jedem einzelnen Schritt, was zu Chaos führt.

**Beispiel aus dem echten Leben:** Stellen Sie sich vor, Sie lernen kochen, indem Sie nach jedem einzelnen Bissen Ihr gesamtes Rezept ändern. Sie könnten von „zu salzig“ zu „gar kein Salz“ zu „viel zu salzig“ wechseln und nie die richtige Menge finden. Genau das passiert hier!

---

## Was Sie sehen werden

Wenn Sie `dqn_cartpole.py` ausführen:
- Die Scores könnten stark schwanken (instabiles Lernen).
- Manchmal wird der Agent richtig gut und vergisst dann plötzlich alles wieder.
- Das Loss-Diagramm zeigt wilde Ausschläge.

**Das ist zu erwarten!** Es zeigt, WARUM wir Verbesserungen brauchen – Experience Replay und Target-Netzwerke. Diese folgen als Nächstes!

---

## Der ε-Greedy-Trick 🎲

Der Roboter wählt nicht immer die beste Aktion. Manchmal wählt er zufällig!

Warum? Weil er, wenn er immer nur das wählt, was am besten erscheint, vielleicht nie bessere Optionen entdeckt.

> Mit Wahrscheinlichkeit ε (Epsilon): wähle eine ZUFÄLLIGE Aktion (explorieren!)
> Mit Wahrscheinlichkeit 1-ε: wähle die BESTE bekannte Aktion (ausnutzen / exploit!)

Wir beginnen mit ε = 1,0 (100 % zufällig) und verringern es langsam auf ε = 0,01 (1 % zufällig). Auf diese Weise exploriert der Roboter am Anfang viel und konzentriert sich dann auf das, was er gelernt hat.

**Beispiel aus dem echten Leben:** Wenn Sie eine neue Stadt besuchen, probieren Sie am Anfang vielleicht zufällige Restaurants aus (explorieren). Nach einer Weile gehen Sie zu Ihren Favoriten zurück (ausnutzen). Aber gelegentlich probieren Sie immer noch etwas Neues aus, nur für den Fall, dass es dort einen Geheimtipp gibt!

---

## Wichtige Begriffe

| Begriff | Bedeutung |
|------|---------|
| **Neuronales Netzwerk** | Schichten verbundener mathematischer Neuronen, die aus Daten lernen |
| **Deep** | Mehr als eine versteckte Schicht (daher „Deep“ Learning) |
| **DQN** | Deep Q-Network — verwendet ein neuronales Netz statt einer Q-Tabelle |
| **ε-Greedy** | Strategie: Manchmal zufällig explorieren, sonst bestes Wissen nutzen |
| **Instabilität** | Das Netzwerk „vergisst“ ständig, weil sich Aktualisierungen gegenseitig stören |

---

## Was fehlt (und warum es wichtig ist)

Dieses naive DQN hat zwei große Probleme:

1. **Korrelierte Updates**: Jede Erfahrung kommt der Reihe nach (Schritt 1, Schritt 2, Schritt 3...). Wenn Schritt 5 schlecht war, werden ALLE naheliegenden Updates gleichermaßen verwirrt.
   
2. **Moving Target**: Nach jedem Update ändert sich das Netzwerk. Aber das nächste Update verwendet das GLEICHE Netzwerk, um zu berechnen, wie das Ziel (Target) aussehen sollte. Es ist, als würde man auf ein bewegliches Ziel schießen!

Diese Probleme werden durch **Experience Replay** und **Target-Netzwerke** in den nächsten Skripten gelöst. Zusammen machen sie aus DQN einen Champion!
