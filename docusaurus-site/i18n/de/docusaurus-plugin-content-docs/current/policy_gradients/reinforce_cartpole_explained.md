# REINFORCE: Einem Roboter beibringen, bessere Entscheidungen zu treffen

## Was versuchen wir zu erreichen?

Stellen Sie sich vor, Sie haben einen Roboter, der ein Videospiel spielt. Jede Sekunde muss der Roboter wählen:
**„Soll ich den Knopf drücken oder nicht?“**

Anstatt jede Situation in einer Tabelle auswendig zu lernen (wie beim Q-Learning), möchten wir, dass der Roboter ein **Rezept** lernt — einen Satz von Regeln, der direkt besagt: „In dieser Situation führe diese Aktion aus.“

Dieses Rezept wird **Policy** (Strategie, π, pi) genannt. Im Reinforcement Learning steht π für „die Regel zur Auswahl von Aktionen“.

---

## Der alte Weg vs. der neue Weg {#the-old-way-vs-the-new-way}

**Alter Weg (Q-Learning / DQN):** Lerne, wie GUT jede Aktion ist (Q-Werte) und wähle dann die beste.
> „Links drücken hat Score 7, Rechts drücken hat Score 5 → Drücke LINKS!“

**Neuer Weg (Policy Gradient):** Lerne direkt, welche Aktion zu WÄHLEN ist.
> „Wenn der Stab nach rechts kippt, drücke RECHTS mit 80 % Wahrscheinlichkeit, drücke LINKS mit 20 % Wahrscheinlichkeit.“
*(Das Wort **Gradient** bezieht sich auf den mathematischen „Schritt“, den wir machen, um diese Wahrscheinlichkeiten langsam in die richtige Richtung anzupassen.)*

**Beispiel aus dem echten Leben:** Fahrradfahren lernen.
- Der alte Weg: Berechne den *exakten Score* für „5 Grad nach links lehnen“ vs. „7 Grad nach links lehnen“.
- Der neue Weg: Übe einfach so lange, bis dein **Körper** es lernt — bewege den Fuß so, wie es sich richtig anfühlt!

---

## Wie funktioniert REINFORCE?

REINFORCE beobachtet den Roboter dabei, wie er ein ganzes Spiel von Anfang bis Ende spielt (eine **Episode**), und fragt dann: „Welche Aktionen haben zu einer guten Punktzahl geführt? Machen wir mehr davon!“

### Schritt für Schritt

**1. Eine Episode spielen**

Der Roboter trifft Entscheidungen und sammelt Erfahrungen:
```
Schritt 1: Zustand = [Stab kippt nach rechts] → Aktion = RECHTS drücken → Belohnung = +1
Schritt 2: Zustand = [Stab fast ausbalanciert] → Aktion = RECHTS drücken → Belohnung = +1
Schritt 3: Zustand = [Stab kippt nach links]  → Aktion = LINKS drücken  → Belohnung = +1
...
Schritt 47: Zustand = [Stab ist umgefallen!] → Episode beendet
```

**2. Returns berechnen**

Berechne für jeden Schritt G_t — die **Gesamtbelohnung ab diesem Zeitpunkt vorwärts**:
```
G bei Schritt 47 = 1
G bei Schritt 46 = 1 + 0,99 × 1 = 1,99
G bei Schritt 45 = 1 + 0,99 × 1,99 = 2,97
...
G bei Schritt 1  = 47 (ungefähr — höherer Return, da es der Start war)
```

Der Diskontierungsfaktor **γ = 0,99** bedeutet, dass Belohnungen in ferner Zukunft etwas weniger zählen.

**Beispiel aus dem echten Leben:** Einen goldenen Stern am 1. Schultag zu bekommen, fühlt sich aufregender an, als zu wissen, dass man am 100. Tag vielleicht einen bekommt. Zukünftige Belohnungen werden leicht „diskontiert“.

**3. Die Policy aktualisieren**

Für jede ausgeführte Aktion gilt:
> Wenn G_t HOCH war (diese Aktion führte zu einem tollen Ergebnis): **Mach es öfter!**
> Wenn G_t NIEDRIG war (diese Aktion führte zu einem schlechten Ergebnis): **Mach es seltener!**

Die Mathematik: `Loss = -log_prob(Aktion) × G_t`

Den Gradienten zu nehmen und die Policy zu aktualisieren, ist so, als würde man dem Roboter sagen:
*„Diese Aktion, die du bei Schritt 20 gemacht hast? Du solltest sie beim nächsten Mal 5 % häufiger wählen!“*

---

## Was ist ein Policy-Netzwerk?

Anstelle einer Tabelle verwenden wir ein **neuronales Netzwerk**, um die Policy darzustellen.

```
Beobachtung         Policy-Netzwerk       Aktionswahrscheinlichkeiten
[Wagen-Pos]    →    [128 Neuronen]    →    [LINKS drücken: 30 %]
[Wagen-Geschw] →    [128 Neuronen]         [RECHTS drücken: 70 %]
[Stab-Winkel]  →
[Stab-Geschw]  →
```

Das Netzwerk gibt **Wahrscheinlichkeiten** für jede Aktion aus. Wir ziehen dann eine Stichprobe:
> Würfeln → 1-30: LINKS drücken, 31-100: RECHTS drücken.

**Beispiel aus dem echten Leben:** Eine Wetter-App sagt: „70 % Regenwahrscheinlichkeit.“ Sie WISSEN nicht, ob es regnen wird — Sie entscheiden basierend auf der Wahrscheinlichkeit. Der Roboter macht das Gleiche!

---

## Normalisierung: Warum wir den Mittelwert abziehen

Bevor wir G_t für das Update verwenden, normalisieren wir:
```
G_normalisiert = (G - Mittelwert(G)) / Standardabweichung(G)
```

**Warum?** Stellen Sie sich vor, alle Belohnungen sind positiv (was sie bei CartPole sind — immer +1 pro Schritt). Ohne Normalisierung sieht JEDE Aktion „gut“ aus und das Update-Signal ist verwirrend. Nach der Normalisierung sind einige Returns positiv (überdurchschnittlich → öfter wählen) und einige negativ (unterdurchschnittlich → seltener wählen). Das Signal wird viel klarer!

**Beispiel aus dem echten Leben:** Ein Lehrer bewertet nach einem Notenspiegel. Wenn die Durchschnittspunktzahl 70 ist und Sie 85 haben, ist das super! Wenn der Durchschnitt aber 90 ist und Sie 85 haben, ist das unterdurchschnittlich. Die reine Punktzahl allein sagt nicht alles aus.

---

## Das Problem: Hohe Varianz

REINFORCE hat eine große Schwäche: **Varianz**. Die Returns G_t sind sehr verrauscht.

**Beispiel aus dem echten Leben:** Stellen Sie sich vor, Sie bewerten einen Koch, indem Sie nur EIN Gericht aus jedem Restaurant probieren. Manchmal hatte der Koch einen schlechten Tag, manchmal waren die Zutaten nicht gut. Ein Gericht reicht nicht aus, um zuverlässig zu wissen, ob das Restaurant gut ist!

REINFORCE wartet eine VOLLE Episode ab, bevor ein Update erfolgt. Eine Episode kann sehr glücklich verlaufen, eine andere sehr unglücklich. Die Gradienten springen wild hin und her.

Deshalb sieht die Lernkurve (im Diagramm) so zackig aus:
- Einige Durchläufe erreichen 500 (großartig!).
- Einige Durchläufe fallen auf 50 zurück (schrecklich!).

Trotz des Rauschens lernt REINFORCE schließlich — aber es erfordert Geduld.

---

## Die Ergebnisse

```
Episode  100 | Durchschnittliche Belohnung (letzte 100):  43.1
Episode  200 | Durchschnittliche Belohnung (letzte 100): 193.9
Episode  500 | Durchschnittliche Belohnung (letzte 100): 408.4
Episode 1000 | Durchschnittliche Belohnung (letzte 100): 500.0  ← Gelöst!
```

Der Roboter lernt, den Stab für die maximalen 500 Schritte zu balancieren — GELÖST!

Trotz der Varianzprobleme ist REINFORCE bei CartPole effektiv, weil:
1. Die Episoden kurz sind (wir bekommen also viele pro Trainingslauf).
2. Die optimale Policy einfach ist (meistens in die Richtung drücken, in die der Stab kippt).

---

## Wichtige Begriffe

| Begriff | Einfache Erklärung |
|---------|---------------|
| **Policy** | Das Rezept des Roboters zur Auswahl von Aktionen |
| **Episode** | Ein vollständiges Spiel vom Anfang bis zum Ende |
| **Return G_t** | Die gesamte zukünftige Belohnung ab diesem Moment |
| **Diskontierung γ** | Zukünftige Belohnungen zählen etwas weniger als sofortige |
| **Normalisieren** | Den Durchschnitt abziehen, damit das Signal klarer wird |
| **Varianz** | Wie stark die Gradientenschätzungen hin und her springen |

---

## Was kommt als Nächstes?

Die große Schwäche von REINFORCE ist die **Varianz**. Im nächsten Skript (`reinforce_baseline.py`) fügen wir eine **Baseline** hinzu, die dieses Rauschen drastisch reduziert — ohne das zu ändern, was der Algorithmus im Durchschnitt lernt.

Die Kernidee: Anstatt zu fragen „war diese Aktion gut?“, fragen wir „war diese Aktion **besser als erwartet**?“. Diese kleine Änderung macht das Lernen viel stabiler.
