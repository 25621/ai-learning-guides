# REINFORCE mit Baseline: Das Rauschen reduzieren

## Das Problem mit einfachem REINFORCE

Stellen Sie sich vor, Sie sind ein Schüler und versuchen zu entscheiden, ob Ihre Antwort in einem Test gut war.

**Schlechtes Feedback:** „Du hast 7 Punkte bekommen!“

Ist 7 gut? Wenn das Maximum 10 ist, ja! Wenn alle anderen 9 haben, nein! Ohne Kontext können Sie nicht sagen, ob Sie Ihren Antwortstil ändern sollten.

Genau das ist das Problem bei REINFORCE: Es verwendet die **rohen Returns** (G_t), um Aktionen zu bewerten. Ein Gesamtergebnis von 200 Punkten kann je nach Situation großartig oder schrecklich sein.

---

## Vorhang auf für die Baseline

Eine **Baseline** b(s) ist ein Referenzpunkt: „Welche Belohnung **erwarte** ich in dieser Situation?“

Anstatt zu fragen „War diese Aktion gut?“, fragen wir:

> **„War diese Aktion besser als das, was ich normalerweise erwarten würde?“**

```
Altes Signal: Update ∝ G_t
Neues Signal: Update ∝ (G_t - b(s_t))
```

**Beispiel aus dem echten Leben:** Sie haben in einem Mathetest 85 Punkte erreicht.
- Wenn der Klassendurchschnitt bei 60 liegt → Ihre Antwort war **25 Punkte über dem Durchschnitt** → super!
- Wenn der Klassendurchschnitt bei 90 liegt → Ihre Antwort war **5 Punkte unter dem Durchschnitt** → verbesserungswürdig!

Der **Vorteil (Advantage)** (G_t - b(s)) ist positiv, wenn Sie besser als erwartet abgeschnitten haben, und negativ, wenn Sie schlechter waren. Dies ist ein viel klareres Lernsignal!

---

## Was ist die Baseline?

Die natürliche Baseline ist die **Value-Funktion V(s)**:

> V(s) = „Erwartete Gesamtbelohnung, wenn ich mich im Zustand s befinde und meiner aktuellen Strategie folge“

Wir lernen dies mit einem separaten **Value-Netzwerk** (auch Baseline-Netzwerk oder Critic genannt):

```
Zustand  →  [128 Neuronen]  →  [128 Neuronen]  →  V(s)   (einzelne Zahl)
```

Für jeden Zustand, den der Agent besucht, sagt V(s) den erwarteten Return voraus. Wenn der tatsächliche Return G_t höher als V(s) ist, war die Aktion besser als erwartet!

---

## Zwei Netzwerke lernen gemeinsam

```
Episode findet statt
         ↓
Berechne tatsächliche Returns G_t
         ↓
         ┌─────────────────────────────┐
         │ Advantage = G_t - V(s_t)    │
         │  +: Aktion war besser        │
         │  -: Aktion war schlechter    │
         └─────────────────────────────┘
              ↓                  ↓
    Update Policy-Netzwerk   Update Value-Netzwerk
    (gute Aktionen mehr/     (Vorhersagen beim nächsten
     weniger wahrscheinlich)  Mal genauer machen)
```

**Beispiel aus dem echten Leben:** Zwei Freunde gehen zusammen in ein Restaurant.

- Freund 1 (Value-Netzwerk): „Ich sage voraus, dieses Gericht wird eine 7/10.“
- Freund 2 (Policy-Netzwerk): Sie probieren das Gericht und bewerten es mit 9/10.
- Advantage = 9 - 7 = +2 → „Das war besser als erwartet! Bestelle es wieder!“

Beim nächsten Besuch korrigiert Freund 1 seine Vorhersage näher an die 9/10. Freund 2 wird dieses Gericht beim nächsten Mal eher bestellen.

---

## Warum reduziert dies die Varianz?

**Mathematischer Beweis (Intuition):**

Ohne Baseline: `Gradient ∝ ∇log π(a|s) × G_t`

Die G_t-Werte variieren stark von Episode zu Episode:
```
Episode 1: G = [45, 44, 43, ..., 1]   (mittelmäßiges Spiel)
Episode 2: G = [500, 499, ..., 1]     (großartiges Spiel!)
Episode 3: G = [12, 11, ..., 1]       (schlechtes Spiel)
```

Die Gradientenschätzungen springen wild umher, weil G_t groß und verrauscht ist.

Mit Baseline: `Gradient ∝ ∇log π(a|s) × (G_t - V(s_t))`

Der Vorteil (Advantage) (G_t - V(s_t)) ist viel kleiner und um Null zentriert:
```
Episode 1: Advantage ≈ [-2, +1, -3, ..., 0]   (klein, zentriert)
Episode 2: Advantage ≈ [+10, +8, ..., +3]     (dieses Spiel WAR gut)
Episode 3: Advantage ≈ [-5, -6, ..., -2]      (dieses Spiel WAR schlecht)
```

**Beispiel aus dem echten Leben:** Messen der Laufgeschwindigkeit.
- Ohne Baseline: „Ich bin 8 km/h gelaufen“ (bedeutungslos ohne Kontext).
- Mit Baseline: „Ich bin 2 km/h SCHNELLER gelaufen als mein Durchschnitt“ (eindeutig gut!).

Der Vorteil ist immer ein Vergleich — er ist von Natur aus kleiner und stabiler.

---

## Entscheidend: Kein Bias!

Die Baseline ändert nicht, WAS der Algorithmus lernt — nur WIE SCHNELL und STABIL er lernt.

**Warum?** Weil der erwartete Vorteil mathematisch gesehen immer 0 ist:

> E[G_t - V(s_t)] = E[G_t] - V(s_t) = V(s_t) - V(s_t) = 0

Jedes b(s), das nicht von der Aktion abhängt, funktioniert als gültige Baseline!

**Beispiel aus dem echten Leben:** Eine Notenanpassung (Notenspiegel) ändert nichts daran, wer am besten abgeschnitten hat — sie macht die Ergebnisse lediglich leichter interpretierbar. Das Ranking bleibt gleich, nur die Skala ändert sich.

---

## Die Ergebnisse

```
Ohne Baseline  — Finaler 100-Ep.-Schnitt: 500.0, Grad.-Var.: 599.3
Mit Baseline — Finaler 100-Ep.-Schnitt: 491.4, Grad.-Var.: 578.8
```

Beide Methoden erreichen eine nahezu perfekte Leistung bei CartPole, aber beachten Sie:
1. Die **Gradientenvarianz** ist messbar (Diagramm auf der rechten Seite zeigt die Varianz während des Trainings).
2. Mit Baseline erreicht der Agent hohe Leistungen **zuverlässiger** — es gibt weniger Einbrüche auf niedrige Belohnungen während des Trainings.

Die Varianzreduktion ist in schwierigeren Umgebungen (LunarLander, MuJoCo) noch deutlicher.

---

## Wichtige Gleichungen

```
Baseline-Wert:   V(s) ← V(s) + α(G_t - V(s))   [MSE minimieren]
Policy-Gradient: θ ← θ + α ∇log π(a_t|s_t) · (G_t - V(s_t))
Advantage:       A_t = G_t - V(s_t)
```

---

## Wichtige Erkenntnisse

| Konzept | Einfache Erklärung |
|---------|---------------|
| **Baseline b(s)** | Erwartete Belohnung im Zustand s — unser Referenzpunkt |
| **Advantage A_t** | „War diese Aktion besser als erwartet?“ |
| **Value-Netzwerk** | Ein neuronales Netz, das lernt, erwartete Returns vorherzusagen |
| **Varianzreduktion** | Weniger Rauschen in den Gradientenschätzungen → stabileres Lernen |
| **Unbiased (Unverzerrt)** | Die Baseline verändert die Zielstrategie im Durchschnitt nicht; sie macht das Lernsignal lediglich weniger verrauscht |

---

## Was kommt als Nächstes?

Die Baseline ist eigentlich der Anfang von etwas viel Mächtigerem: den **Actor-Critic**-Methoden.

Anstatt V(s) erst am Ende einer Episode zu berechnen, aktualisiert der Actor-Critic V(s) bei jedem einzelnen Schritt mittels **Temporal Difference** Learning. Dies macht die Updates viel schneller und ermöglicht es dem Agenten, aus unvollständigen Episoden zu lernen!

Siehe `a2c_lunarlander.py` für die vollständige Actor-Critic-Implementierung.
