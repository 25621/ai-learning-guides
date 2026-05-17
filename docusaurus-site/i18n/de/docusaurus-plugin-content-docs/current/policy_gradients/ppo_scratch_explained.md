# PPO: Sichere und beständige Policy-Updates

## Das Problem mit A2C

Stellen Sie sich vor, Sie lernen, einen Besenstiel auf Ihrem Finger zu balancieren. Nach wochenlanger Übung können Sie ihn 30 Sekunden lang oben halten!

Nun gibt Ihnen Ihr Trainer einen Rat: „Neige dein Handgelenk ein kleines Stück weiter nach links.“

**Guter Rat → vorsichtige Änderung → immer noch 30 Sekunden Balance ✓**

Aber was wäre, wenn der Trainer überreagiert und sagt: „NEIGE DICH SOFORT EXTREM NACH LINKS!“
Sie korrigieren übermäßig → der Besenstiel fällt um → Sie haben den Fortschritt von Wochen verloren.

Dies ist das A2C-Problem: **Große Gradienten-Updates können eine gute Policy zerstören.**

**PPO (Proximal Policy Optimization)** ist ein Sicherheitssystem, das dies verhindert.

---

## Die Kernidee: Bleib nah an dem, was funktioniert hat

Die zentrale Einschränkung von PPO lautet:

> **„Ändere die Policy in einem einzelnen Update nicht zu stark.“**

Vor einem Update haben wir die „alte“ Policy π_old.
Nach dem Update haben wir die „neue“ Policy π_new.

PPO misst die Änderung der Policy mit dem **Wahrscheinlichkeitsverhältnis (Ratio)**:

```
r(θ) = π_new(a|s) / π_old(a|s)
```

- r = 1,0: Policy unverändert.
- r = 1,5: Die neue Policy wählt diese Aktion mit einer um 50 % höheren Wahrscheinlichkeit.
- r = 0,5: Die neue Policy wählt diese Aktion mit einer um 50 % geringeren Wahrscheinlichkeit.

**Beispiel aus dem echten Leben:** Sie sind ein Koch und passen ein Rezept an.
- r = 1,0: Die gleiche Menge Salz wie zuvor.
- r = 2,0: Doppelt so viel Salz — zu extrem!
- r = 0,9: 10 % weniger Salz — eine kleine, sichere Änderung.

---

## Der Clipping-Trick

PPO begrenzt (clippt) das Verhältnis, damit es im Bereich [1-ε, 1+ε] bleibt (typischerweise ε = 0,2):

```
L_CLIP = E[min(r(θ) · A,  clip(r(θ), 1-ε, 1+ε) · A)]
```

Lassen Sie uns das aufschlüsseln:

**Fall 1: Die Aktion war GUT (A > 0)**

Wir wollen diese Aktion öfter ausführen (r > 1). Aber wir deckeln die Steigerung:
```
wenn r > 1,2: auf 1,2 begrenzen, kein weiterer Anreiz für Steigerung
```
Dies verhindert, dass wir zu weit in eine Richtung ausschlagen.

**Fall 2: Die Aktion war SCHLECHT (A < 0)**

Wir wollen diese Aktion seltener ausführen (r < 1). Aber auch hier deckeln wir:
```
wenn r < 0,8: auf 0,8 begrenzen, keine weitere Strafe für Verringerung
```

**Visualisierung:**
```
ε = 0,2, daher ist das sichere Verhältnis-Fenster 0,8 bis 1,2.

GUTE Aktion (A > 0): Erhöhe Aktionswahrscheinlichkeit, aber höre nach 1,2 auf zu belohnen
Verhältnis r:  0,6      0,8      1,0      1,2      1,4
Anreiz:         ↑        ↑        ↑        ↑        -
Bedeutung:   zu niedrig  ok      alt      max    begrenzt (clipped)

SCHLECHTE Aktion (A < 0): Verringere Wahrscheinlichkeit, aber höre unter 0,8 auf zu bestrafen
Verhältnis r:  0,6      0,8      1,0      1,2      1,4
Anreiz:         -        ↓        ↓        ↓        ↓
Bedeutung:   begrenzt   max      alt       ok    zu hoch
```

Das `-` markiert den flachen, begrenzten Bereich. In diesem Bereich verbessert ein noch extremeres Wahrscheinlichkeitsverhältnis das Ziel nicht weiter, sodass PPO keinen zusätzlichen Anreiz hat, weiter zu gehen.

**Beispiel aus dem echten Leben:** Ein Geschwindigkeitsbegrenzer im Auto. Man kann beschleunigen, aber sobald man 120 km/h erreicht, greift der Begrenzer ein und lässt einen nicht schneller fahren. Er hält einen sicher, ohne die Bewegung zu stoppen.

---

## Warum dies katastrophale Updates verhindert

Ein **katastrophales Update** tritt auf, wenn eine einzige große Änderung der Policy alles zerstört, was der Agent gelernt hat – Stunden an Training sind in einem einzigen Gradientenschritt verloren.

Ohne Clipping: Ein großer Gradientenschritt könnte die Policy drastisch verändern.
Mit Clipping: Der Gradient ist außerhalb von [1-ε, 1+ε] gleich Null, sodass sich die Policy pro Schritt nur ein kleines Stück bewegen kann.

**Beispiel aus dem echten Leben:** Ein guter Chirurg macht kleine, präzise Schnitte – keine großen, ausladenden. PPO ist der „vorsichtige Chirurg“ unter den RL-Optimizern.

---

## GAE: Intelligentere Vorteilsschätzungen

PPO verwendet **Generalized Advantage Estimation (GAE)**, um den Vorteil (Advantage) zu berechnen:

```
δ_t = r_t + γ · V(s_{t+1}) - V(s_t)          (TD-Fehler)
A_t = δ_t + γλ · δ_{t+1} + (γλ)² · δ_{t+2} + ...
```

GAE hat einen Parameter λ (Lambda):
- λ = 0: Verwende nur den One-Step TD-Fehler (geringe Varianz, hoher Bias).
- λ = 1: Verwende die vollständigen Monte-Carlo-Returns (hohe Varianz, geringer Bias).
- λ = 0,95: Eine gute Balance zwischen beiden!

**Beispiel aus dem echten Leben:** Planung einer Autoreise.
- λ=0: Schau nur auf die nächsten 10 km (sicher, aber man könnte später eine Abkürzung verpassen).
- λ=1: Betrachte die gesamte 500-km-Reise (mehr Informationen, aber sehr unsicher).
- λ=0,95: Schau weit voraus, aber gewichte die nahen Straßen stärker ← beste Balance!

---

## Mehrere Epochen: Daten effizient wiederverwenden

Nach dem Sammeln eines Erfahrungs-Batches (Rollout) verwirft REINFORCE diesen nach EINEM Update. PPO verwendet jeden Batch für **K Epochen** wieder (typischerweise 4–10 Durchgänge durch dieselben Daten):

```
Sammle 512 Schritte × 4 Umgebungen = 2048 Übergänge
Epoche 1: 32 Minibatches × jeweils ein Update
Epoche 2: Mischen, 32 weitere Minibatches × jeweils ein Update
Epoche 3: ...
Epoche 4: ...
```

**Was ist ein „Minibatch“?** Ein Update mit allen 2048 Übergängen auf einmal ist langsam und speicherintensiv; ein Update mit nur einem Übergang nach dem anderen ist sehr verrauscht. Ein **Minibatch** ist ein kleiner Block dazwischen — hier 2048 ÷ 32 = **64 Übergänge pro Minibatch**. Wir berechnen einen Gradientenschritt pro Minibatch, sodass jede Epoche 32 kleine, stabile Updates anstelle von einem riesigen ausführt.

Das Clipping stellt sicher, dass diese mehrfachen Durchgänge nicht überschießen — ohne Clipping würden mehrere Epochen die Policy zerstören, indem sie sie zu weit treiben!

**Beispiel aus dem echten Leben:** Ein Schüler hat 30 Übungsaufgaben.
- REINFORCE: Jede Aufgabe einmal machen, ein wenig lernen, dann wegwerfen.
- PPO: Jede Aufgabe 4-mal machen (jedes Mal unter einem anderen Blickwinkel), die Änderungen begrenzen, damit man keine falschen Muster auswendig lernt.

---

## Der vollständige PPO-Loss

```
L = L_CLIP - c₁ · L_entropy + c₂ · L_critic

L_CLIP    = begrenzter Policy-Gradient
L_entropy = Entropie-Bonus (fördert Exploration)
L_critic  = MSE zwischen V(s) und den Returns
```

Typische Koeffizienten: c₁ = 0,01 (Entropie), c₂ = 0,5 (Critic).

**Zwei Begriffe näher erklärt:**

- **Policy-Gradient** — die „Actor“-Hälfte des Loss. Er nutzt das Gradientensignal, um die Policy in Richtung von Aktionen mit höherem Vorteil und weg von Aktionen mit geringerem Vorteil zu drücken. Dies ist die gleiche Kernidee wie bei REINFORCE. PPO fügt lediglich den Clipping-Wrapper hinzu.
- **MSE (Mean Squared Error)** — die „Critic“-Hälfte des Loss. Der Critic V(s) sagt den erwarteten Return eines Zustands voraus; wir vergleichen seine Vorhersage mit dem tatsächlichen Return und quadrieren die Differenz: `MSE = Mittelwert((V(s) - Return)²)`. Das Quadrieren bestraft große Fehler stärker als kleine und liefert ein glattes Signal für das Training.

---

## Ergebnisse

```
Update  200 | Durchschnittliche Belohnung: ~120
Update  400 | Durchschnittliche Belohnung: ~280
Update  800 | Durchschnittliche Belohnung: ~280-300
```

PPO auf CartPole zeigt eine stetige Verbesserung, neigt aber zu einem Plateau um 280–300. (Ein **Plateau** bedeutet, dass die Lernkurve abflacht — die Belohnung verbessert sich nicht weiter, obwohl das Training fortgesetzt wird.) Dies ist eigentlich zu erwarten — PPO ist für schwierigere Umgebungen mit längeren Episoden konzipiert.

Eine interessante Beobachtung: **REINFORCE hat CartPole schneller gelöst!** (500er-Schnitt vs. 300er-Schnitt). Warum? CartPole-Episoden sind kurz (≤500 Schritte), daher sind die exakten Returns von REINFORCE sehr genau. Die Bootstrapping-Schätzungen von PPO fügen unnötige Komplexität hinzu. PPO glänzt wirklich in Umgebungen, in denen das Warten auf vollständige Episoden unpraktisch ist (wie beim BipedalWalker).

---

## Wichtige Gleichungen

```
Verhältnis (Ratio): r_t(θ) = π_θ(a_t|s_t) / π_θ_alt(a_t|s_t)
Clip-Loss:          L_CLIP = E[min(r_t A_t, clip(r_t, 1-ε, 1+ε) · A_t)]
GAE:                A_t = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}
```

---

## Wichtige Erkenntnisse

| Konzept | Einfache Erklärung |
|---------|---------------|
| **Ratio r(θ)** | Wie stark sich die Policy bei dieser Aktion geändert hat |
| **Clip ε** | Die Sicherheitsgrenze — ändere die Policy nicht mehr als diesen Wert |
| **GAE** | Eine kluge Methode, Vorteile zu schätzen, indem man mehrere Schritte vorausblickt |
| **Dateneffizienz** | Jeder Rollout wird in mehreren parallelen Umgebungen gesammelt und dann für K Epochen von Minibatch-Updates wiederverwendet — Clipping macht diese wiederholten Durchgänge sicher |

---

## Was kommt als Nächstes?

Bisher hatten alle unsere Umgebungen **diskrete** Aktionen (nach LINKS oder RECHTS schieben). Echte Roboter müssen **kontinuierliche** Aktionen steuern — wie „übe exakt 0,73 Newton Kraft aus“.

`ppo_continuous.py` erweitert PPO auf kontinuierliche Aktionen unter Verwendung einer **Gauß-Policy** und testet dies in der viel schwierigeren BipedalWalker-v3-Umgebung!
