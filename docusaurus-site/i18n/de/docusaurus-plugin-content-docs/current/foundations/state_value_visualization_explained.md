# Zustands-Wertfunktionen (State-Value Functions) 🗺️

## Was ist ein "Zustand" (State)?

Stellen Sie sich vor, Sie spielen ein Brettspiel. In jedem Moment stehen Sie auf *einem* Feld des Brettes. Dieses Feld ist Ihr **Zustand** – es ist der Ort, an dem Sie sich gerade befinden.

In unserem 4×4-Gitterspiel gibt es 16 Felder (Zustände). Jedes Feld ist ein Ort, an dem der Agent stehen kann.

---

## Was ist ein "Wert" (Value)?

Hier ist die entscheidende Frage: **"Wenn ich jetzt auf diesem Feld stehe, wie viel Schatz kann ich erwarten zu sammeln, bevor das Spiel endet?"**

Die Antwort darauf ist der **Wert** dieses Zustands!

Ein Feld mit einem **hohen Wert** bedeutet: "Das ist ein großartiger Ort – von hier aus werde ich wahrscheinlich viel Schatz sammeln!"

Ein Feld mit einem **niedrigen Wert** bedeutet: "Oje – von hier aus gehen die Dinge normalerweise schlecht aus."

**Beispiel aus dem echten Leben:** Stellen Sie sich vor, Sie spielen Verstecken. Wenn Sie sich hinter einem großen Baum verstecken (ein tolles Versteck), ist Ihre Chance zu gewinnen hoch – das ist ein Zustand mit hohem Wert! Wenn Sie sich mitten in einem leeren Raum verstecken, werden Sie wahrscheinlich gefunden – das ist ein Zustand mit niedrigem Wert.

---

## Unsere Gitterwelt (Grid World)

Hier ist das Spielfeld, das wir verwendet haben:

```
S  .  .  .      S = Start
.  H  .  H      H = Loch (Hole) (Belohnung -1, Spiel endet)
.  .  .  H      G = Ziel (Goal) (Belohnung +1, Spiel endet)
H  .  .  G      . = Leeres sicheres Feld
```

- Wenn Sie **G** (Ziel) erreichen: Sie erhalten **+1 Punkt** 🎉
- Wenn Sie auf **H** (Loch) treten: Sie erhalten **-1 Punkt** 😢
- Andere Schritte: **0 Punkte**

Wir haben γ (Gamma) = 0,99 verwendet, was bedeutet, dass zukünftige Belohnungen fast so viel zählen wie sofortige Belohnungen. (Eine Süßigkeit morgen ist fast so gut wie eine Süßigkeit heute!)

---

## Zwei verschiedene Pläne (Policies)

Wir haben zwei Policies getestet und den Wert jedes Feldes für beide berechnet:

### Policy 1: Gleichverteilt zufällig (Uniform Random)
Zufällige Auswahl von oben, unten, links oder rechts mit gleicher Wahrscheinlichkeit.

```
Werte (Uniform Random Policy):
-0,912  -0,932  -0,912  -0,942
-0,929   (H)   -0,898   (H)
-0,901  -0,801  -0,696   (H)
 (H)   -0,630  -0,104   (G)
```

Fast überall ist der Wert **negativ** – die zufällige Policy fällt so oft in Löcher, dass es ziemlich schlecht ist, irgendwo zu sein!

---

### Policy 2: Zum Ziel hin orientiert (Biased Toward Goal)
Bevorzugt Bewegungen nach rechts und unten (Richtung Ziel), geht aber manchmal immer noch in andere Richtungen.

```
Werte (Biased-Toward-Goal Policy):
-0,838  -0,895  -0,814  -0,961
-0,798   (H)   -0,665   (H)
-0,595  -0,143  -0,213   (H)
 (H)    0,254   0,673   (G)
```

Jetzt haben die Felder in der Nähe des **Ziels** **positive Werte** (0,254 und 0,673)! Die kluge Policy sorgt dafür, dass diese Felder gute Orte sind.

---

## Was die Farben in unserem Bild bedeuten

In unserer Visualisierung:
- **Grüne Felder** = hoher Wert (tolle Orte zum Sein)
- **Rote Felder** = niedriger Wert (vermeiden Sie diese!)
- **Gelbe Felder** = irgendwo dazwischen

Man kann den **Gradienten** sehen – die Werte werden grüner, wenn man sich auf das Ziel zubewegt, und röter in der Nähe von Löchern.

---

## Warum interessieren uns Werte?

Werte sind das *Fundament* des Reinforcement Learning! Sobald man den Wert jedes Zustands kennt, kann man kluge Entscheidungen treffen:

> "Ich bin bei Feld A. Ich kann zu Feld B gehen (Wert = 0,5) oder zu Feld C (Wert = -0,3).
> Ich gehe zu B – es hat einen höheren Wert!"

Genau so lernen viele RL-Algorithmen (wie Q-Learning), gute Entscheidungen zu treffen, ohne dass ihnen die Regeln erklärt werden.

**Beispiel aus dem echten Leben:** Stellen Sie sich vor, Sie wählen aus, in welcher Schlange Sie sich im Supermarkt anstellen. Jede Schlange ist ein "Zustand". Der Wert dieses Zustands ist, wie schnell Sie durch die Kasse kommen. Sie schauen sich die Schlangen an (beobachten Zustände) und wählen diejenige mit dem höchsten Wert (kürzeste Wartezeit + wenigste Artikel).

---

## Wie wir die Werte berechnet haben

Wir haben die **Iterative Policy Evaluation** verwendet, die so funktioniert:

1. Start: Alle Werte werden auf 0 geschätzt.
2. Update: Für jedes Feld wird berechnet, wie hoch der Wert sein *sollte*, basierend darauf, wohin die Policy Sie als Nächstes führt.
3. Wiederholung, bis sich die Werte nicht mehr ändern (Konvergenz).

Mathematisch: **V(s) = Σ_a π(a|s) × [R(s,a) + γ × V(nächster_Zustand)]**

In einfachem Deutsch: "Der Wert dieses Feldes = die durchschnittliche Belohnung, die ich jetzt bekomme + ein kleiner Teil des Wertes des Ortes, an dem ich lande."

---

## Wichtige Begriffe

- **Zustand (State)**: Wo Sie sich gerade befinden (ein Feld auf dem Brett).
- **Wert V(s)**: Erwartete Gesamtbelohnung, beginnend im Zustand s.
- **Policy**: Ihr Plan, was in jedem Zustand zu tun ist.
- **Diskontierungsfaktor γ**: Wie sehr Ihnen zukünftige Belohnungen wichtig sind (0,99 = sehr wichtig!).
- **Policy Evaluation**: Berechnung der Werte für jeden Zustand unter einer gegebenen Policy.

Die Kernbotschaft: **Einige Orte sind besser als andere – und die Wertfunktion sagt Ihnen genau, wie gut jeder Ort ist!**
