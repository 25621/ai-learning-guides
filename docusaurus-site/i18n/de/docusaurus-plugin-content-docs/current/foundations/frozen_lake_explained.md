# Frozen Lake mit einer Zufallsstrategie 🧊

## Was ist Frozen Lake?

Stellen Sie sich vor, Sie spielen mit Ihren Freunden auf einem **zugefrorenen Teich**.

Das Eis ist größtenteils sicher, aber einige Stellen haben **Löcher** – wenn Sie in ein Loch treten, fallen Sie hinein und das Spiel ist vorbei! Am einen Ende des Teichs liegt ein **Geschenk** 🎁. Ihre Aufgabe ist es, vom **Start** zum **Geschenk** zu gleiten, ohne einzubrechen.

So sieht der zugefrorene See aus (4 Quadrate × 4 Quadrate):

```
S  F  F  F
F  H  F  H
F  F  F  H
H  F  F  G
```

- **S** = Start (wo Sie beginnen)
- **F** = Gefrorenes Eis (Safe! ❄️)
- **H** = Loch (Hole - hineinfallen, Spiel vorbei 😨)
- **G** = Ziel (Goal - das Geschenk! 🎁)

---

## Der schwierige Teil: Rutschiges Eis!

Auf einem echten zugefrorenen Teich gleitet man manchmal nach *oben* oder *unten*, wenn man eigentlich versucht, nach *rechts* zu gehen! Das macht es so schwierig.

Selbst wenn Sie nach rechts gehen *wollen*, könnte das Spiel Sie woanders hin gleiten lassen. Das nennt man **Stochastik** – ein schickes Wort dafür, dass die Dinge nicht immer so laufen wie geplant.

---

## Was ist eine Zufallsstrategie (Random Policy)?

Eine **Policy** (Strategie) ist einfach ein Plan: „In dieser Situation werde ich DIESE Aktion ausführen.“

Eine **Zufallsstrategie** bedeutet: „Ich habe überhaupt keinen Plan! Ich wähle jedes Mal einfach eine zufällige Richtung – oben, unten, links oder rechts – als würde ich an einem Glücksrad drehen!“

Es ist wie ein Baby, das auf dem Eis läuft und keine Ahnung hat, wo das Geschenk ist.

---

## Was unser Code herausgefunden hat

Wir haben die Zufallsstrategie über **1.000 Spiele** getestet:

| Ergebnis | Wert |
|--------|-------|
| **Wie oft wurde das Geschenk erreicht?** | 11 von 1.000 (1,1 %) |
| **Durchschnittliche Schritte pro Spiel** | 7,5 Schritte |
| **Schnellstes Spiel** | 2 Schritte |
| **Längstes Spiel** | 33 Schritte |

Meistens fiel der zufällige Wanderer schnell in ein Loch. Nur in 1 von 100 Spielen wurde das Geschenk gefunden!

---

## Warum ist das nützlich?

Obwohl die Zufallsstrategie schrecklich ist, gibt sie uns einen **Baseline-Wert** – einen Ausgangspunkt, mit dem wir andere Ergebnisse vergleichen können.

Wenn wir später eine *intelligente* Strategie bauen (mit Q-Learning oder anderen Algorithmen), können wir sagen: „Unser intelligenter Agent ist in 75 % der Fälle erfolgreich – viel besser als die 1 % des Zufallswanderers!“

**Beispiel aus dem echten Leben:** Stellen Sie sich vor, Sie versuchen, Ihr Klassenzimmer in einer neuen Schule zu finden, indem Sie an jeder Flurkreuzung zufällig links oder rechts abbiegen. Irgendwann kommen Sie vielleicht an, aber es würde sehr lange dauern! Eine intelligente Strategie ist wie eine Landkarte.

---

## Was die Heatmap zeigt

In unserem Bild zeigt die **Heatmap**, welche Quadrate der Zufallswanderer am häufigsten besucht hat:

- Das **Startquadrat** wird sehr oft besucht (jedes Spiel beginnt dort).
- Quadrate in der Nähe der **Löcher** werden seltener besucht (der Wanderer fällt oft hinein, bevor er sie erreicht).
- Das **Ziel** wird sehr selten besucht, weil der Zufallswanderer es fast nie bis dorthin schafft.

Das lehrt uns etwas Wichtiges: Die Zufallsstrategie bleibt in der Nähe des Anfangs stecken und erkundet nie wirklich den ganzen See.

---

## Wichtige Begriffe zum Merken

- **Policy**: Ihr Plan, was Sie in jeder Situation tun.
- **Zufallsstrategie (Random Policy)**: Kein Plan – wählen Sie einfach eine zufällige Aktion!
- **Baseline**: Ein schlechtes Ergebnis, das wir zum Vergleich heranziehen (um wie viel besser können wir werden?).
- **Stochastisch**: Dinge laufen nicht immer wie geplant (wie bei rutschigem Eis!).
- **Erfolgsquote**: Wie oft haben wir gewonnen? (Hier: 1,1 % – sehr niedrig!)

Die Kernbotschaft: **Eine Zufallsstrategie ist ein Ausgangspunkt. Echtes Lernen bedeutet, einen besseren Plan zu entwickeln!**
