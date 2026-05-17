# DQN auf Atari Pong 🏓

## Was ist Atari Pong?

Pong ist eines der ältesten Videospiele überhaupt – es ist wie digitales Tischtennis! Zwei Schläger spielen einen Ball hin und her. Man gewinnt einen Punkt, wenn der Gegner den Ball verpasst. Das Spiel endet, wenn jemand 21 Punkte erreicht.

In unserer Version steuert die KI einen Schläger. Der Gegner (Computer) steuert den anderen. Das Spiel beginnt immer mit einem Punktestand von -21 (schlechtestmöglich). Ein guter Agent erreicht 0 oder +21.

---

## Warum ist Pong schwierig für eine KI?

In CartPole konnte der Roboter die Zahlen direkt SEHEN (Stabwinkel, Wagengeschwindigkeit...). In Pong sieht er nur **rohe Pixel** – tausende winzige farbige Punkte auf einem Bildschirm!

```
CartPole-Eingabe: [0.02, -0.14, 0.01, -0.23]   ← 4 Zahlen, einfach!
Pong-Eingabe:     [Pixel-Gitter: 210×160×3]    ← 100.800 Zahlen, VIEL schwerer!
```

Der Roboter muss aus den Pixeln herausfinden:
- Wo ist mein Schläger?
- Wo ist der Ball?
- Bewegt sich der Ball nach links oder rechts?
- Wie schnell?

Menschen machen das automatisch (wir haben ein fantastisches Sehvermögen!). Für eine KI ist das eine riesige Herausforderung.

---

## Bewegung sehen: Frame Stacking 🎬

Ein einzelnes Bild (Screenshot) sagt einem nicht, ob sich der Ball nach links oder rechts bewegt. Man muss MEHRERE Bilder sehen, um Bewegung zu verstehen – genau wie ein Daumenkino nur funktioniert, wenn man durch viele Seiten blättert.

**Frame Stacking:** Speise die letzten 4 Bilder gleichzeitig in das Netzwerk ein.

```
Bild 1: Ball auf Position 40
Bild 2: Ball auf Position 43    → Staple diese 4 Bilder → Netzwerk sieht BEWEGUNG!
Bild 3: Ball auf Position 46
Bild 4: Ball auf Position 49
```

Das Netzwerk kann nun folgern: „Der Ball bewegt sich mit Geschwindigkeit 3 nach rechts.“

**Beispiel aus dem echten Leben:** Einen Film schauen vs. ein einzelnes Standbild betrachten. Ein Standbild eines Autorennens ist nur ein verschwommenes Bild. Schau dir 4 Bilder an, und du kannst sagen, welches Auto schneller ist!

---

## Sehen mit einem CNN 🔍

Für Pixeleingaben verwenden wir ein spezielles neuronales Netzwerk namens **Convolutional Neural Network (CNN)**. Anstatt alle Pixel auf einmal zu betrachten, verwendet ein CNN Schiebefenster (sliding windows), um Muster zu erkennen – wie Augen, die ein Bild scannen.

```
Rohe Pixel (84×84×4 Bilder)
       ↓
Conv-Schicht 1 (8×8 Filter, Stride 4) → findet Kanten und Formen
       ↓
Conv-Schicht 2 (4×4 Filter, Stride 2) → findet Objekte (Schläger, Ball)
       ↓
Conv-Schicht 3 (3×3 Filter, Stride 1) → findet Beziehungen
       ↓
Flatten → 512 Neuronen → Q-Werte (einer pro Aktion)
```

**Beispiel aus dem echten Leben:** Wenn Sie einen Freund in einer Menge suchen, bemerkt Ihr Gehirn zuerst Formen (eine Person), dann Merkmale (Haarfarbe) und dann Details (sein Gesicht). CNNs arbeiten auf die gleiche Weise – von einfachen Mustern zu komplexen!

---

## Vorverarbeitung (Preprocessing): Die Welt schrumpfen

Pong-Bilder sind 210×160 Pixel in Farbe. Das ist zu groß! Wir verarbeiten jedes Bild vor:

1. **Graustufen** — Farbe spielt bei Pong keine Rolle (der Ball ist sowieso immer weiß).
2. **Größe auf 84×84 ändern** — kleiner = schnelleres Training, aber immer noch klar genug zum Sehen.
3. **Normalisieren auf [0,1]** — Pixelwerte durch 255 teilen, damit es kleine Zahlen sind.

**Beispiel aus dem echten Leben:** Wie das Erstellen einer Fotokopie in 50 % der Originalgröße. Die wichtigen Details (Ball, Schläger) sind immer noch sichtbar, nur kleiner. Dem Kopierer sind die Farben auch egal!

---

## Reward Clipping: Alle Spiele gleich behandeln ✂️

In Pong erhält man +1 für einen Treffer und -1, wenn der Gegner trifft. In manchen anderen Atari-Spielen können die Punktzahlen in die Tausende gehen!

Wir **begrenzen die Belohnungen (Reward Clipping)** auf [-1, +1], damit sich das Netzwerk nicht um die Skalierung der Belohnungen kümmern muss. Derselbe Code kann so auf JEDEM Atari-Spiel trainieren, ohne die Belohnungsskalen anpassen zu müssen.

---

## Wie lange dauert das Training?

| Trainingsdauer | Was der Agent lernt |
|---|---|
| 100K Schritte | Meist zufällig, reagiert kaum |
| 1M Schritte | Beginnt manchmal, sich zum Ball zu bewegen |
| 5M Schritte | Erwidert einige Schläge |
| 10M Schritte | Kompetitives Spiel, gewinnt vielleicht einige Runden |
| 20M+ Schritte | Schlägt oft den Computergegner |

Unsere Demo läuft über **300K Schritte** — genug, um zu sehen, dass die Trainingsarchitektur funktioniert und erste Lernerfolge zu beobachten, aber nicht genug, um das Spiel zu meistern.

**Beispiel aus dem echten Leben:** Klavierspielen zu lernen dauert Monate. Eine 10-minütige Übungseinheit zeigt, dass man es richtig macht, aber erwarten Sie noch keine Konzerte!

---

## Was unser Code herausgefunden hat

Nach 300K Schritten bei Pong:
- Der Agent beginnt mit Punktzahlen um -20 (bringt den Ball kaum zurück).
- Am Ende verbessert er sich typischerweise auf etwa -15 bis -10.
- Die Lernkurve zeigt eine allmähliche Verbesserung gegenüber dem zufälligen Spiel.

Um eine wirklich kompetitive Pong-Leistung zu sehen, müsste man das Training für etwa 10M+ Schritte auf einer GPU laufen lassen. Die Implementierung ist vollständig und korrekt – sie braucht nur mehr Zeit!

---

## Wichtige Begriffe

| Begriff | Bedeutung |
|------|---------|
| **CNN** | Convolutional Neural Network — spezialisiert auf Bildeingaben |
| **Frame Stacking** | Einspeisen mehrerer aufeinanderfolgender Bilder, um Bewegung zu erfassen |
| **Preprocessing** | Umwandlung roher Bilder (Graustufen, Größe, Normalisierung) vor dem Netzwerk |
| **Reward Clipping** | Begrenzung der Belohnungen auf [-1, +1] zur Einheitlichkeit über Spiele hinweg |
| **ALE** | Arcade Learning Environment — die Bibliothek, die Atari-Spiele ausführt |

---

## Die historische Errungenschaft

Als DeepMind 2015 DQN veröffentlichte, war die Welt erstaunt. Ein EINZIGER Algorithmus mit der GLEICHEN Architektur lernte 49 verschiedene Atari-Spiele zu spielen – viele auf übermenschlichem Niveau – allein aus rohen Pixeln und dem Punktestand!

Vor DQN dachte man, man müsse die Strategie für jedes Spiel manuell programmieren. DQN zeigte, dass ein allgemeiner Lerner alles selbst herausfinden kann. Es war ein historischer Moment für die KI!
