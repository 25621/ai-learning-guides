# Lineares Q-Learning für CartPole 🎪

## Was ist CartPole?

Stellen Sie sich einen Besenstiel vor, den Sie aufrecht auf Ihrer Fingerspitze balancieren. Wenn Sie Ihren Finger nur ein kleines Stück nach links oder rechts bewegen, können Sie verhindern, dass der Besenstiel umfällt. Das ist **CartPole**!

Ein winziger Roboter sitzt auf einem Wagen (einem Kasten auf Rädern), auf dem oben ein Stab (Pole) befestigt ist. Der Roboter kann den Wagen nur nach **links** oder **rechts** schieben. Er muss lernen, diesen Stab so lange wie möglich in der Balance zu halten – genau wie Sie beim Balancieren eines Besenstiels!

Der Roboter kann 4 Dinge über die Welt sehen:
1. Wo sich der Wagen befindet.
2. Wie schnell sich der Wagen bewegt.
3. Wie stark der Stab geneigt ist.
4. Wie schnell sich der Stab neigt.

---

## Das große Problem: Zu viele Zustände!

Erinnern Sie sich an das Q-Learning aus Phase 2? Es verwendete eine große Tabelle, um sich zu merken, wie gut jede Aktion in jeder Situation (Zustand) ist. Das funktionierte hervorragend für Frozen Lake – dort gab es nur 16 Quadrate auf dem Eis.

Aber CartPole ist anders! Der Wagen kann an **jeder beliebigen Position** sein, sich mit **jeder beliebigen Geschwindigkeit** bewegen und der Stab kann **jeden beliebigen Winkel** einnehmen. Es gibt im Grunde **unendlich viele mögliche Zustände**! Wir können keine Tabelle mit unendlich vielen Zeilen erstellen. Wir bräuchten ein Notizbuch von der Größe des Universums!

**Beispiel aus dem echten Leben:** Stellen Sie sich vor, Sie lernen Fahrradfahren. Sie können sich nicht jedes einzelne Wackeln merken – es gibt zu viele! Stattdessen lernen Sie eine **Regel**: „Wenn ich mich nach links neige, lenke nach links; wenn ich mich nach rechts neige, lenke nach rechts.“ Eine einfache Regel funktioniert für ALLE Arten von Wackeln.

---

## Die Lösung: Eine Zauberformel

Die **lineare Funktionsapproximation** ersetzt die riesige Tabelle durch eine **winzige Formel**:

> **Score(Situation, Aktion) = w₁ × Wagenposition + w₂ × Wagengeschwindigkeit + w₃ × Stabwinkel + w₄ × Stabgeschwindigkeit**

- Die `w`-Zahlen werden **Gewichte** genannt – sie sind wie Regler, an denen man drehen kann.
- Wir lernen **verschiedene Gewichte für jede Aktion** (Links-Schieben und Rechts-Schieben).
- Die Formel liefert einen Score (Punktwert), der angibt, wie gut jede Aktion im Moment ist.

**Beispiel aus dem echten Leben:** Denken Sie an ein einfaches Rezept: „1 Tasse Mehl + 2 Eier + ½ Tasse Butter.“ Die Gewichte (1, 2, ½) sagen Ihnen, wie wichtig jede Zutat ist. Wir lernen das Rezept für gute Entscheidungen!

---

## Wie lernt der Roboter?

Der Roboter probiert Dinge aus, erhält Feedback und passt die Gewichte an:

1. **Der Roboter schiebt den Wagen** (er wählt die Aktion mit dem höchsten Score).
2. **Physik passiert** (der Stab neigt sich ein wenig, der Wagen bewegt sich).
3. **Der Roboter erhält eine Belohnung** (+1 für jeden Schritt, den der Stab oben bleibt, 0 wenn er fällt).
4. **Der Roboter fragt:** „War das tatsächliche Ergebnis besser oder schlechter als vorhergesagt?“
5. **Der Roboter passt die Gewichte an**, um beim nächsten Mal näher an der Realität zu sein.

Dies ist das **Semi-Gradient TD Update** – ein schicker Name für „ändere das Rezept ein kleines Stück, basierend auf der Überraschung“.

> **Neues Gewicht = Altes Gewicht + Lernrate × (Tatsächliches Ereignis − Vorhersage) × Merkmal (Feature)**

---

## Was unser Code herausgefunden hat

Wenn Sie `linear_q_cartpole.py` ausführen, macht der Roboter Folgendes:

- Er beginnt schrecklich (der Stab fällt in 10–30 Schritten um).
- Über 3.000 Versuche lernt er allmählich gute Gewichte.
- Schließlich hält er den Stab für 100–400+ Schritte in der Balance!

Das Diagramm zeigt die **Lernkurve** – wie der Score im Laufe der Zeit besser wird. Sie wird zackig sein (Lernen verläuft nie reibungslos!), aber der Trend sollte nach oben zeigen.

---

## Warum das cool (und begrenzt!) ist

**Cool:** Eine winzige Formel mit nur 8 Zahlen (4 Gewichte × 2 Aktionen) kann einen Stab balancieren! Keine riesige Tabelle nötig.

**Begrenzt:** Die Formel ist zu einfach für komplexe Aufgaben. Sie geht davon aus, dass größere Zahlen immer größere Auswirkungen bedeuten (was nicht immer stimmt). Für schwierigere Spiele wie bei Atari benötigen wir **neuronale Netzwerke** – genau das, was DQN macht!

---

## Wichtige Begriffe

| Begriff | Bedeutung |
|------|---------|
| **Merkmal (Feature)** | Eine messbare Eigenschaft der Welt (z. B. Stabwinkel) |
| **Gewicht (Weight)** | Wie stark ein Merkmal die Entscheidung beeinflusst |
| **Linear** | Die Formel besteht nur aus Multiplikation und Addition (keine komplizierten Kurven) |
| **Semi-Gradient** | Aktualisierung der Gewichte, indem der Richtung des geringeren Fehlers gefolgt wird |
| **Funktionsapproximation** | Verwendung einer Formel anstelle einer Tabelle |

---

## Was kommt als Nächstes?

Lineare Approximation ist so, als würde man ein gerades Lineal verwenden, um eine Kurve zu zeichnen – für einfache Formen funktioniert das gut, aber nicht für komplexe. Für Atari-Spiele mit Millionen möglicher Situationen benötigen wir **Deep Q-Networks (DQN)** – neuronale Netzwerke, die viel komplexere Muster lernen können. Das finden Sie in der nächsten Datei!
