# Q-Learning Agent für Frozen Lake 🧊

## Was ist das?

Stellen Sie sich einen zugefrorenen Teich mit rutschigem Eis vor. Es gibt ein **Startfeld** und ein **Ziel** sowie einige **Löcher** dazwischen. Wenn Sie in ein Loch fallen, müssen Sie von vorne anfangen!

Das Eis ist glatt, also könnten Sie nach oben oder unten rutschen, selbst wenn Sie versuchen, nach rechts zu gehen. Ein **Q-Learning Agent** ist ein Roboter, der durch immer wieder neue Versuche lernt, wie er vom Start zum Ziel kommt, ohne einzubrechen!

---

## Wofür steht das „Q“ in Q-Learning?

Das **„Q“** steht für **„Quality“** (Qualität) — genauer gesagt für die *Qualität*, eine bestimmte Aktion in einer bestimmten Situation auszuführen.

Betrachten Sie es wie eine Restaurantbewertung: „Wie gut (Qualität) ist es, die Pizza in DIESEM Restaurant zu bestellen?“ Q(s, a) fragt: „Wie gut ist es, die Aktion **a** auszuführen, wenn ich mich im Zustand **s** befinde?“

Ein hoher Q-Wert bedeutet: „Gute Wahl! Diese Aktion führt zu einer hohen Belohnung.“
Ein niedriger Q-Wert bedeutet: „Schlechte Idee! Diese Aktion führt meist zu Problemen.“

**Beispiel aus dem echten Leben:** Stellen Sie sich vor, Sie sind ein Kind und entscheiden, ob Sie vor dem Abendessen Süßigkeiten essen. Ihr Q-Wert für „Süßigkeiten jetzt essen“ mag im Moment hoch sein (es schmeckt toll!), aber insgesamt ist er niedrig (Mama schimpft, Ihnen wird später schlecht). Q-Learning lernt, diese zukünftigen Konsequenzen zu berücksichtigen — nicht nur das unmittelbare Gefühl!

---

## Die Kernidee: Eine magische Tabelle voller Scores

Q-Learning baut eine große Tabelle auf, die **Q-Tabelle** genannt wird. Jede Zeile steht für ein Quadrat auf dem Eis und jede Spalte für eine Aktion (links, rechts, oben, unten). Die Zahlen darin sind **Scores**: „Wie gut ist es, von diesem Quadrat aus diese Aktion zu wählen?“

Jedes Mal, wenn der Roboter einen Zug versucht:
1. Erhält er Feedback (ist er eingebrochen? hat er das Ziel erreicht?).
2. Aktualisiert er den Score in der Tabelle mit dieser Formel:

> **Neuer Score = Alter Score + Lernrate × (Was wirklich passiert ist − Was ich erwartet habe)**

Der Roboter fragt im Grunde: „War dieser Zug besser oder schlechter, als ich dachte?“

**Beispiel aus dem echten Leben:** Denken Sie an ein Baby, das laufen lernt. Jedes Mal, wenn es einen Schritt versucht und hinfällt, lernt es: „Dieser Schritt war schlecht.“ Jedes Mal, wenn es Erfolg hat, merkt es sich: „Das hat funktioniert!“ Nach vielen Versuchen findet es heraus, wie man läuft. Q-Learning macht dasselbe, nur mit einer Tabelle!

---

## Was Q-Learning besonders macht: Es ist Off-Policy!

Hier ist etwas Cleveres: Wenn Q-Learning seine Tabelle aktualisiert, geht es *immer davon aus, dass es beim nächsten Mal den perfekten Zug macht*, selbst wenn es während des Trainings manchmal zufällige Züge ausprobiert (Exploration).

Dies macht Q-Learning **Off-Policy**: Die Strategie, die es *lernt* (wähle immer die beste bekannte Aktion), ist getrennt von der Strategie, der es während des Trainings *folgt* (wähle manchmal eine zufällige Aktion, um zu erkunden). Konkret nutzt das Update der Q-Tabelle den *maximalen* Q-Wert des nächsten Zustands — das theoretische Beste —, selbst wenn der tatsächliche nächste Schritt des Roboters zufällig sein wird.

Einfach ausgedrückt: Der Roboter mag zufällig nach links wandern, um zu erkunden, aber sein Lernalgorithmus berechnet so, als würde er als Nächstes die *beste* Aktion wählen. Diese Trennung ermöglicht es dem Q-Learning, zur optimalen Strategie zu konvergieren, egal wie viel es exploriert.

---

## Was unser Code herausgefunden hat

Wir haben über **50.000 Episoden** auf dem rutschigen 4×4 Frozen Lake trainiert:

| Metrik | Ergebnis |
|--------|--------|
| Erfolgsquote bei gieriger Evaluation | **73,1 %** |
| Zielmarke (>70 %) | ✓ **ERREICHT** |

Das Eis ist sehr rutschig, daher kann selbst die beste Strategie nicht in 100 % der Fälle gewinnen! Die gelernte Q-Tabelle zeigt, dass der Agent herausgefunden hat: Gehe nach unten und rechts, während du die Löcher meidest.

---

## Beispiele aus dem echten Leben

- **Selbstfahrende Autos**: Durch Testfahrten lernen, welche Spuren an Kreuzungen zu wählen sind.
- **Empfehlungssysteme**: Lernen, welche Filme vorgeschlagen werden sollen, basierend darauf, ob Nutzer vorherige Vorschläge mochten.
- **Videospiel-KI**: Ein Charakter, der lernt, durch ein Labyrinth zu navigieren, indem er viele Pfade ausprobiert.

---

## Wichtige Begriffe zum Merken

- **Q-Tabelle**: Die Tabelle „Wie gut ist jede Aktion in jedem Zustand“.
- **Q(s, a)**: Der Score für die Ausführung von Aktion a im Zustand s.
- **Belohnung (Reward)**: Was der Agent nach einer Aktion erhält (+1 für Ziel, sonst 0).
- **Off-Policy**: Lernt die optimale Strategie, auch während er zufällig exploriert.
- **ε-greedy** (ε = Epsilon): Meistens die beste bekannte Aktion wählen, manchmal zufällig erkunden.
- **Diskontierungsfaktor γ** (γ = Gamma): Wie viel zukünftige Belohnungen wert sind (lieber Geld jetzt als später).

Die Kernbotschaft: **Q-Learning erstellt einen „Spickzettel“ für jede Situation und verbessert ihn so lange, bis er überall den besten Zug kennt.**
