# Matrix-Spiele: Die einfachste Multiagenten-Welt 🎲

## Was ist ein Matrix-Spiel?

Stellen Sie sich vor, Sie und ein Freund wählen *gleichzeitig* ein Handzeichen – **Schere, Stein oder Papier**. Sie sehen die Wahl des anderen nicht. Der Gewinner wird durch eine kleine Tabelle ermittelt:

|        | Stein | Papier | Schere |
|--------|:----:|:-----:|:--------:|
| Stein    |  0,0  | -1,+1 | +1,-1 |
| Papier   | +1,-1 |  0,0  | -1,+1 |
| Schere   | -1,+1 | +1,-1 |  0,0  |

Diese Tabelle ist die *ganze Welt* des Spiels. Keine Bewegung, keine Zeit, keine Karte. Nur eine einmalige Entscheidung. Wir nennen dies ein **Matrix-Spiel**, weil die Auszahlungsmatrix die gesamte Umgebung darstellt.

Matrix-Spiele sind der reinste Ort, um **Multiagenten-RL** (MARL) zu untersuchen, da sich während des Trainings nur die *Policy* jedes Spielers ändern kann – also die Wahrscheinlichkeit, mit der jede Aktion gewählt wird.

---

## Warum es „Multiagenten“ ist

Im Single-Agent-RL ist die Umgebung fixiert: Der Wind weht immer aus der gleichen Richtung, der Boden bewegt sich nie. Der Agent verbessert sich und gewinnt schließlich.

In einem Matrix-Spiel ist Ihre „Umgebung“ ein *anderer lernender Agent*. Wenn dieser schlauer wird, ändert sich das, was für Sie als guter Spielzug gilt. Dies wird als **Nicht-Stationarität** bezeichnet und ist das zentrale Problem des Multiagenten-RL.

> Wenn Sie immer Stein spielen, wird Ihr Gegner irgendwann anfangen, immer Papier zu spielen. Also wechseln Sie zu Schere. Daraufhin wechselt er zu Stein. Dann wechseln Sie zu Papier... und so weiter. Der „beste Zug“ bleibt nie an einer Stelle.

Die klassische Lösung sind **gemischte Strategien**: Wählen Sie keine Aktion deterministisch aus – mischen Sie Ihre Züge so, dass der Gegner sie nicht ausnutzen kann.

---

## Die drei Spiele, die wir spielen

### 1) Schere, Stein, Papier (Nullsummenspiel)
- Der Gewinn des einen Spielers ist der Verlust des anderen.
- Das **Nash-Gleichgewicht** lautet: Jeder Spieler wählt jede Aktion mit einer Wahrscheinlichkeit von ⅓. Jede Abweichung davon kann vom Gegner ausgenutzt werden.
- Wir erwarten, dass unsere zwei Q-Learner um ⅓-⅓-⅓ schwanken – nie perfekt stabil, denn jedes Mal, wenn einer abweicht, reagiert der andere.

### 2) Gefangenendilemma (Allgemeines Summenspiel)
Zwei Verdächtige werden getrennt verhört:

|           | Kooperieren | Verraten |
|-----------|:---------:|:------:|
| Kooperieren |   3, 3    |  0, 5  |
| Verraten   |   5, 0    |  1, 1  |

- „Verraten“ schlägt „Kooperieren“, egal was der andere tut – es ist eine **dominante Strategie**.
- Beide Spieler sind rational → beide verraten → beide erhalten 1, obwohl (Kooperieren, Kooperieren) für beide 3 gewesen wäre. Eigennütziges Handeln zerstört das Gemeinwohl.
- Wir erwarten, dass Q-Learning sauber gegen (Verraten, Verraten) konvergiert.

### 3) Hirschjagd (Koordination)
Zwei Jäger können zusammen einen Hirsch erlegen (große Beute) oder sich jeweils mit einem Hasen begnügen (kleine, aber sichere Beute):

|       | Hirsch | Hase |
|-------|:----:|:----:|
| Hirsch | 4, 4 | 0, 3 |
| Hase   | 3, 0 | 2, 2 |

- (Hirsch, Hirsch) ist **auszahlungsdominant** – das Beste für beide.
- (Hase, Hase) ist **risikodominant** – sicher, wenn man dem Partner nicht traut.
- Das Ergebnis hängt von den Anfangsbedingungen ab: Unabhängige Q-Learner landen oft im *schlechteren* (Hase, Hase) Gleichgewicht, weil Hasen „sicherer“ zu lernen sind.

---

## Beispiele aus dem echten Leben

- **Preisgestaltung im Duopol.** Zwei Cafés in derselben Straße wählen jeden Morgen einen Preis. Die Form der Auszahlungsmatrix entscheidet darüber, ob sie bei einem hohen „kooperativen“ Preis (gut für sie, schlecht für Kunden) oder einem niedrigen Kampfpreis landen.
- **Netzwerkprotokolle.** Router und Absender wählen Strategien für das Timing; das Ergebnis bei Netzwerküberlastung wird durch die Matrix-Spiel-ähnliche Auszahlung zwischen „Durchkommen“ und „Zurückweichen“ bestimmt.
- **Gebote in einer Auktion.** Jeder Bieter gibt ein Gebot ab, ohne die anderen zu kennen; die Auszahlungen hängen vom gesamten Vektor aller Gebote ab. Das Nash-Gleichgewicht ist hier eine *Bieterstrategie*, keine einzelne Zahl.

---

## Was unser Code tut

Für jedes Spiel gehen wir wie folgt vor:
1. Wir erstellen zwei zustandslose Q-Learner (Q ist nur eine Zahl pro Aktion – in einem Ein-Schuss-Spiel gibt es keine Zustände).
2. Wir lassen eine Schleife über 20.000 Schritte laufen. In jedem Schritt wählen beide Agenten gleichzeitig eine ε-greedy Aktion, erhalten eine Belohnung und aktualisieren ihre Q-Werte.
3. Wir verfolgen die **empirische Aktionshäufigkeit** jedes Agenten in einem gleitenden 500-Schritte-Fenster. Anstatt nur abstrakte Wahrscheinlichkeiten zu betrachten, zählen wir, welche Aktionen sie tatsächlich in letzter Zeit gewählt haben (z. B. „in den letzten 500 Runden haben sie zu 40 % Stein gespielt“). Dies gibt uns ein praktisches Echtzeitbild ihrer sich ändernden Strategie.
4. Wir plottet die Häufigkeiten über die Zeit, speichern dies unter `outputs/<spiel>.png` und geben die finalen Q-Werte aus.

### Was Sie sehen sollten

| Spiel | Erwartetes Ergebnis im Diagramm |
|------|------------------------------|
| **Schere, Stein, Papier** | Beide Spieler pendeln um ⅓-⅓-⅓, aber mit sichtbarem Zittern. Die Kurven jagen einander – ein klassisches zyklisches Verhalten. |
| **Gefangenendilemma** | Die Häufigkeit von „Verraten“ steigt bei beiden Spielern schnell auf ~1,0. „Kooperieren“ wird verdrängt. |
| **Hirschjagd** | Die meisten Durchläufe landen bei (Hase, Hase). Einige glückliche Durchläufe erreichen (Hirsch, Hirsch) – versuchen Sie, den Seed im Skript zu ändern, und beobachten Sie den Umschwung. |

---

## Wo unabhängiges Lernen scheitert

Unsere Agenten sind *unabhängig* – sie sehen nur ihre eigene Belohnung, niemals die Aktion oder die Q-Werte des Gegners. Dies ist die einfachste Baseline und hat ihre Grenzen:

- Sie kann **keine Konvergenz** in allgemeinen Summenspielen garantieren.
- Sie kann in **schlechten Gleichgewichten** stecken bleiben (Hirschjagd).
- Sie kann den **Gegner nicht modellieren**.

Echte Multiagenten-Algorithmen beheben dies, indem sie explizit über den anderen Lerner nachdenken. Hier ist eine Übersicht, was die gängigsten Ansätze tun:

| Algorithmus | Kernidee | Analogie aus dem echten Leben |
|-----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|
| **Fictitious Play** | Führe eine Strichliste darüber, wie oft der Gegner jede Aktion gewählt hat. Nimm an, dass er morgen das tun wird, was er bisher immer getan hat – und wähle darauf basierend deine beste Antwort. | Die Gewohnheiten eines Gegners über viele Schachpartien beobachten und die Eröffnung entsprechend anpassen. |
| **CFR (Counterfactual Regret Minimization)** | Frage dich nach jeder Runde: „Wie sehr habe ich es bereut, nicht die anderen Aktionen gewählt zu haben?“ Verschiebe die Wahrscheinlichkeit schrittweise zu den Aktionen, die du bereut hast. Wird beim Poker verwendet, da es mit **unvollständiger Information** umgehen kann. | Nach einer Pokerhand das Spiel im Kopf durchgehen: „Ich hätte mehr setzen sollen – das mache ich beim nächsten Mal.“ |
| **LOLA (Learning with Opponent-Learning Awareness)** | Dein Gradientenschritt berücksichtigt die Tatsache, dass der Gegner *ebenfalls* einen Gradientenschritt macht. Du optimierst dein Update, während du das nächste Update des Gegners antizipierst – zwei Schritte voraus statt einem. | Einen Deal aushandeln und dabei denken: „Wenn ich X anbiete, werden sie mit Y kontern, also sollte ich mit Z beginnen.“ |
| **MADDPG (Multi-Agent Deep Deterministic Policy Gradient)** | Der *Critic* (Wertschafter) jedes Agenten wird mit der **globalen Sicht** trainiert: Er sieht die Beobachtungen und Aktionen aller Beteiligten. Der *Actor* (die Policy, die tatsächlich eingesetzt wird) nutzt weiterhin nur lokale Informationen. (CTDE-Muster). | Ein Basketballtrainer, der das ganze Spielfeld im Blick hat (zentralisierter Critic), aber jedem Spieler beibringt, nur auf das zu reagieren, was er selbst sehen kann (dezentralisierter Actor). |

Aber unabhängiges Q-Learning ist der richtige erste Schritt. Man sieht das Problem der Nicht-Stationarität direkt vor Augen, und die Lösungen ergeben danach viel mehr Sinn.

---

## Wichtige Begriffe zum Merken

| Begriff | Bedeutung |
|------|---------|
| **Auszahlungsmatrix** | Die Tabelle, die ein Multiagenten-Ein-Schuss-Spiel definiert |
| **Nash-Gleichgewicht** | Ein Zustand der Policies, bei dem sich kein Agent durch Abweichen verbessern kann |
| **Gemischte Strategie** | Eine Policy, die zwischen mehreren Aktionen zufällig wählt |
| **Nicht-Stationarität** | Die Umgebung (= andere Agenten) ändert sich ständig während des Lernens |
| **Unabhängiger Lerner** | Ein Agent, der die Existenz anderer Lerner ignoriert |
| **Nullsummenspiel** | Der Gewinn eines Agenten ist exakt der Verlust des anderen |
| **Allgemeines Summenspiel** | Beide Agenten können gewinnen, verlieren oder alles dazwischen |

---

## Zusammenfassung in einem Satz

> **In Matrix-Spielen ist die „Umgebung“ ein anderer Lerner – daher bleibt der beste Spielzug nie an derselben Stelle.**

Dies ist die Grundidee hinter jedem Multiagenten-Algorithmus, dem Sie später begegnen werden, von Self-Play über MADDPG bis hin zu MARL mit Kommunikation.
