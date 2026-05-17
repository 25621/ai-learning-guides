# Vergleich von Explorationsstrategien 🔦

## Das Problem in einem Satz

Ein RL-Agent muss zwei Dinge tun, die in entgegengesetzte Richtungen ziehen:

- **Exploit (Ausnutzen)**: Das tun, was bisher am besten funktioniert hat.
- **Explore (Erforschen)**: Etwas Neues ausprobieren, falls es noch besser ist.

Wenn man sich zu sehr auf Exploit konzentriert, gibt man sich ewig mit einer mittelmäßigen Routine zufrieden. Wenn man sich zu sehr auf Explore konzentriert, schöpft man nie den Gewinn ab. *Wie* man exploriert – nicht nur *ob* – unterscheidet einen Agenten, der Montezuma's Revenge löst, von einem, der null Punkte erzielt.

Dieses Skript stellt **fünf** Explorationsstrategien bei den gleichen zwei schwierigen Aufgaben gegenüber, damit Sie deren „Persönlichkeiten“ sehen können.

## Analogie aus dem echten Leben: Auswahl eines Mittagsrestaurants

Sie sind gerade in eine neue Stadt mit 200 Restaurants gezogen.

- **ε-greedy** = „Geh in mein aktuelles Lieblingsrestaurant, aber einmal alle zehn Tage würfle ich und wähle ein *völlig zufälliges* Restaurant.“ Sie werden viel ausprobieren, aber *ziellos* – und Sie werden immer wieder Orte besuchen, die Sie bereits hassen.
- **Optimistische Initialisierung** = „Gehe davon aus, dass *jedes* Restaurant, das ich noch nicht probiert habe, das beste der Stadt ist, bis das Gegenteil bewiesen ist.“ Sie werden methodisch alle 200 durcharbeiten und jedes von der Liste streichen, wenn die Realität Sie enttäuscht – und Sie werden die wirklich guten schnell finden.
- **UCB (Upper Confidence Bound)** = „Bevorzuge mein Lieblingsrestaurant, aber gib einen *Bonus* für Orte, die ich kaum ausprobiert habe – je weniger ich darüber weiß, desto größer der Bonus.“ Dies ist clever bei der Auswahl, *welchen* unbekannten Ort man heute ausprobiert, aber jede Entscheidung ist lokal: Es wählt die aktuell am besten aussehende Option, ohne eine Route durch ganze unerforschte Viertel zu planen. Es wird nicht denken: „Ich sollte die Stadt in den Osten durchqueren, weil dort zwanzig unversuchte Orte beieinander liegen“ – jedes Restaurant wird isoliert, Schritt für Schritt, bewertet.
- **Zählbasierter Belohnungsbonus** = Wie UCB, aber man *genießt auch die Neuheit selbst* – eine Mahlzeit an einem brandneuen Ort ist intrinsisch befriedigend, und diese Zufriedenheit prägt Ihren langfristigen Plan, in welche Viertel Sie wandern möchten.
- **Vorhersagefehler-Belohnungsbonus** = „Ich bekomme einen Kick von einer Mahlzeit, die mich *überrascht* hat – etwas, das ich nicht hätte vorhersagen können.“ Ein neuer Ort, der sich genau wie erwartet herausstellt? Meh. Einer, der völlig anders ist als Ihr mentales Modell? Faszinierend, und Sie aktualisieren Ihren Plan, um mehr davon zu suchen.

## Die fünf Strategien (alle in `compare_exploration.py`)

### 1. ε-greedy – der Standard, und es ist *Zaudern*, nicht Explorieren

Handeln Sie gierig (greedy), aber wählen Sie mit der Wahrscheinlichkeit ε eine gleichmäßig zufällige Aktion. Es ist der Standard-Baseline-Wert in DQN und ähnlichen Verfahren. Sein fataler Fehler bei schwierigen Aufgaben: **Jeder Schritt ist ein unabhängiger Münzwurf.** Um eine Kette von `N` korrekten Zügen herabzusteigen, muss die Münze `N`-mal hintereinander richtig landen – das ist exponentiell unwahrscheinlich. ε-greedy ist ein *Zappeln*, kein *Explorieren*.

### 2. Optimistische Initialisierung – „unschuldig, bis Langeweile bewiesen ist“

Starten Sie *jeden* Q-Wert mit dem größten überhaupt möglichen Return, `R_max / (1 − γ)`. Nun sieht eine Aktion, die der Agent noch nie ausprobiert hat, wie das Beste auf der Welt aus, sodass die **gierige** Policy gezwungen ist, sie auszuprobieren; erst nach dem Besuch sinkt der Wert in Richtung Wahrheit. Der Optimismus über *un*erforschte Regionen **pflanzt sich automatisch durch die Value-Funktion fort** (über das Bootstrap-Verfahren von Q-Learning), sodass der Agent Schritt für Schritt in die Teile der Welt gezogen wird, die er noch nicht gesehen hat. Fast kostenlos, keine zusätzliche Buchführung – und, wie Sie sehen werden, der stärkste *tiefe* Explorer in einer kleinen tabellarischen Welt.

### 3. UCB-artige Aktionsauswahl – Bonus in der *Wahl*, nicht in der *Belohnung*

Wählen Sie `argmax_a [ Q(s,a) + c·√(ln t / N(s,a)) ]`: Bevorzugen Sie Aktionen mit hohem Wert, aber blähen Sie diejenigen auf, die Sie selten ausprobiert haben. Bekannt aus Multi-Armed Bandits. Der Haken: Der Bonus existiert nur in der **Aktionsauswahlregel**, niemals in der Belohnung – daher fließt er *nicht* durch die Value-Funktion. UCB ist großartig darin, sicherzustellen, dass man jede Aktion in *diesem* Zustand ausprobiert hat, aber schwach darin, eine Route zu einer weit entfernten unerforschten Region zu planen.

### 4. Zählbasierter **Belohnungsbonus** – Neugier, die klassische Version

Fügen Sie der **Belohnung** `1/√(N(s,a))` hinzu (mit einem Gewicht `beta`, das abnimmt). Da es in der Belohnung enthalten ist, propagiert Q-Learning es: Zustände, die zu neuartigen Regionen führen, werden wertvoll. Dies ist die MBIE-EB / klassische „Explorationsbonus“-Idee.

### 5. Vorhersagefehler-**Belohnungsbonus** – Neugier, die ICM/RND-Version

Fügen Sie der Belohnung `−log P(s'|s,a)` aus einem winzigen gelernten Forward-Modell hinzu (wieder mit abnehmendem `beta`). Das schärfste Neuheitssignal der fünf: In einer deterministischen Welt sinkt die Überraschung eines Übergangs auf ~0, sobald man ihn einmal gesehen hat, anstatt langsam wie `1/√N` zu verblassen. Der tabellarische Cousin von ICM / RND.

## Die zwei Testaufgaben

- **Aufgabe A – MiniMontezuma**: Das Gridworld-Szenario Schlüssel→Tür→Schatz, Belohnung nur beim Schatz (~15 perfekte Züge entfernt). Testet: „Kann man überhaupt eine lange Kette mit spärlicher Belohnung überleben?“
- **Aufgabe B – DeepSea(N)**: Die klassische Deep-Exploration-Kette, ausgeführt mit den Längen `N = 5, 8, 11, 14`. Die Belohnung verbirgt sich hinter `N` korrekten Zügen, die jeweils winzige unmittelbare Kosten verursachen – ein kurzsichtiger (myopic) Agent lernt also, die Kosten zu vermeiden und findet den Preis nie. Testet: „Funktioniert Ihre Strategie noch, wenn die Kette *länger* wird?“

## Was tatsächlich passiert (ausführen und sehen)

**Aufgabe A – MiniMontezuma:**

| Strategie | Erster Schatz | Finale Lösungsrate |
|----------|---------------:|-----------------:|
| ε-greedy | nie | 0.00 |
| Optimistische Init | ~Episode 1 | 1.00 |
| UCB Aktionsauswahl | ~Episode 3 | ~0.95 |
| Zählbasierter Belohnungsbonus | ~Episode 82 | ~0.41 |
| Vorhersagefehler-Belohnungsbonus | ~Episode 23 | 1.00 |

**Aufgabe B – DeepSea, Anteil der Durchläufe, die die Belohnung fanden:**

| Strategie | N=5 | N=8 | N=11 | N=14 |
|----------|----:|----:|-----:|-----:|
| ε-greedy | 0 | 0 | 0 | 0 |
| Optimistische Init | 1.0 | 1.0 | 1.0 | 1.0 |
| UCB Aktionsauswahl | 1.0 | 1.0 | 0.0 | 0.0 |
| Zählbasierter Belohnungsbonus | 1.0 | 1.0 | ~0.1 | 0.0 |
| Vorhersagefehler-Belohnungsbonus | ~0.9 | ~0.8 | ~0.9 | ~0.2 |

*(Die Zahlen schwanken etwas je nach Zufallssaat, aber die Tendenz ist eindeutig.)*

## Die Lektionen

1. **ε-greedy ist keine Exploration.** Es löst *keine* der schwierigen Aufgaben. Zufälliges Zaudern fädelt einfach keine langen korrekten Sequenzen ein. (Dennoch ist es in vielen Codes der Standard – weil es bei *einfachen* Aufgaben gut genug und extrem simpel ist.)

2. **Echte Exploration bedeutet, optimistisch gegenüber dem Unbekannten zu sein – auf die eine oder andere Weise.** Ob man den Optimismus in die *Initialwerte* (Strategie 2), in die *Aktionswahl* (Strategie 3) oder in eine *selbst generierte Belohnung* (Strategien 4–5) einbaut, der gemeinsame Nenner ist: *Lasse das Unerforschte attraktiv aussehen* und überlasse es dann dem Lernen, dich dorthin zu bringen.

3. **Auf einem Grid mit spärlicher Belohnung funktionieren alle vier „echten“ Strategien – und der Vorhersagefehler-Bonus ist am schnellsten am Ziel**, weil er das klarste „Das ist neu“-Signal liefert.

4. **Auf einer *tiefen* Kette, bei der der Optimismus einen weiten Weg zurücklegen muss, ist die optimistische Initialisierung der einfache Champion.** Sie propagiert den Optimismus kostenlos durch die Value-Funktion. UCB bricht zuerst zusammen (sein Bonus geht nie in die Value-Funktion ein, daher kann es nicht *tief* planen). Belohnungsboni schneiden besser ab – sie propagieren den Wert –, aber einfaches tabellarisches Q-Learning ist zu langsam, um diesen Optimismus eine lange Kette hinunterzuschieben, bevor der Bonus abklingt.

5. **Genau dieser letzte Punkt ist der Grund, warum die Skalierung tiefer Exploration auf Pixel zusätzliche Feuerkraft benötigte** – bootstrapped DQN, RND mit einem echten neuronalen Netz (damit der Optimismus über ähnliche Zustände *generalisiert*, anstatt Zelle für Zelle propagiert zu werden), Go-Explore (buchstäblich vielversprechende Zustände merken und zu ihnen zurückkehren). Die tabellarischen Beispiele hier zeigen Ihnen die *Prinzipien*; die realen Systeme sind diese Prinzipien plus ein Netzwerk, das generalisiert.

## Wichtige Begriffe zum Merken

| Begriff | Bedeutung |
|------|---------|
| **Exploration–Exploitation-Trade-off** | Neue Dinge ausprobieren vs. das Wissen versilbern – das zentrale Spannungsfeld im RL |
| **Dithering (Zaudern/Zappeln)** | „Exploration“ durch Hinzufügen von Zufallsrauschen zu Aktionen (ε-greedy, Gauß-Rauschen) – schwach bei schwierigen Aufgaben |
| **Optimism in the face of uncertainty** | Das Dachprinzip: Behandle das Unbekannte als großartig, bis du es überprüft hast |
| **Optimistische Initialisierung** | Implementierung dieses Prinzips, indem alle Werte beim maximal möglichen Return gestartet werden |
| **UCB** | Upper Confidence Bound: Wähle `argmax (Wert + Bonus, der mit der Anzahl der Besuche schrumpft)` |
| **Deep Exploration** | Exploration, die eine lange *kohärente* Sequenz „ungewöhnlicher“ Aktionen erfordert, nicht nur eine einzelne |
| **`beta` Annealing** | Verringerung des Neugier-Gewichts im Laufe der Zeit, damit der Agent schließlich aufhört zu explorieren und das Gelernte ausnutzt |

---

## Zusammenfassung in einem Satz

> **ε-greedy ist nur Rauschen; jede echte Explorationsstrategie funktioniert, indem sie das Unerforschte attraktiv aussehen lässt – über optimistische Werte, einen Aktionswahl-Bonus oder eine selbst generierte Neuheitsbelohnung – und die richtige Wahl hängt davon ab, ob die Belohnung lediglich *spärlich* (wie ein einzelner versteckter Preis in einem flachen Feld) oder wirklich *tief* ist (wie ein Zahlenschloss, das eine lange, präzise Sequenz spezifischer Entscheidungen erfordert).**
