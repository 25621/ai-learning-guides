# Dyna-Q: Schneller lernen durch Vorstellungskraft 🧠

## Was ist das?

Stellen Sie sich ein Kind namens Mia vor, das lernt, sich in seiner neuen Schule zurechtzufinden. Jeden Tag läuft sie durch die Flure und entdeckt neue Dinge: „Die Bibliothek liegt hinter der Cafeteria“, „Herr Schmidts Zimmer ist oben in der Nähe des Treppenhauses.“

Ein Schüler, der nach dem **einfachen Q-Learning** lernt, lernt nur aus dem, was er *heute* tut. Wenn er heute nur von der Klasse zur Cafeteria gelaufen ist, aktualisiert er sein Gedächtnis nur über diesen einen Weg.

Ein **Dyna-Q**-Schüler ist anders. Nach jedem echten Gang setzt er sich für eine Minute hin und **spielt in seinem Kopf** mehrere vergangene Gänge durch, an die er sich erinnert. Jedes Durchspielen in Gedanken stärkt seine mentale Landkarte. Nach ein paar Wochen kennt er die Schule in- und auswendig – nicht weil er mehr gelaufen ist, sondern weil er **mehr über das nachgedacht hat, was er bereits gesehen hat**.

Genau das macht Dyna-Q für einen RL-Agenten: Er lernt aus realer Erfahrung **und** aus eingebildeter Erfahrung, die er aus einem Modell zieht, das er nebenbei aufbaut.

---

## Die drei Zutaten

Dyna-Q ist „Q-Learning + Modell + Planung“. Ein echter Schritt erledigt **drei** Aufgaben:

1. **Direktes RL** — das übliche Q-Learning-Update aus `(s, a, r, s')`.
2. **Modell-Lernen** — aufschreiben: „Wenn ich *a* in *s* gemacht habe, bekam ich *r* und landete in *s'*.“
3. **Planung** — ziehe *n* zufällige `(s, a)`-Paare aus dem Gedächtnis des Modells und führe *n* weitere Q-Learning-Updates durch, wobei du **so tust**, als wären diese Schritte gerade passiert.

Dieser dritte Schritt ist die Magie. Mit `n = 50` bewirkt jeder reale Schritt in der Welt **51 Updates** der Q-Tabelle. Der Agent lernt – gemessen an realen Schritten – etwa 50-mal schneller als ein reiner Q-Learner.

---

## Ein Bild des Kreislaufs

```
                   ┌────────────────────────────────────┐
                   │                                    │
   reale Welt  ──► Aktion a ausführen ──► (r, s') beobachten
                            │                           │
              ┌─────────────┼──────────────┐            │
              ▼             ▼              ▼            │
        Q-Learning      Modell[s,a] ← (r,s')   Planung ─┘
          Update                            (n gedachte Updates)
```

Das Modell ist nur eine Nachschlagetabelle:
`(Zustand, Aktion) → (Belohnung, nächster Zustand)`. Billig zu bauen, billig abzufragen.

---

## Beispiele aus dem echten Leben

- **Schachstudium.** Großmeister verbringen Stunden damit, ihre eigenen Partien und Meisterpartien im Kopf nachzuspielen. Jedes Nachspielen ist „Planung“ – zusätzliches Lernen aus Erfahrungen, die bereits stattgefunden haben.
- **Ein Musiker übt Tonleitern.** Nachdem er einen schwierigen Takt einmal gespielt hat, geht er ihn im Geist noch zehnmal durch, bevor er weitermacht. Die Finger bewegen sich nicht, aber das Gehirn aktualisiert sich.
- **Ein selbstfahrendes Auto.** Während es an einer roten Ampel wartet, spielt es die letzten hundert Spurwechsel in der Simulation durch, um seine Strategie zu verfeinern, ohne Reifen zu verbrauchen.

---

## Was unser Code tut

Wir verwenden das klassische **Dyna-Maze** ([Sutton & Barto, Abbildung 8.2](http://incompleteideas.net/book/the-book.html)): ein 6×9-Gitter mit einigen Mauern, einem Startpunkt `S` in der mittleren linken Hälfte und einem Ziel `G` oben rechts.

Wir lassen drei Varianten laufen, jeweils gemittelt über 30 Zufallssaatwerte (Seeds):

| Einstellung | Planungsschritte pro realem Schritt | Bedeutung |
|---------|------------------------------|---------|
| `n = 0` | 0 | einfaches Q-Learning |
| `n = 5` | 5 | ein wenig Übung in der Vorstellung |
| `n = 50` | 50 | viel Übung in der Vorstellung |

Das Skript berichtet die **durchschnittliche Anzahl realer Schritte pro Episode**, während das Training fortschreitet. Weniger Schritte bedeuten, dass der Agent einen direkteren Weg zum Ziel gelernt hat.

### Was Sie beim Ausführen sehen sollten

Der kürzeste Weg in diesem Labyrinth beträgt etwa 9 Schritte; mit ε-greedy Exploration erreicht ein gut trainierter Agent im Durchschnitt etwa 10 Schritte pro Episode. Führen Sie 50 Episoden aus, und alle drei Einstellungen konvergieren dort – der Unterschied liegt darin, **wie schnell** das geschieht:

| Einstellung | Schritte pro Episode (letzte 10 Episoden) | Bedeutung |
|---------|--------------------------------:|---------------|
| `n = 0`   | ~10 | Konvergiert – aber es brauchte ~30–50 Episoden des Herumwanderns, um hierher zu kommen |
| `n = 5`   | ~10 | Konvergiert innerhalb von ~10 Episoden |
| `n = 50`  | ~10 | Konvergiert innerhalb von ~3–5 Episoden |

Das interessante Signal ist die *Lernkurve*, nicht die endgültige Zahl. Das in `outputs/dyna_q.png` gespeicherte Diagramm zeigt drei Kurven, die mit sehr unterschiedlichen Geschwindigkeiten gegen Null tauchen: `n = 50` erreicht den Boden in einer Handvoll Episoden, während `n = 0` noch weit im Lauf nach unten klettert. (In einem winzigen deterministischen Labyrinth wie diesem kommt einfaches Q-Learning schließlich auch ans Ziel – Dyna-Q braucht nur weitaus weniger reale Episoden, was genau der Punkt in Umgebungen ist, in denen reale Schritte kostspielig sind.)

---

## Warum es in diesem Labyrinth so gut funktioniert

Zwei Gründe:

1. **Die Umgebung ist deterministisch.** Jedes `(s, a)` führt immer zum gleichen `(r, s')`, sodass das Modell nach einem einzigen Besuch exakt ist. Eingebildete Erfahrung ist so gut wie echte Erfahrung.
2. **Reale Schritte sind teuer, gedachte Schritte sind kostenlos.** Jedes gedachte Update ist nur ein kurzes Nachschlagen in einer Tabelle, während ein realer Schritt erfordert, dass der Agent sich bewegt. Wenn reale Interaktionen kostspielig sind (denken Sie an einen echten Roboter oder ein echtes Spiel), ist Dyna-Q enorm effizient in Bezug auf die Datenmenge (Sample Efficiency).

---

## Wo Dyna-Q an Grenzen stößt

- **Stochastische Umgebungen.** Wenn `(s, a)` zu vielen verschiedenen `s'`-Werten führen kann, lügt Sie ein Modell nach dem Motto „erinnere dich an das letzte Ergebnis“ an. Lösung: Besuchszahlen speichern oder ein probabilistisches Modell trainieren.
- **Sich ändernde Umgebungen.** Wenn sich die Welt ändert – zum Beispiel eine Tür, die offen war, plötzlich schließt oder eine Abkürzung erscheint –, veraltet das Modell und liefert falsche Vorhersagen. **Dyna-Q+** behebt dies durch einen *Explorationsbonus*: Zustände, die lange nicht besucht wurden, erhalten eine kleine zusätzliche Belohnung, die den Agenten dazu bringt, nachzusehen, ob sich die Welt verändert hat.
- **Große Zustandsräume.** Ein Wörterbuch, das nach `(s, a)` indexiert ist, lässt sich nicht auf Millionen von Zuständen oder auf kontinuierliche Zustände skalieren. Genau diese Lücke füllen **gelernte (neuronale) Weltmodelle** – siehe als Nächstes `world_model.py`.

---

## Wichtige Begriffe

| Begriff | Bedeutung |
|------|---------|
| **Modell**       | Gedächtnis von `(Zustand, Aktion) → (Belohnung, nächster Zustand)` |
| **Planungsschritt** | Ausführen eines Q-Updates mit gedachten Daten aus dem Modell |
| **Direktes RL**   | Ein Q-Update mit realen Daten |
| **Sample Efficiency** | Misst, wie effektiv ein KI-Modell Trainingsdaten nutzt, um ein bestimmtes Leistungsniveau zu erreichen |
| **Dyna**        | Suttons Architektur, die Lernen und Planung verzahnt |

---

## Zusammenfassung in einem Satz

> **Dyna-Q lernt aus dem Handeln UND aus der Vorstellung – und Vorstellen ist kostenlos.**

Diese Idee, in ihrer modernen neuronalen Form, treibt einige der stärksten RL-Agenten an, die jemals gebaut wurden (MuZero, Dreamer, World Models).
