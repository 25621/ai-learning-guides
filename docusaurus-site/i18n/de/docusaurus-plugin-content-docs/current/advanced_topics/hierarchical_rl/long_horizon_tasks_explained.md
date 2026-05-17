# Aufgaben mit langem Zeithorizont (Long-Horizon Tasks)

## Die Kernidee: Wenn die Belohnung in weiter Ferne liegt

Stellen Sie sich vor, Sie sind ein Koch und versuchen, ein neues Rezept zu lernen, indem Sie ausschließlich das fertige Gericht probieren. Sie führen 40 Schritte aus – schneiden, anbraten, würzen, köcheln lassen, anrichten – aber Sie erhalten erst ganz am Ende Feedback: „Zu salzig.“ Welcher der 40 Schritte hat das Problem verursacht? Sie haben keine Ahnung.

Dies ist das **Problem des langen Zeithorizonts**: Wenn das Belohnungssignal von den Entscheidungen, die es verursacht haben, durch Dutzende (oder Hunderte) von Schritten getrennt ist, wird das Lernen sehr schwierig.

---

## Warum „flache“ Agenten Schwierigkeiten haben

Ein flacher (flat) RL-Agent (wie die DQN-Agenten aus Phase 3) versucht, den Wert jedes einzelnen Schritts auf einmal zu lernen. Bei kurzen Aufgaben – einen Stab balancieren, einer Wand ausweichen – funktioniert das gut. Die Belohnung erfolgt schnell, und der Agent kann Ursache und Wirkung verknüpfen.

Aber bei einer langen Aufgabe – einen Schlüssel sammeln, damit eine Tür öffnen und dann das Labyrinth verlassen – muss der Agent:

1. Zufällig auf den Schlüssel stoßen (Glück gehabt!).
2. Sich merken, dass das Sammeln von Schlüsseln nützlich ist.
3. Zufällig auf die Tür stoßen (wieder Glück gehabt!).
4. Die gesamte Sequenz mit der einzigen Belohnung am Ausgang verknüpfen.

Bei zufälliger Exploration sinkt die Wahrscheinlichkeit, diese gesamte Sequenz zufällig abzuschließen, mit jedem neuen erforderlichen Schritt exponentiell. Das flache DQN muss im Grunde sehr, sehr oft Glück haben, bevor es eine einzige positive Belohnung sieht, aus der es lernen kann.

---

## Die hierarchische Lösung: Teile und herrsche

Hierarchisches RL zerlegt die lange Aufgabe in eine **Zwei-Ebenen-Struktur**:

| Ebene | Bezeichnung | Aufgabe |
|-------|--------|-----|
| Hoch | **Manager** | Wählt das nächste Unterziel (Subgoal) aus |
| Niedrig | **Worker** (Arbeiter) | Navigiert zu diesem Unterziel |

Genau so bewältigen Menschen komplexe Aufgaben. Sie planen Ihre Autoreise nicht Kurve für Kurve, bevor Sie losfahren. Stattdessen:

- **Manager (Sie, zu Hause):** „Erster Halt: die Tankstelle. Nächster Halt: die Autobahnauffahrt. Dann: Ausfahrt 42.“
- **Worker (Sie, beim Fahren):** Erledigt alle individuellen Lenkentscheidungen, um jeden Halt zu erreichen.

Der Manager denkt in *Checkpoints*. Der Worker denkt in *Lenkbewegungen*.

---

## Warum dies flaches Lernen bei langen Aufgaben schlägt

Der Worker muss nur das *nächste Unterziel* erreichen – eine kurze Aufgabe mit einer klaren, nahen Belohnung. Er erhält schnell Feedback und lernt effizient.

Der Manager muss nur über die *Reihenfolge der Unterziele* entscheiden – ein viel einfacheres Problem, als jeden einzelnen Schritt zu planen.

Zusammen teilen die beiden Ebenen das schwierige Problem des langen Zeithorizonts in zwei einfache Probleme mit kurzem Zeithorizont auf.

---

## Das Schlüssel-Tür-Gitter-Experiment

Unser Skript testet beide Ansätze auf einem **offenen 9x9-Gitter** mit zwei Objekten:

- Einem **SCHLÜSSEL** in einer Ecke (muss zuerst gesammelt werden).
- Einer **TÜR** in der gegenüberliegenden Ecke (zählt nur, wenn man den Schlüssel hat).

Die einzige echte Belohnung ist +1, wenn der Agent die Tür erreicht, *nachdem* er den Schlüssel aufgehoben hat. Diese eine Belohnung erfordert, dass zwei aufeinanderfolgende Unteraufgaben korrekt verkettet werden.

Zwei Agenten treten gegeneinander an:

**Flaches DQN:** Muss zufällig über beide Unteraufgaben in der richtigen Reihenfolge stolpern und dann ein Signal durch beide zurückpropagieren. Da der Erfolg zwei Glückstreffer in einer Episode erfordert, lernt das DQN selten etwas Nützliches.

**Hierarchischer Agent:**
- Manager-Regel: „Gehe zuerst zum Schlüssel, dann zur Tür.“
- Der Worker erhält **+1 jedes Mal, wenn er ein Unterziel erreicht** – egal ob Schlüssel oder Tür.
- Zwei getrennte kurze Aufgaben, jede mit einer klaren Belohnung in der Nähe.

---

## Was die Diagramme zeigen

![Ergebnisse der Aufgabe mit langem Zeithorizont](outputs/long_horizon_tasks.png)

**Links — Erfolgsquote über die Zeit:** Der hierarchische Agent (blau) lernt, das Labyrinth viel früher zu lösen als das flache DQN (rot). Der flache Agent lernt vielleicht irgendwann auch – nach genügend Episoden –, aber der hierarchische Agent ist schneller am Ziel, weil sein Lernsignal dicht und lokal ist.

**Rechts — Finale Leistung:** Das Balkendiagramm zeigt die über die letzten 500 Episoden gemittelte Erfolgsquote. Der Vorteil des hierarchischen Agenten ist deutlich: Die Zerlegung des Problems in Unterziele macht es handhabbar.

---

## Wo hierarchisches Denken zum Einsatz kommt

| Bereich | Beispiel für langen Zeithorizont |
|--------|---------------------|
| Robotik | Ein Gerät mit 30 Teilen in der richtigen Reihenfolge zusammenbauen |
| Spiele | Eine Schachpartie gewinnen (viele Züge, ein Gewinner) |
| Sprache | Eine vollständige Forschungsarbeit schreiben (viele Schreibentscheidungen, eine Qualitätsbewertung) |
| Wissenschaft | Ein mehrmonatiges Experiment durchführen und die Ergebnisse auswerten |

Genau aus diesem Grund wurden Feudal Networks (eine Architektur, bei der Manager Richtungsziele für untergeordnete Worker festlegen) und HIRO (Hierarchical RL with Subgoals) erfunden – als flaches RL bei diesen Problemen an seine Grenzen stieß, wurde die hierarchische Dekomposition zur dominierenden Strategie.

---

## Die Verbindung zu zielkonditionierten Strategien

Beachten Sie, dass der **Worker** in unserem hierarchischen Agenten im Grunde eine **zielkonditionierte Strategie** ist – er erhält ein Unterziel und navigiert dorthin. Dies ist das Standarddesign in HIRO und verwandten Arbeiten: Der Manager setzt Ziele, der Worker ist eine zielkonditionierte Strategie, die diese verfolgt.

Die beiden Ideen – zielkonditionierte Strategien und hierarchische Struktur – sind daher zwei Seiten derselben Medaille, weshalb sie in diesem Modul zusammen erscheinen.

---

## Zusammenfassung in einem Satz

> **Aufgaben mit langem Zeithorizont sind schwierig, weil die Belohnung zu spät eintrifft, um einzelne Entscheidungen zu bewerten – hierarchisches RL löst dies durch das Einfügen von nahen Unterzielen, die den Worker schnell lernen lassen, während der Manager die übergeordnete Sequenz steuert.**
