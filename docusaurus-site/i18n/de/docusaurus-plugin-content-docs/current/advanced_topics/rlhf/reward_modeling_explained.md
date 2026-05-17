# Belohnungsmodellierung (Reward Modeling): Einem Computer beibringen, was Menschen bevorzugen

## Die Kernidee

Ein Belohnungsmodell (Reward Model) ist ein kleiner Richter. Man zeigt ihm zwei Antworten auf dieselbe Frage, sagt ihm, welche einem Menschen besser gefallen hat, und mit der Zeit lernt es, Antworten, die Menschen bevorzugen würden, eine höhere Punktzahl (Score) zu geben.

Warum brauchen wir einen solchen Richter? Weil das meiste, was wir von einem Sprachmodell erwarten, schwer in einer mathematischen Formel auszudrücken ist. Es gibt keine einfache Gleichung für „hilfreich“, „höflich“ oder „gut geschrieben“. Aber Menschen können fast immer auf die bessere von zwei Optionen zeigen. Das Belohnungsmodell verwandelt diese einfachen „Diese hier ist besser“-Stimmen in einen Score, den ein Lernalgorithmus nutzen kann.

## Eine Analogie aus dem echten Leben

Stellen Sie sich vor, Sie bringen einem Freund bei, Brownies zu backen.

Sie geben ihm kein 50-seitiges Regelwerk darüber, „was einen guten Brownie ausmacht“. Stattdessen probieren Sie zwei Bleche und sagen:

„Dieser hier ist besser.“

Nach ein paar Runden fängt Ihr Freund an, Muster zu bemerken. Vielleicht gewinnt immer der saftigere (fudgier). Vielleicht verliert immer der zu trocken gebackene. Ihr Freund baut aus Ihren Vergleichen ein mentales Bewertungssystem auf.

Ein Belohnungsmodell macht genau das, aber mit Zahlen. Es muss nicht wissen, *warum* die gewählte Antwort besser ist. Es braucht nur viele „Dies schlägt das“-Beispiele und lernt schrittweise einen Score, der mit den Präferenzen übereinstimmt.

## Wie das Lernen funktioniert (Intuition)

Jedes Beispiel ist ein Tripel: ein Prompt, eine **gewählte (chosen)** Antwort und eine **abgelehnte (rejected)** Antwort. Wir möchten, dass das Modell der gewählten Antwort einen höheren Score gibt als der abgelehnten – egal mit welchem Abstand.

Der Anstoß beim Training ist im Prinzip einfach:

- Score der gewählten Antwort zu niedrig? Drück ihn nach oben.
- Score der abgelehnten Antwort zu hoch? Drück ihn nach unten.
- Bereits in der richtigen Reihenfolge mit deutlichem Abstand? Lass sie in Ruhe.

Dieser Anstoß wird Bradley-Terry-Loss genannt und ist das Standardrezept in modernen RLHF-Systemen.

## Was das Experiment zeigt

Wir haben ein Belohnungsmodell auf 2.000 synthetischen Präferenzpaaren trainiert. Die Grafik unten zeigt drei Ansichten desselben Trainingslaufs.

![Reward Model Training](outputs/reward_modeling.png)

- **Links** — der Loss sinkt schnell. Das Modell wird sicherer bei seinen Einstufungen.
- **Mitte** — die Präferenzgenauigkeit steigt auf nahezu 100 %. Bei fast jedem Paar erhält die gewählte Antwort einen höheren Score als die abgelehnte.
- **Rechts** — die Score-Verteilungen für gewählte vs. abgelehnte Antworten driften auseinander. Zu Beginn überlappten sie sich; nach dem Training liegen die gewählten Antworten deutlich weiter rechts.

Diese Trennung ist der entscheidende Punkt. Ein späterer Schritt (PPO oder DPO) kann diesen Score nun als Zielwert nutzen, auf den hin optimiert wird.

## Wo PPO in der RLHF-Pipeline steht

Der Fahrplan beschreibt RLHF als „Ausrichtung (Alignment) von Modellen an menschlichen Präferenzen“. Die Belohnungsmodellierung ist Schritt eins von drei:

1. **Belohnungsmodell (diese Datei)** – verwandelt Präferenzstimmen in einen Score.
2. **PPO-Feintuning** – drückt das Sprachmodell in Richtung höherer Scores, während es nah an seinem ursprünglichen Verhalten bleibt.
3. **DPO** – eine neuere Abkürzung, die das Belohnungsmodell komplett überspringt.

Die Belohnungsmodellierung ist also die Brücke zwischen *menschlichem Urteilsvermögen* und *maschineller Optimierung*. Wenn diese Brücke fehlerhaft ist, wird jeder nachfolgende Schritt in die falsche Richtung gelenkt.

## Warum das außerhalb des Labors wichtig ist

Dieselbe Idee findet sich an vielen Stellen:

- **Empfehlungssysteme** lernen aus Klicks, Skips und der verbrachten Zeit, was Ihnen gefällt.
- **Suchmaschinen** lernen das Ranking daraus, „welches Ergebnis Sie angeklickt haben“.
- **Restaurants** erfahren durch wiederholte Bestellungen, welche Gerichte beliebt sind, nicht dadurch, dass Kunden Aufsätze darüber schreiben, was ihnen geschmeckt hat.

Wann immer es einfacher ist zu *vergleichen* als zu *bewerten*, ist ein Belohnungsmodell das richtige Werkzeug.

## Zusammenfassung in einem Satz

**Ein Belohnungsmodell ist ein gelernter Richter, der „Diese hier ist besser“-Präferenzen in einen Zahlenwert verwandelt, den der Rest von RLHF optimieren kann.**
