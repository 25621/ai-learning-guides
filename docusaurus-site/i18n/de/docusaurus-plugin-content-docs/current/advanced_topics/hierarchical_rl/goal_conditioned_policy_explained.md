# Zielkonditionierte Strategie (Goal-Conditioned Policy)

## Die Kernidee: Eine Strategie für alle Ziele

Stellen Sie sich vor, Sie sind ein Lieferfahrer. Sie benötigen nicht für jede Adresse völlig neue Fähigkeiten. Sie wissen, wie man fährt, eine Karte liest und durch den Verkehr navigiert – Sie geben einfach das *heutige Ziel* ein und fahren los.

Eine **zielkonditionierte Strategie** funktioniert genauso. Anstatt einen Agenten zu trainieren, der nur zu einem einzigen festen Ziel gelangen kann, trainieren wir einen einzelnen Agenten, der jedes beliebige Ziel als Eingabe akzeptiert und herausfindet, wie er dorthin gelangt.

---

## Wie sie sich vom Standard-RL unterscheidet

Im Standard-RL (wie es in den früheren Phasen des Lehrplans behandelt wurde) ist die Belohnungsfunktion fest eingebaut: „erreiche Zelle (7, 7), erhalte +1“. Der Agent lernt genau eine Sache: wie er *diese eine* Zelle erreicht.

Beim zielkonditionierten RL hängt die Belohnung davon ab, ob der Agent das Ziel erreicht, das *ihm diesmal vorgegeben wurde*. Die Policy lernt:

> **„Basierend darauf, wo ich bin und wo ich sein möchte, was sollte ich tun?“**

Das Ziel reist mit dem Agenten mit, wie ein in eine Navigations-App eingegebenes Ziel.

---

## Das Problem der spärlichen Belohnung (Sparse Rewards)

Der Haken ist: Das Lernen aus spärlichen Belohnungen (nur +1 am Ziel, 0 überall sonst) ist extrem schwierig. Die meisten Versuche scheitern – der Agent wandert zufällig umher, stößt nie auf das Ziel, und das Netzwerk erhält keine nützlichen Informationen zum Lernen.

Stellen Sie sich vor, Sie versuchen mit verbundenen Augen Dartspielen zu lernen. Sie werfen tausendmal und treffen nie. Nach tausend Fehlwürfen haben Sie immer noch keine Ahnung, wie sich ein „guter Wurf“ anfühlt.

Hier kommt **Hindsight Experience Replay (HER)** ins Spiel.

---

## Hindsight Experience Replay: Aus Fehlern lernen

Der Trick von HER ist bestechend einfach. Nach einer fehlgeschlagenen Episode fragt HER:

> *„Auch wenn du dein Ziel nicht erreicht hast… wo bist du eigentlich gelandet?“*

Dann **spielt es dieselbe Episode noch einmal ab**, gibt aber vor, die tatsächliche Endposition des Agenten **wäre** von Anfang an das Ziel gewesen. Plötzlich wird aus einer fehlgeschlagenen Episode eine erfolgreiche – für ein anderes Ziel.

Es ist wie ein Basketballspieler, der immer wieder auf den Korb wirft und daneben trifft. HER würde sagen: „Okay, du hast jedes Mal die linke Wand getroffen. Herzlichen Glückwunsch – du bist großartig darin, die linke Wand zu treffen! Notieren wir diese Würfe als erfolgreiche Versuche, die linke Wand zu treffen.“ Mit der Zeit baut der Spieler Fähigkeiten auf, *jedes beliebige* Ziel zu treffen, und überträgt diese schließlich auf den echten Korb.

Dies verwandelt tausende „Fehlschläge“ in eine reiche Bibliothek von *erfolgreichen* Navigationen zu vielen verschiedenen Punkten. Der Agent lernt, sie alle zu erreichen, was sich auf das eigentliche Ziel verallgemeinern lässt.

---

## Analogie aus dem echten Leben: Ein Kleinkind lernt, Bauklötze zu stapeln

Ein Kleinkind, das versucht, einen Klotz in einen Eimer zu legen, verfehlt ihn ständig. Aber jeder „Fehlwurf“ landet *irgendwo*. Wenn man jeden Fehlwurf so wertet wie „du hast versucht, ihn genau *dorthin* zu legen – und du hast es geschafft!“, baut das Kind motorische Fähigkeiten über den gesamten Tisch hinweg auf. Bald kann es einen Klotz überall platzieren – auch im Eimer.

---

## Was unser Code tut

Das Skript `goal_conditioned_policy.py` läuft in einem **7x7-Labyrinth** mit Wänden. Zu Beginn jeder Episode wird eine zufällige Zielzelle ausgewählt. Der Agent muss sie finden.

Die Policy erhält bei jedem Schritt zwei Eingaben:
1. Wo der Agent sich aktuell befindet.
2. Wohin er gelangen möchte.

Nach jeder Episode (ob erfolgreich oder nicht) generiert HER mehrere zusätzliche synthetische „Erfolge“, indem es die tatsächlich besuchten Positionen als alternative Ziele umdeklariert.

Das Training läuft über 3.000 Episoden mit einer abnehmenden Explorationsrate – der Agent exploriert anfangs mehr und vertraut dann zunehmend dem, was er gelernt hat.

---

## Was die Diagramme zeigen

![Ergebnisse der zielkonditionierten Strategie](outputs/goal_conditioned_policy.png)

**Links — Erfolgsquote über das Training:** Jede Episode ist entweder ein Erfolg (Ziel erreicht) oder ein Misserfolg. Die Kurve steigt stetig an, während sich die universelle Navigationsfähigkeit des Agenten verbessert. Am Ende erreicht der Agent fast jedes Mal jedes beliebige Ziel.

**Rechts — Heatmap der Zielerfolgsquote:** Nach dem Training testen wir den Agenten für jede mögliche Zielzelle und färben jede Zelle danach ein, wie oft der Agent sie erreicht. Grün bedeutet, dass der Agent diesen Punkt zuverlässig erreicht; Rot bedeutet, dass er dort noch Schwierigkeiten hat. Ein gut trainierter Agent zeigt fast über das gesamte Labyrinth hinweg Grün.

---

## Wo dies in der realen Welt vorkommt

| Anwendung | Das „Ziel“ |
|-------------|------------|
| Roboterarm-Greifen | Zielposition im 3D-Raum |
| Selbstfahrendes Auto | GPS-Koordinate |
| Sprachmodell-Assistent | Anweisung des Benutzers |
| Nicht-Spieler-Charakter in Videospielen | Beliebiger Wegpunkt auf der Karte |

Zielkonditionierte Strategien sind einer der wichtigsten Bausteine für HIRO (Hierarchical RL mit Subgoals) – der übergeordnete Manager wählt ein Unterziel (Subgoal) aus, und der untergeordnete Arbeiter ist genau diese Art von zielkonditionierter Strategie.

---

## Zusammenfassung in einem Satz

> **Eine zielkonditionierte Strategie ist ein einzelner Agent, der zu jedem beliebigen Ziel navigieren kann – und HER macht das Lernen aus Fehlern möglich, indem so getan wird, als wäre jeder Fehlwurf genau dorthin gezielt gewesen, wo er gelandet ist.**
