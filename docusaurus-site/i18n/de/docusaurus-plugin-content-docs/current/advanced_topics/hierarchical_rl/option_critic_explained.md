# Option-Critic-Architektur

## Die Kernidee: In Kapiteln arbeiten, nicht Wort für Wort

Stellen Sie sich vor, Sie schreiben einen Roman. Sie planen nicht jedes einzelne Wort, bevor Sie beginnen. Stattdessen denken Sie in **Kapiteln**: „Kapitel 1 stellt den Helden vor. Kapitel 2 ist die Suche. Kapitel 3 ist der Showdown.“ Innerhalb jedes Kapitels legen Sie die Details fest, während Sie schreiben.

Genau so denkt die Option-Critic-Architektur über Entscheidungen nach.

---

## Was ist ein „flacher“ Agent?

Ein normaler RL-Agent (wie die aus Phase 3 und 4 des Lehrplans) entscheidet sich in jedem einzelnen Schritt für genau eine Aktion. Das ist wie ein GPS, das bei jedem Meter, den Sie sich bewegen, die gesamte Route von Grund auf neu berechnet. Es funktioniert, ist aber anstrengend und langsam zu lernen.

---

## Was ist eine „Option“?

Eine **Option** ist eine **benannte Fertigkeit** — eine Mini-Strategie (Policy), die der Agent über mehrere Schritte hinweg ausführen kann, bevor er die Kontrolle wieder abgibt.

Stellen Sie es sich wie einen Manager vor, der Aufgaben an Spezialisten delegiert:

| Wer | Was sie tun |
|-----|-------------|
| **Manager (Meta-Policy)** | Entscheidet, *welchen* Spezialisten er für eine Aufgabe schickt |
| **Spezialist A** | Experte für das Navigieren im Raum oben links |
| **Spezialist B** | Experte für das Durchqueren von Türöffnungen |
| **Spezialist C** | Experte für das Stürmen auf das Ziel |
| **Spezialist D** | Ersatz-Generalist |

Der Manager wählt einen Spezialisten aus. Der Spezialist arbeitet so lange, bis er entscheidet, dass er fertig ist (dies nennt man **Terminierung**). Dann wählt der Manager erneut aus.

---

## Die drei beweglichen Teile

Jede Option besteht aus drei Komponenten — betrachten Sie diese als die **Jobbeschreibung** des Spezialisten:

1. **Initiierung**: Wann kann dieser Spezialist gerufen werden? *(z. B. „Spezialist A wird nur in der Nähe des Raums oben links aktiv.“)*
2. **Intra-Option-Policy**: Was tut der Spezialist, während er arbeitet? *(z. B. „Gehe in Richtung der Ecke oben links.“)*
3. **Terminierung**: Wann gibt der Spezialist die Kontrolle zurück? *(z. B. „Höre auf, wenn du eine Türöffnung erreicht hast.“)*

Das Schöne an Option-Critic ist, dass alle drei Komponenten **automatisch gelernt werden** — Sie müssen die Spezialisten nicht manuell entwerfen. Der Algorithmus findet ganz von selbst heraus, dass es nützlich ist, eine Option für jeden Raum oder eine für den Endspurt zum Ziel zu haben.

---

## Ein Tag im Leben eines Option-Critic-Agenten

1. Der Agent betritt einen neuen Raum (Zustand).
2. Der **Manager** betrachtet den Raum und wählt eine Option aus — sagen wir, Option 2.
3. Der **Spezialist von Option 2** übernimmt: Er geht Schritt für Schritt auf die Türöffnung zu.
4. An einem bestimmten Punkt sagt Option 2: „Ich bin hier fertig“ (Terminierung).
5. Der **Manager** wird wieder aktiv und wählt eine neue Option für die neue Situation.
6. Wiederholung.

Vergleichen Sie dies mit dem flachen Agenten: Der flache Agent quält sich bei jedem einzelnen Schritt mit der Entscheidung ab. Option-Critic delegiert ganze Verhaltensabschnitte, sodass jeder Spezialist in seinem engen Aufgabengebiet richtig gut werden kann.

---

## Warum hilft das?

In einem Labyrinth muss der Agent ein Ziel erreichen, das 30–50 Schritte entfernt sein kann. Beim flachen Lernen ist jeder Schritt auf dem Pfad gleichermaßen „unsichtbar“, bis am Ende endlich die Belohnung eintrifft — dieses Signal muss über Dutzende von Schritten rückwärts wandern.

Mit Optionen wird der Pfad in **Unteraufgaben** unterteilt. Jede Unteraufgabe erhält ihr eigenes Mini-Belohnungssignal (Erreichen der Tür, Betreten des nächsten Raums). Das Lernen breitet sich über kürzere Segmente aus. **Der Agent lernt schneller bei Problemen, die viele Schritte erfordern.**

Dies ist die Kernidee hinter dem gesamten hierarchischen RL — und Option-Critic ist eine der reinsten Umsetzungen davon.

---

## Was unser Code tut

Das Skript `option_critic.py` setzt einen Option-Critic-Agenten in eine **7x7-Gridworld** mit einem festen Ziel. Der Agent startet an einer beliebigen Stelle im Gitter und muss zur Zielzelle navigieren.

Der Agent verfügt über vier Optionen und muss gleichzeitig lernen:

- Eine Policy für jede Option (wohin er gehen soll)
- Wann jede Option beendet werden soll (Terminierungsbedingung)
- Eine Meta-Policy zur Auswahl zwischen den Optionen

Die Belohnung verwendet **potentialbasiertes Shaping** — der Agent erhält in jedem Schritt einen kleinen Bonus, wenn er sich dem Ziel nähert, zusätzlich zu +1 für das Erreichen des Ziels. Dieses dichte Feedback macht das Lernen stabil genug, um innerhalb von 2.500 Episoden zu sehen, wie die Optionen funktionieren.

Kein Mensch gibt jemals vor, was jede Option tun soll. Der Algorithmus entdeckt selbst, auf welche Bereiche des Gitters sich jede Option spezialisiert.

---

## Was die Diagramme zeigen

![Option-Critic Lernkurven](outputs/option_critic.png)

**Links — Shaped Return:** Ein höherer Return bedeutet, dass der Agent das Ziel zuverlässiger erreicht *und* kürzere Wege wählt (das Shaping gibt einen Bonus pro Schritt näher am Ziel). Das Ansteigen und anschließende Stabilisieren der Kurve zeigt, wie die Optionen lernen, sich zu koordinieren.

**Rechts — Schritte zum Ziel:** Weniger Schritte bedeuten, dass der Agent einen effizienteren Weg gefunden hat. Der Abwärtstrend zeigt, wie die Optionen zu kohärenten Fertigkeiten reifen, die den Agenten direkter zum Ziel führen.

Die geglätteten Kurven zeigen den allgemeinen Trend über Fenster von 100 Episoden — ein gewisses Rauschen ist beim RL normal, besonders wenn mehrere Komponenten (Optionen, Terminierung, Meta-Policy) gleichzeitig lernen.

---

## Zusammenfassung in einem Satz

> **Option-Critic bringt einem Agenten bei, in Fertigkeiten statt in Einzelschritten zu arbeiten — ein Manager wählt aus, welcher Spezialist läuft, jeder Spezialist erledigt seinen Job, und das gesamte System lernt gemeinsam aus demselben Belohnungssignal.**
