# Self-Play: Einem Agenten beibringen, indem er gegen sich selbst spielt ♟️

## Was ist Self-Play?

Stellen Sie sich ein Kind vor, das im Schach richtig gut werden will, aber niemanden zum Spielen hat. Also spielt sie gegen sich selbst. Linke Hand gegen rechte Hand. In jedem Spiel versuchen *beide* Seiten zu gewinnen. In jedem Spiel lernen *beide* Seiten, was funktioniert hat.

Das ist **Self-Play**: Ein einzelner Agent agiert als beide Spieler, und jeder Zug wird zu einer Lektion für denjenigen, der als Nächstes zieht. Kein Lehrer, kein Experten-Gegner. Nur ein Lernender, der gleichzeitig seine eigene Messlatte ist.

Self-Play klingt wie ein Trick – braucht man nicht einen echten Gegner? Aber es ist der Motor hinter den berühmtesten RL-Meilensteinen des letzten Jahrzehnts: **AlphaGo Zero**, **AlphaZero**, **MuZero**, **OpenAI Five**. Sie alle nutzen Self-Play. Der Grund ist einfach: Wenn Sie besser werden, wird Ihr Gegner im gleichen Maße besser. Die Herausforderung passt sich immer Ihrem Können an.

---

## Warum es funktioniert

Drei Dinge machen Self-Play besonders:

1. **Endlose Gegner.** Ihnen gehen nie die Spiele aus. Der Gegner ist immer da und kostenlos.
2. **Lehrplan, der mitwächst.** Ein Anfänger kann nur gegen andere Anfänger spielen. Wenn Sie besser werden, wird auch Ihr Schatten automatisch besser.
3. **Symmetrie.** In einem Nullsummenspiel (der Gewinn des einen ist der Verlust des anderen) beschreibt ein Satz von Q-Werten beide Seiten; man dreht einfach das Vorzeichen um, wenn der andere Spieler an der Reihe ist. So kann ein *einziger* Q-Table sich selbst unterrichten.

Tic-Tac-Toe ist das perfekte Testfeld: klein genug für ein Dictionary, aber komplex genug, dass zufällige Züge gegen einen strategischen Spieler fast immer zum Verlust führen.

---

## Eine Analogie aus dem echten Leben

- **Tennis gegen eine Wand üben.** Man kann nicht gegen eine Wand verlieren, aber man kann seine Aufschläge üben. Self-Play ist so, als würde man das an beiden Enden tun – man ist die Wand *und* der Spieler und wechselt hin und her.
- **Ein Debattierclub, der beide Seiten argumentiert.** Bessere Debattierer entstehen dadurch, dass sie immer die gegenteilige Ansicht zu dem verteidigen, was sie persönlich glauben. Jedes Argument trainiert sowohl Angriff als auch Verteidigung.
- **AlphaGo Zero.** Es hat von Null menschlichen Spielen gelernt. Beginnend mit zufälligen Zügen spielte es Millionen von Spielen gegen sich selbst; in wenigen Tagen war es besser als jedes vorherige Go-Programm, einschließlich desjenigen, das Lee Sedol besiegte.

---

## Was unser Code macht

Wir lernen einen Q-Table für den *Spieler, der gerade am Zug ist*:

```
Q[(Brett, Spieler_am_Zug)][Aktion] = erwarteter Ertrag für diesen Spieler
```

Die Trainingsschleife ist:

1. Start mit einem leeren Brett. `Spieler = X`.
2. Beide Spieler agieren mit dem **gleichen Agenten** unter Verwendung von ε-greedy.
3. Nach jedem Spiel gehen wir rückwärts durch jedes (Brett, Spieler, Aktion)-Tripel in der Historie und wenden das Q-Learning-Update an.
4. Die Belohnung dreht das Vorzeichen zwischen den Zügen: Wenn X gewinnt, erhält jeder Zug, den X gemacht hat, +1 (oder übernimmt den Wert von einem zukünftigen Gewinnzustand); jeder Zug, den O gemacht hat, erhält -1.
5. Wir verringern (decay) langsam unsere Explorationsrate (ε) von 0,2 → 0,02, damit der Agent sich gegen Ende des Trainings auf sein bestes Spiel festlegt, anstatt zufällige Züge auszuprobieren.

Alle 2.500 Episoden bewerten wir den Agenten gegen einen **zufälligen Gegner** (wir frieren den Lernprozess ein, sodass während der Bewertung keine neuen Updates am Q-Table vorgenommen werden und beide Seiten gierig/greedy spielen). Der Agent sollte nach genügend Self-Play ~100% dieser Spiele gewinnen oder unentschieden spielen.

### Was Sie sehen sollten

Nach 50.000 Self-Play-Episoden:

| Paarung | Erwartetes Ergebnis |
|----------|-----------------|
| Trainierter Agent vs. Zufälliger Gegner (1000 Spiele) | **~95-99% Siege oder Unentschieden**, praktisch 0% Verluste |
| Trainierter Agent vs. sich selbst (200 greedy Spiele) | **Alle 200 Unentschieden**. Tic-Tac-Toe ist ein Spiel, das immer unentschieden endet, wenn beide Spieler perfekt spielen. Die Tatsache, dass Self-Play jedes Spiel unentschieden beendet, ist ein Zeichen für Konvergenz. |

Das Diagramm `outputs/self_play_tic_tac_toe.png` zeigt die Anteile von Siegen/Unentschieden/Verlusten des Agenten gegen einen zufälligen Gegner im Zeitverlauf:
- Die Siegquote beginnt bei ~60% (wenn beide Spieler zufällig spielen, hat der erste Spieler einen inhärenten Vorteil, da er mehr Steine auf das Brett legen kann, was zu einer Basis-Siegquote von etwa 60% für Spieler X führt).
- Steigt auf >90%.
- Die Verlustquote fällt auf fast 0%.

Das Skript gibt am Ende auch ein Beispielspiel Zug für Zug aus, damit Sie den Agenten beim Spielen beobachten können.

---

## Achten Sie auf diese Feinheiten

- **Vorzeichenwechsel sind wichtig.** Ein häufiger Fehler: Vergessen, dass "der Gegner maximiert seinen Wert" für uns bedeutet, dass er *unseren Wert minimiert*. Das Update in unserem Code verwendet `target = reward - gamma * max(Q[next, opponent])`.
- **Symmetrie wird hier nicht ausgenutzt.** Eine echte Implementierung würde Bretter kanonisieren (was bedeutet, dass sie jeden Brettzustand rotieren oder spiegeln würde, damit der Agent identische Brettsituationen erkennt), um Q-Werte über 8 Symmetrien hinweg zu teilen. Wir überspringen dies – der Zustandsraum ist klein genug für Brute-Force.
- **Der Q-Table wächst.** Nach 50.000 Self-Play-Spielen werden Sie einige tausend Zustand-Spieler-Keys sehen. Das ist hier in Ordnung; für Schach oder Go bräuchten Sie stattdessen ein neuronales Netz, weshalb **AlphaZero den Table durch ein CNN + MCTS ersetzt**.

---

## Wo Self-Play an seine Grenzen stößt

- **Nicht-Nullsummenspiele.** "Beide Seiten glücklich" ist unvereinbar mit symmetrischem Spiel; man kann nicht einfach ein Vorzeichen umdrehen.
- **Asymmetrische Rollen.** Wenn "Angreifer" und "Verteidiger" unterschiedliche Aktionsräume haben, benötigen Sie zwei separate Netzwerke.
- **Strategie-Zyklen.** Reines Self-Play kann in Zyklen wie Schere-Stein-Papier stecken bleiben. AlphaStar löste dies, indem es einen großen *Pool* (oder eine "League") gespeicherter vergangener Versionen des Agenten behielt und Gegner aus diesem Pool zufällig auswählte, sodass der Agent lernt, viele verschiedene Spielstile zu besiegen, anstatt nur den aktuellen.
- **Reward-Hacking.** Self-Play macht beide Seiten klüger, aber nur in dem Spiel, *wie Sie es definiert haben*. Wenn Ihr Belohnungssystem unbeabsichtigte Schlupflöcher hat (wie die Belohnung eines Spielers nur für das längere Überleben anstatt für den Sieg), werden beide Seiten das Schlupfloch gegenseitig ausnutzen, was zu bizarrem, wenig hilfreichem Verhalten führt, anstatt das eigentliche Spiel zu meistern.

---

## Wichtige Begriffe

| Begriff | Bedeutung |
|------|---------|
| **Self-Play**      | Derselbe Agent spielt beide Seiten eines Spiels |
| **Nullsummenspiel** | Der Gewinn des einen = der Verlust des anderen |
| **Symmetrie**       | Ein Q-Table kann beiden Seiten dienen, wenn man Vorzeichen umdreht |
| **Population Play** | Self-Play mit *vielen* vergangenen Versionen von sich selbst als Gegner (AlphaStar) |
| **Curriculum**     | Eine natürliche Schwierigkeitssteigerung – Self-Play bekommt dies umsonst |
| **MCTS**           | Monte-Carlo Tree Search – der Planungsalgorithmus, den AlphaZero mit Self-Play kombiniert |

---

## Zusammenfassung in einem Satz

> **Self-Play macht Fortschritt zu seiner eigenen Leiter: Jedes Mal, wenn Sie besser werden, wird es auch Ihr Gegner – ganz automatisch.**

Diese Idee, skaliert mit **neuronalen Netzen** (von Gehirnen inspirierte mathematische Funktionen, die Muster aus Daten lernen) und Baumsuche, besiegte die besten Menschen in Go, Schach, Shogi, Dota 2 und StarCraft.
