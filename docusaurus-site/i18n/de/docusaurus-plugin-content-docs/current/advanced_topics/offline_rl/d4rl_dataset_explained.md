# D4RL Benchmark-Datensätze 📦

## Was ist das?

Stellen Sie sich vor, Sie möchten einem Roboter beibringen, Pfannkuchen zu wenden. Ihn einen Monat lang an einem echten Herd üben zu lassen, wäre langsam, gefährlich und teuer. Aber Sie haben Videoaufzeichnungen von Köchen aus zehn Jahren, die Pfannkuchen wenden (einige gut, einige schlecht, einige zufällig). Können Sie es dem Roboter *nur anhand dieser Daten* beibringen, ohne dass er jemals eine echte Pfanne berührt?

Das ist **Offline Reinforcement Learning**. Der Agent lernt aus einem festen Datensatz vergangener Erfahrungen – ohne Live-Umgebung. Der schwierigste Teil ist, dass der Agent das Gelernte erst ganz am Ende *ausprobieren* kann.

Um dies wissenschaftlich untersuchbar zu machen, benötigte die Forschungsgemeinschaft einen *Standard-Datensatz*. Das ist **D4RL** (**D**atasets for **D**eep **D**ata-**D**riven **R**einforcement **L**earning): eine Sammlung vorab aufgezeichneter Übergänge für klassische Steuerungsaufgaben, die 2020 von der UC Berkeley veröffentlicht wurde. Jedes Paper trainiert auf denselben Bytes, sodass die Ergebnisse vergleichbar sind.

---

## Was steckt in einem D4RL-Datensatz?

Für jede Aufgabe liefert D4RL **vier Qualitätsstufen**:

| Stufe | Woher die Daten kommen | Warum das wichtig ist |
|-------|---------------------------|----------------|
| **random**        | Eine Policy, die Aktionen gleichmäßig zufällig auswählt | Worst Case: Kann man trotzdem noch etwas Nützliches lernen? |
| **medium**        | Eine teilweise trainierte Policy (etwa die Hälfte des Expert-Scores) | Realistisch: Die meisten protokollierten Daten sind mittelmäßig |
| **expert**        | Eine nahezu konvergierte Policy | Best Case: Kann man die Quell-Policy erreichen? |
| **medium-replay** | Der *gesamte Replay-Buffer*, der zum Trainieren der Medium-Policy verwendet wurde | Gemischt: Enthält frühe Misserfolge UND spätere Erfolge |

Der Unterschied zwischen `medium` und `medium-replay` ist entscheidend:
- **`medium`** wird erzeugt, indem man eine einzelne, feste „durchschnittliche“ Policy nimmt und sie viele Spiele spielen lässt. Alle Daten spiegeln dieses stetige, durchschnittliche Fähigkeitsniveau wider.
- **`medium-replay`** ist ein historisches Protokoll. Es enthält alle Erfahrungen, die *während des Lernens* von Grund auf bis zum Medium-Niveau gesammelt wurden. Es mischt **schlechte und gute** Übergänge – genau so sieht ein reales Protokoll aus (die ersten tollpatschigen Versuche eines Roboters *und* sein späteres verfeinertes Verhalten, alles in einem Topf).

---

## Beispiele für Offline-Datensätze aus dem echten Leben

- **Medizinische Aufzeichnungen.** Jahre von (Patientenzustand, Behandlung, Ergebnis)-Tupeln. Man kann Behandlungen an lebenden Menschen nicht zufällig verteilen, aber man kann aus dem Protokoll eine bessere Behandlungsstrategie lernen.
- **Kundenservice-Chat-Logs.** Millionen von (Benutzernachricht, Agentenantwort, Zufriedenheit)-Datensätzen. Trainieren Sie einen besseren Assistenten, ohne weitere Benutzer zu belästigen.
- **Daten von Flotten autonomer Fahrzeuge.** Jedes Tesla- / Waymo-Auto lädt seine Fahrten hoch. Die Flotte ist ein riesiger Medium-Replay-Datensatz.
- **Empfehlungssysteme.** Klick-Protokolle vom letzten Jahr sind ein eingefrorener Datensatz: Man kann denselben Benutzern nicht dieselben Anzeigen erneut zeigen.

In allen vier Fällen **kann man die Umgebung nicht nach einer neuen Stichprobe fragen.** Der Datensatz ist alles, was man hat. Für immer.

---

## Was unser Code tut

Die echten D4RL-Datensätze werden bei MuJoCo-Locomotion-Aufgaben (Multi-Joint dynamics with Contact) aufgezeichnet (wie HalfCheetah, Hopper, Walker2d, Ant – dies sind fortgeschrittene 3D-Physiksimulationen, in denen virtuelle Roboter das Laufen lernen). MuJoCo ist aufwendig zu installieren, daher erstellen wir die **gleiche vierstufige Struktur auf CartPole-v1** nach – der Standardumgebung für Anfänger aus früheren Phasen. Die Lektionen sind direkt übertragbar.

Das Skript `d4rl_dataset.py`:

1. **Trainiert ein DQN** (Deep Q-Network, ein Standard-RL-Algorithmus) auf CartPole, bis es die Aufgabe löst (Return ≥ 475).
2. **Erstellt zwei Checkpoints** auf dem Weg dorthin:
   - „medium“ — der Moment, in dem der aktuelle Return 150 überschritt
   - „expert“ — der Moment, in dem der aktuelle Return 475 überschritt
3. **Speichert den gesamten Replay-Buffer der Medium-Policy** – jeden Übergang, den sie jemals gesehen hat. Das ist unser „medium-replay“-Datensatz.
4. **Führt drei neue Policies aus**, für jeweils 10.000 Übergänge:
   - `random`   — gleichmäßig zufällig
   - `medium`   — der Medium-Checkpoint + ε=0,10 Rauschen
   - `expert`   — der Expert-Checkpoint + ε=0,02 Rauschen
5. **Speichert vier `.npz`-Dateien** (NumPys komprimiertes Array-Format) in `outputs/`, jeweils mit den Arrays `obs / action / reward / next_obs / terminal`.

Diese vier Dateien sind die Eingaben für `cql.py` und `behavioral_cloning.py`.

---

## Was Sie sehen sollten, wenn Sie es ausführen

Eine Zusammenfassung wird in der Konsole ausgegeben und in `outputs/d4rl_summary.txt` gespeichert:

```
Datensatz       |   N    |  Mittlerer Return |  Min  |  Max
------------------------------------------------------------
random          | 10000  |          ~22  |    ~9 |   ~80
medium          | 10000  |         ~180  |   ~50 |  ~500
expert          | 10000  |         ~490  |  ~400 |   500
medium-replay   | 10000  |          ~60  |    ~9 |  ~200
```

Es wird auch ein Histogramm (`outputs/d4rl_returns.png`) erstellt, das zeigt, wie sich die vier Datensätze überschneiden. Die wichtigsten Merkmale:

- **Random** häuft sich um 20 (die durchschnittliche Länge einer zufälligen CartPole-Episode).
- **Expert** häuft sich bei der Obergrenze von 500.
- **Medium** liegt dazwischen, mit hoher Varianz.
- **Medium-replay** hat einen langen Ausläufer nach rechts – es besteht hauptsächlich aus frühen fehlgeschlagenen Versuchen (niedrige Returns), hat aber einen Ausläufer zu höheren Returns, während der Agent lernte.

---

## Warum der Datensatz wichtig ist

Egal auf welchem Datensatz Sie Ihren Offline-Algorithmus trainieren, Sie setzen eine *Obergrenze* für das Mögliche:

- **Von `expert`** — selbst ein einfacher Algorithmus wie BC (Behavioral Cloning, der die Daten einfach exakt kopiert) kann gut abschneiden, da alle Daten gut sind.
- **Von `random`** — Sie benötigen einen intelligenten Algorithmus, der seltene gute Übergänge *zusammenfügen* kann (indem er einen Pfad zum Erfolg findet, indem er kurze Sequenzen guter Aktionen aus verschiedenen Versuchen kombiniert). BC wird völlig scheitern.
- **Von `medium-replay`** — am realistischsten und interessantesten. Gute Algorithmen (wie **CQL** — Conservative Q-Learning, das vermeidet, bei Aktionen, die es nie gesehen hat, zu selbstsicher zu sein) können manchmal **die durchschnittliche Datenqualität schlagen**, weil sie Struktur aus gemischten Signalen extrahieren. Einfache Algorithmen (BC) fallen auf den Durchschnitt zurück.

Wir werden genau diese Geschichte in den nächsten beiden Skripten sehen.

---

## Wichtige Begriffe zum Merken

| Begriff | Bedeutung |
|------|---------|
| **Offline RL**         | Training aus einem festen Datensatz; keine Interaktion mit der Umgebung erlaubt |
| **Behaviour Policy**   | Die Policy, die den Datensatz *erzeugt* hat |
| **Datensatzqualität**    | Wie gut die Behaviour Policy war (random / medium / expert) |
| **Replay Buffer**      | Die vollständige Historie der Übergänge, die während eines Trainingslaufs gesehen wurden |
| **Distribution Shift** | Die Lücke zwischen den Aktionen im Datensatz und den Aktionen, die Ihre trainierte Policy wählen möchte. Da der Datensatz nie zeigt, was passiert, wenn die neue Policy etwas ausprobiert, das nicht aufgezeichnet wurde, können die Wertschätzungen des Algorithmus für diese neuartigen Aktionen gefährlich falsch sein. |

---

## Zusammenfassung in einem Satz

> **D4RL macht aus RL einen Benchmark im Stil des Supervised Learning: die gleichen Bytes für alle, kein Schummeln mit der Umgebung, möge der beste Algorithmus gewinnen.**
