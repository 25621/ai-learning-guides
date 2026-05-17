# Conservative Q-Learning (CQL) 🛡️

## Was ist das?

Stellen Sie sich vor, Sie lernen Investieren, indem Sie ein riesiges Buch mit vergangenen Aktientransaktionen anderer Leute lesen. Das Buch enthält Käufe, Verkäufe und Haltepositionen – aber **keine Aufzeichnung einer Transaktion, die tatsächlich niemand getätigt hat**.

Stellen Sie sich nun vor, ein übermütiger Student schaut in das Buch und sagt:
*„Was wäre, wenn jemand jeden Montag Lotteriescheine gekauft hätte? Das wäre ein fantastisches Geschäft gewesen!“*

Das Problem: **Das Buch enthält keine Daten über den Kauf von Lottoscheinen am Montag**, also halluziniert der Student einfach. Dennoch sieht dieses halluzinierte Geschäft auf dem Papier großartig aus, sodass die „Policy“ (Strategie) des Studenten es immer wieder tun möchte.

Dieses Halluzinationsproblem ist der **Distribution Shift** (Verteilungsverschiebung): Ein Offline-Lerner liebt Aktionen, die der Datensatz nie getestet hat, weil es keine Daten gibt, die dem Optimismus widersprechen. CQL ist die Lösung dafür.

---

## Wie Q-Learning offline schiefgeht

Das Ziel (Target) beim normalen Q-Learning ist:

```
target(s, a) = r + γ · max_{a'} Q(s', a')
```

Dieses `max_{a'}` ist die Gefahr. Wenn der Datensatz die Aktion `a'` im Zustand `s'` nie aufgezeichnet hat, *rät* das Netzwerk einfach einen Q-Wert – und neuronale Netzwerke neigen dazu, Q-Werte für ungesehene Eingaben zu **überschätzen**. Das Target erbt die Überschätzung, das Netzwerk lernt, diese größere Zahl vorherzusagen, und beim nächsten Schritt **extrapolieren** wir (projizieren noch weiter über alles hinaus, was die Daten stützen) noch höher. Die Policy jagt einem Phantom hinterher.

Wenn man weiterhin neue Daten sammeln könnte, würde sich dies selbst korrigieren (die Phantom-Aktion stellt sich in der Realität als schlecht heraus). Aber **beim Offline RL kann man keine neuen Daten sammeln.** Das Phantom bleibt für immer bestehen.

---

## Der Trick von CQL

CQL (Kumar et al., 2020) fügt der Loss-Funktion einen Strafterm hinzu:

```
cql_loss(s)  =  log Σ_a exp Q(s, a)   -   Q(s, a_dataset)
```

Zwei Teile:

1. **`log Σ_a exp Q(s, a)`** (sprich: *„Log-Sum-Exp über alle Aktionen“*) ist ein **Soft-Maximum** über alle Aktionen – eine glatte, differenzierbare Annäherung an `max`, die jede Aktion gleichzeitig berücksichtigt, anstatt einen harten Gewinner auszuwählen. Das Bestrafen dieses Terms senkt die Q-Werte **auf breiter Front** (drückt alle Vorhersagen gleichmäßig nach unten) – besonders für Aktionen mit den *höchsten* Q-Werten, also genau dort, wo die Halluzinationen leben.
2. **`- Q(s, a_dataset)`** belohnt hohe Q-Werte bei der Aktion, die der Datensatz tatsächlich aufgezeichnet hat – dies schützt In-Distribution Q-Werte vor der oben genannten Senkung.

Nettoeffekt: **Q-Werte werden bei ungesehenen Aktionen nach unten gedrückt und bei gesehenen Aktionen nach oben gezogen.** Das gelernte Q wird zu einer *unteren Schranke* (Lower Bound) des wahren Q. Die **`argmax`**-Policy (die Regel, die einfach die Aktion mit dem höchsten Q-Wert wählt) hört auf, Phantomen nachzujagen.

Vollständiger Loss:

```
L  =  Bellman_MSE   +   α · cql_loss
```

(Wobei **`Bellman_MSE`** der Standardfehler aus dem normalen Q-Learning ist, der misst, wie sehr die aktuelle Schätzung des Netzwerks mit seiner eigenen zukünftigen Schätzung nicht übereinstimmt).

`α` ist der Regler für den Konservatismus. Zu klein → der Distribution Shift schleicht sich wieder ein. Zu groß → der Agent ist so konservativ, dass er sich nie über die Daten hinaus verbessert.

---

## Beispiele aus dem echten Leben

- **Konservativer Schachtrainer.** Man kann nur aus bereits gespielten Partien lernen. Ein leichtsinniger Trainer sagt: „Dieser hypothetische Zug ohne Vorbild könnte brillant sein!“ CQL ist der Trainer, der sagt: „Wir haben dazu keine Daten – bleiben wir bei Zügen, die echte Spieler ausprobiert haben.“
- **Speisekartenauswahl im Restaurant.** Yelp-Bewertungen decken niemals die Gerichte ab, die nicht auf der Karte stehen. Eine naive Policy würde die Gerichte außerhalb der Karte basierend auf halluzinierten Fünf-Sterne-Bewertungen empfehlen. CQL empfiehlt nur das, was oft genug bestellt wurde, um darauf zu vertrauen.
- **Robotergreifen aus Protokollen.** Der Roboter hat Videos vom Greifen von Tassen, Flaschen und Büchern – aber nie von einem Messer. CQL weigert sich, selbstbewusst zu empfehlen: „Greifen Sie das Messer an der Klinge.“

---

## Was unser Code tut

Das Skript `cql.py`:

1. **Lädt die vier Datensätze**, die durch `d4rl_dataset.py` erstellt wurden.
2. **Wählt `medium-replay`** als Trainingssatz – er ist am realistischsten (gemischte Qualität) und am schädlichsten für naive Methoden.
3. **Trainiert drei Agenten rein offline**, unter identischen Bedingungen, außer für `α`:
   - `α = 0`   →  naives Offline-DQN (keine Strafe – die fehlerhafte Baseline)
   - `α = 1.0` →  leichtes CQL
   - `α = 5.0` →  starkes CQL
4. **Evaluiert jeden Agenten alle 2.500 Gradientenschritte**, indem er gierig (greedy) in der echten Umgebung ausgeführt wird (10 Episoden). Dies ist der *einzige* Kontakt zur Umgebung; das Training selbst sieht die Umgebung nie.
5. **Plottet die Lernkurven** in `outputs/cql.png`.

---

## Was Sie sehen sollten

Ein typischer Durchlauf gibt etwa Folgendes aus:

```
Final evaluation returns (avg over 10 episodes, greedy):
  naive offline DQN (alpha=0)         ->  ~30-150  (instabil; stürzt oft ab)
  CQL (alpha=1.0)                     ->  ~300-450
  CQL (alpha=5.0)                     ->  ~450-500
```

Im Lernkurven-Diagramm:

- Die **rote Kurve** (`α = 0`) steigt früh an und **fällt dann oft steil ab**, sobald Halluzinationen durch Distribution Shift das **Bellman-Target** infizieren (die Zahl, die wir als „richtige Antwort“ beim Training des Q-Netzwerks verwenden: `r + γ · max Q(s', ·)`). Wenn Phantom-Q-Werte dieses Target verschmutzen, macht jeder Gradientenschritt die Sache schlimmer. Der **Bellman-Loss** (der MSE zwischen der Vorhersage des Q-Netzwerks und dem Bellman-Target) sieht gut aus – das ist die **Tücke** des Problems: Das Netzwerk ist vollkommen konsistent mit seinen eigenen falschen Überzeugungen, sodass der Loss keine Warnung gibt.
- Die **orangefarbene Kurve** (`α = 1.0`) steigt langsamer an, **bleibt aber oben**.
- Die **grüne Kurve** (`α = 5.0`) ist am stabilsten und meistens am besten.

Das Bellman-Loss-Feld zeigt ein weiteres Indiz: Der Loss des naiven DQN kann klein bleiben, während seine Policy schrecklich ist, weil das Netzwerk intern konsistent mit seinen eigenen Halluzinationen ist.

---

## Wo CQL im Forschungsfeld steht

CQL war ein *großer* Durchbruch, da es eine fundierte, einfache Lösung für den Distribution Shift lieferte. Die Abstammungslinie:

```
DQN (online)
   │
   ▼
Naives offline DQN  ── scheitert wegen Distribution Shift
   │
   ▼
CQL (Kumar 2020)    ── fügt eine konservative Strafe hinzu: Q ist eine untere Schranke
   │
   ▼
IQL (Kostrikov 2021)  ── fragt Q-Werte für ungesehene Aktionen gar nicht erst ab
   │
   ▼
Decision Transformer (Chen 2021)  ── verzichtet ganz auf Q, behandelt RL als Sequenzmodellierung
                                      (sagt die *nächste Aktion* voraus, gegeben vergangene Zustände
                                       und einen gewünschten Gesamt-Return, genau wie ein LLM
                                       das nächste Wort vorhersagt)
```

Jeder Schritt in dieser Linie ist eine andere Antwort auf dieselbe Frage: **Wie vermeide ich es, mein Q-Netzwerk nach Dingen zu fragen, die es noch nie gesehen hat?**

---

## Wichtige Begriffe zum Merken

| Begriff | Bedeutung |
|------|---------|
| **Distribution Shift** | Die trainierte Policy wünscht sich Aktionen außerhalb der Daten |
| **Out-of-distribution (OOD)** | Ein (s, a)-Paar, das der Datensatz nie aufgezeichnet hat |
| **Wahres Q** | Der echte erwartete zukünftige Return für die Aktion `a` im Zustand `s`, wenn wir ihn perfekt messen könnten |
| **Konservatives Q** | Eine gelernte Q-Funktion, die versucht, unter dem wahren Q zu bleiben, anstatt zu viel zu versprechen |
| **Logsumexp** | Eine glatte, differenzierbare Annäherung an `max` |
| **Alpha (α)** | Regler für CQL-Konservatismus – wie stark Q-Werte bei OOD-Aktionen gesenkt werden |

---

## Zusammenfassung in einem Satz

> **CQL fügt eine „Pessimismus-Strafe“ hinzu, die hohe Q-Werte für Aktionen bestraft, die der Datensatz nie ausprobiert hat – so kann sich die Policy nicht in Halluzinationen verlieben.**
