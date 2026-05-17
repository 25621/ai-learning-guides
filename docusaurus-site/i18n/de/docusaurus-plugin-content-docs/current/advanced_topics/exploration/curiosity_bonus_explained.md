# Curiosity-Bonus (Intrinsische Motivation) 🧭

## Was ist das?

Stellen Sie sich ein Kleinkind vor, das in ein neues Zimmer gesetzt wird. Niemand bezahlt es, niemand klatscht – und doch steuert es zielsicher auf den Schrank zu, den es noch nicht geöffnet hat, auf den Knopf, den es noch nicht gedrückt hat, auf das laute Spielzeug in der Ecke. Es wird von einer **internen Belohnung** angetrieben: *„Das sieht neu aus. Schau es dir an.“*

Ein **Curiosity-Bonus** (Neugier-Bonus) verleiht einem Reinforcement-Learning-Agenten denselben internen Antrieb. Die reale Belohnung der Umgebung (die „extrinsische“ Belohnung – Punkte, Geld, das Spiel gewinnen) bleibt genau so, wie sie ist. Wir fügen lediglich eine zweite, selbst generierte Belohnung für den Besuch von Dingen hinzu, die der Agent als *neu* oder *überraschend* empfindet, und trainieren auf der Summe:

```
Belohnung, von der der Agent lernt = reale Belohnung + Beta * Curiosity-Bonus
```

`Beta` ist ein Regler, der groß beginnt (sei neugierig!) und mit der Zeit schrumpft (hör auf zu trödeln, nutze das Gelernte aus).

## Warum der Aufwand? Das Problem der spärlichen Belohnung (Sparse Rewards)

Normale RL-Agenten lernen aus Belohnungen, die sie tatsächlich erhalten. Das funktioniert hervorragend, wenn Belohnungen überall zu finden sind („+1 für jeden Schritt, den du aufrecht stehst“ in CartPole). Es scheitert jedoch, wenn die Belohnung **spärlich** (sparse) ist – Null, Null, Null, ..., Null, und dann schließlich ein +1 nach einer langen, sehr spezifischen Sequenz korrekter Aktionen.

Echte Beispiele für spärliche Belohnungen:

- **Montezuma's Revenge** (das Atari-Spiel): Ihr erster Punkt erscheint erst nach etwa 100 präzisen Bewegungen – eine Leiter hinunterklettern, einem Totenkopf ausweichen, wieder hinaufklettern, einen Schlüssel greifen. Bis dahin bleibt der Punktestand eine flache Null.
- **Ein Zahlenschloss.** 9.999 falsche Codes bringen Ihnen nichts; einer bringt den Preis.
- **Wirkstoffsuche / wissenschaftliche Experimente.** Tausende fehlgeschlagene Versuche, dann einer, der funktioniert.
- **Schreiben eines langen Beweises oder Programms.** Keine Teilpunkte, bis das Ganze stimmt.

Ein Agent, der nur auf Belohnungen reagiert, ist in diesen Situationen wie ein Student, der sich weigert zu lernen, es sei denn, er wird pro richtiger Antwort in der Abschlussprüfung bezahlt – er fängt nie an. Neugier ist der Bonus, der besagt: *„Erforschen ist seine eigene Belohnung“*, sodass der Agent so lange herumstochert, bis er über den echten Preis stolpert.

## Zwei Arten von Neugier (beide in `curiosity_bonus.py` implementiert)

### 1. Zählbasierte Neuheit: „Ich war hier kaum“

Das einfachste mögliche Neuheitssignal. Führen Sie eine Liste `N(s, a)` darüber, wie oft Sie die Aktion `a` im Zustand `s` ausgeführt haben, und geben Sie sich selbst einen Bonus, der schrumpft, je größer diese Zahl wird:

```
Curiosity-Bonus = 1 / sqrt( N(s, a) + 1 )
```

Beim ersten Mal, wenn Sie etwas ausprobieren: Bonus = 1,0. Nach 100 Versuchen: Bonus = 0,1. Nach 10.000 Versuchen: 0,01. Der Agent wird dafür belohnt, dorthin zu gehen, wo er noch nicht war, und die Verlockung verblasst natürlich auf ausgetretenen Pfaden.

**Analogie aus dem echten Leben:** Ein Tourist mit einer Liste „Orte, die ich noch nicht besucht habe“. Brandneues Viertel? Oberste Priorität. Das Café, in dem man schon fünfzig Mal war? Nicht mehr aufregend.

Dies ist eine der ältesten Ideen überhaupt (MBIE-EB, UCB). Ihre Schwäche: In einer riesigen oder kontinuierlichen Welt besucht man nie *exakt* denselben Zustand zweimal, sodass die rohe Zählung immer 1 ist – weshalb die nächste Variante existiert.

### 2. Vorhersagefehler-Neuheit: „Das habe ich nicht kommen sehen“

Dies ist die Idee hinter dem berühmten **ICM** (Intrinsic Curiosity Module, Pathak et al. 2017) und seinem Verwandten **RND** (Random Network Distillation, Burda et al. 2018). Anstatt zu zählen, führt der Agent ein kleines **Modell, das versucht vorherzusagen, was als Nächstes passiert** – „wenn ich hier bin und dies tue, wo lande ich?“ – und belohnt sich selbst damit, **wie falsch das Modell lag**:

```
Curiosity-Bonus = Überraschung = -log P( der Zustand, den ich tatsächlich erreicht habe | wo ich war, was ich tat )
```

- Eine Situation, die das Modell noch nie gesehen hat → es sagt schlecht voraus → große Überraschung → großer Bonus → „geh dort erforschen!“
- Eine Situation, die das Modell hundertmal gesehen hat → es sagt perfekt voraus → null Überraschung → null Bonus → „schon gesehen, verstanden, weiter geht's.“

**Analogie aus dem echten Leben:** Ein Kind, das durch Spielen lernt, wie die Welt funktioniert. Ein Glas zum *ersten* Mal vom Tisch zu stoßen, ist faszinierend (es ist zerbrochen!). Beim hundertsten Mal wusste man bereits, dass es zerbrechen würde – nicht mehr interessant. Neugier = die Lücke zwischen dem, was man erwartet hat, und dem, was passiert ist.

In unserem tabellarischen Code ist das „Modell“ nur eine Tabelle mit Übergangshäufigkeiten, und „wie falsch es lag“ ist die Überraschung (Surprisal) `-log P`. Echtes ICM/RND verwenden neuronale Netzwerke, damit dieselbe Idee auf rohen Pixeln funktioniert – aber das Prinzip ist identisch.

> **Warum zwei Versionen?** Zählbasiert ist extrem simpel und ein großartiger Baseline-Wert. Vorhersagefehler skaliert auf große Welten, die sich nie exakt wiederholen, und liefert ein *schärferes* Signal: In einer deterministischen Umgebung sinkt die Überraschung sofort auf ~0, sobald man einen Übergang einmal gesehen hat, während ein Zählbonus nur langsam als `1/sqrt(N)` verblasst. In unseren Experimenten löst der Agent mit Vorhersagefehler MiniMontezuma in ein paar Dutzend Episoden; der zählbasierte Agent schafft es auch, nur langsamer und weniger zuverlässig.

## Was unser Code tut

`curiosity_bonus.py` trainiert einen einfachen **tabellarischen Q-Learner** auf `MiniMontezumaEnv` – einer winzigen Zwei-Zimmer-Gridworld, in der man zu einem **Schlüssel** gehen, ihn aufheben (jetzt öffnet sich die **Tür**), hindurchgehen und den **Schatz** erreichen muss. Die Belohnung (+1) erscheint *nur* beim Schatz, nach etwa 15 perfekten Zügen. Es werden drei Agenten ausgeführt und verglichen:

| Agent | Verhalten in MiniMontezuma |
|-------|-------------------------------|
| **Epsilon-Greedy (keine Neugier)** | Wandert in der Nähe des Starts umher, erreicht *nie* den Schlüssel, der Punktestand bleibt ewig bei 0. |
| **Zählbasierter Bonus** | Findet zuverlässig den Schlüssel; schafft die gesamte Kette zum Schatz in etwa 40 % der Episoden. Funktioniert – ist nur etwas instabil. |
| **Vorhersagefehler-Bonus** | Erreicht den Schlüssel *und* den Schatz zum ersten Mal innerhalb von ~20–25 Episoden; wenn `Beta` abnimmt, konvergiert er dahin, die Aufgabe in jeder Episode zu lösen. |

Die Abbildung zeigt:
- eine Lernkurve: *Wahrscheinlichkeit, den Schatz zu erreichen* im Verlauf des Trainings,
- eine zweite Kurve für den Zwischenschritt *Wahrscheinlichkeit, den Schlüssel aufzuheben*,
- und **Heatmaps der Zustandsbesuche** im Gitter – der Agent ohne Neugier ist ein dichter Klumpen in der Nähe des Starts; die neugierigen Agenten fluten *beide* Räume.

## Der Mechanismus in einem Bild

```
            keine Neugier                      mit Curiosity-Bonus
   Belohnung:  0 0 0 0 0 0 0 0 ... 0  (+1?)     0 0 0 0 0 0 0 0 ... 0  (+1!)
               └──── nichts zu lernen ──┘       └ + 0,4 0,3 0,9 0,2 ... ┘  (selbst gemacht,
                                                  dicht, zeigt „Richtung Neues“)
   Ergebnis:   Random Walk, findet nie +1       systematisches Durchkämmen der Welt,
                                                 stolpert über +1, dann verblasst der Bonus
```

Der Curiosity-Bonus verwandelt „Das habe ich noch nicht gesehen“ in Belohnung, sodass der Agent **gezielt in unerforschtes Territorium vordringt**, anstatt zufällig herumzuflattern. Und da der Bonus schrumpft, wenn Dinge vertraut werden (und `Beta` abnimmt), hört der Agent, sobald er die echte Belohnung gefunden hat, natürlich auf zu trödeln und beginnt mit der Exploitation (Ausnutzung).

## Ein paar ehrliche Einschränkungen

- **Das „Noisy-TV“-Problem.** Ein Agent mit Vorhersagefehler-Neugier kann von einer Quelle reiner Zufälligkeit (ein Fernseher, der Rauschen zeigt, würfelnde Würfel) hypnotisiert werden – er kann sie *nie* vorhersagen, also verblasst die Überraschung nie. Der eigentliche Trick von ICM besteht darin, in einem *gelernten Merkmalsraum* (Feature Space) vorherzusagen, der Dinge ignoriert, die der Agent nicht kontrollieren kann. RND umgeht dies auf andere Weise. Unsere deterministische Gridworld hat kein solches Rauschen, daher tritt dies hier nicht auf.
- **Neugier ist ein Mittel, kein Zweck.** Deshalb nimmt `Beta` ab. Ein Agent, der ewig maximal neugierig bleibt, kommt nie zur Ruhe, um tatsächlich zu *gewinnen*.
- **Die Skalierung tiefer Exploration ist immer noch schwer.** Ein Bonus in der Belohnung hilft, aber einfaches tabellarisches Q-Learning ist zu langsam, um den resultierenden Optimismus eine lange Kette hinunterzuschieben (siehe `compare_exploration.py`). Um das echte Montezuma mit Pixeln zu knacken, war zusätzliche Feuerkraft nötig – RND mit einem neuronalen Netz, bootstrapped DQN, Go-Explore.

## Wichtige Begriffe zum Merken

| Begriff | Bedeutung |
|------|---------|
| **Intrinsische Belohnung** | Eine Belohnung, die der Agent für sich selbst generiert, getrennt von der Belohnung der Umgebung |
| **Extrinsische Belohnung** | Die reale Belohnung der Umgebung (Punkte, Sieg/Niederlage) |
| **Spärliche Belohnung** | Belohnung ist fast überall Null; man erhält sie erst nach einer langen korrekten Sequenz |
| **Neuheit / Überraschung** | Wie neu oder unerwartet ein Zustand (oder Übergang) ist – das, was Neugier belohnt |
| **Zählbasierter Bonus** | Neuheit ≈ `1/sqrt(Besuchsanzahl)` – der klassische Explorationsbonus |
| **ICM** | Intrinsic Curiosity Module: Neuheit = Vorhersagefehler eines Forward-Modells (in einem gelernten Merkmalsraum) |
| **`Beta`** | Die Gewichtung des Curiosity-Bonus; wird normalerweise gegen 0 reduziert, damit der Agent schließlich ausnutzt (exploits) |

---

## Zusammenfassung in einem Satz

> **Ein Curiosity-Bonus ist eine selbst gegebene Belohnung für Neuheit – er erzeugt ein dichtes Signal nach dem Motto „geh dort drüben erforschen“, das den Agenten durch Welten mit spärlichen Belohnungen zieht, die er sonst nie lösen würde, und verblasst dann höflich, sobald alles vertraut ist.**
