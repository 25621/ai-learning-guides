# Training auf Montezuma's Revenge 🏛️🔑

## Warum dieses Spiel berühmt ist (in RL-Kreisen)

Im Jahr 2015 lernte das DQN von DeepMind, Dutzende von Atari-Spielen auf übermenschlichem Niveau allein anhand von rohen Pixeln zu spielen. Das sorgte für Schlagzeilen. Aber in der Ergebnistabelle versteckt war ein Spiel, bei dem DQN **0** Punkte erzielte – genauso viel, als würde man gar nichts tun: **Montezuma's Revenge**.

Warum? Schauen Sie sich an, was das Spiel bereits im allerersten Raum von Ihnen verlangt:

1. Eine Leiter *hinunterklettern*.
2. Über einen Sims gehen.
3. Über einen rollenden Totenkopf springen (falsches Timing → du stirbst).
4. Eine weitere Leiter *hinaufklettern*.
5. Den Schlüssel greifen.

Das sind etwa **100 präzise Tastendrücke**, und das Spiel gibt Ihnen **keinen einzigen Punkt**, bis Sie den Schlüssel in der Hand halten. Das Belohnungssignal ist eine flache, merkmalslose **Null** für die gesamte Eröffnungssequenz.

Ein normaler RL-Agent lernt, indem er sich an Belohnungen anpasst, die er tatsächlich erhält. Wenn die Belohnung überall dort, wo er hinkommen kann, Null ist, gibt es *nichts, woraus er lernen könnte* – es ist, als würde man versuchen, den Boden eines perfekt flachen Tal zu finden, indem man nach der Richtung des Gefälles tastet. So zuckte DQN einfach ewig auf der Startplattform herum. Montezuma wurde *der* Benchmark für **Hard Exploration**: das Spiel, das man nur schlagen kann, wenn man *geschickt* exploriert, nicht zufällig.

Der Durchbruch gelang 2018 mit **Random Network Distillation (RND)** – und der Trick war genau das Thema von Aufgabe 1: Füge einen **intrinsischen Curiosity-Bonus** hinzu, damit der Agent sich *selbst* belohnt, wenn er neue Bildschirme erreicht. Plötzlich hat er ein dichtes Signal, das ihn tiefer in das Level zieht. RND erzielte einen übermenschlichen Score bei Montezuma. (Später folgten: Go-Explore, Agent57, …)

## Beispiele aus dem echten Leben für „Montezuma-artige“ spärliche Belohnungen

- **Ein Zahlenschloss / eine Schatzsuche mit kryptischen Hinweisen.** Keine Teilpunkte. Du stehst bei Null, bis du plötzlich beim Preis bist.
- **Die Annahme einer wissenschaftlichen Arbeit oder ein Startup, das profitabel wird.** Monate ohne externe Belohnung, dann (vielleicht) eine große.
- **Eine Speedrun-Route in einem Videospiel.** Dutzende von Frame-perfekten Eingaben hintereinander ohne Feedback, bis der Trick entweder funktioniert oder nicht.
- **Escape Rooms.** Der Raum verrät einem fast nichts, bis man mehrere Entdeckungen miteinander verknüpft hat.

In all diesen Fällen ist „einfach mal zufällig Zeug ausprobieren“ aussichtslos. Man muss *systematisch* explorieren – und ein internes „Oha, das ist neu, weiter geht's“-Signal ist das, was einen systematisch hält.

## Warum wir hier nicht wirklich auf dem Pixel-Montezuma trainieren

Um die *echte* Sache richtig zu machen, bräuchte man:

- ein Convolutional Network, um den 210×160 RGB-Bildschirm zu sehen,
- Frame-Stacking (damit der Agent erkennen kann, in welche Richtung sich der Totenkopf bewegt),
- ein RND-Modul (zwei weitere Netzwerke: ein festes zufälliges „Target“ und ein trainierter „Predictor“),
- und **zehn Millionen Umgebungs-Frames** – viele GPU-Stunden.

Das ist ein Forschungsprojekt, kein Lernskript. Deshalb macht `montezuma_revenge.py` stattdessen zwei ehrliche Dinge:

### 1. Es „berührt“ das echte Spiel (falls `ale-py` installiert ist)

Es lädt `ALE/MontezumaRevenge-v5` über Gymnasium, lässt einen **gleichmäßig zufälligen Agenten für 2000 Schritte** laufen und berichtet die Gesamtbelohnung. Die Zahl, die ausgegeben wird, ist fast immer **0.0** – der abstrakte Begriff „spärliche Belohnung“ wird so zu einer konkreten, selbst erlebbaren Tatsache. Falls das Atari-Paket nicht installiert ist, gibt es den `pip install`-Befehl aus und macht weiter.

### 2. Es trainiert einen tabellarischen Agenten auf einem *Modell*: `MiniMontezumaEnv`

Dies ist eine winzige Gridworld mit demselben *Skelett* wie Montezumas erster Raum:

```
###############
#S....#.......#
#.....#.......#
#.....#...T...#     S = Start
#.....D.......#     K = Key (Schlüssel)   D = Door (Tür, nur mit Schlüssel passierbar)
#..K..#.......#     T = Treasure (Schatz - das EINZIGE Feld mit Belohnung)
###############
```

Um zu gewinnen, müssen Sie: zum **Schlüssel** gehen (~6 Züge), ihn aufheben; zur **Tür** gehen (~4 Züge) — die sich nun öffnet; hindurchgehen und den **Schatz** erreichen (~5 Züge). Etwa **15 perfekte Züge**, mit **null Feedback bis zum Schatz**. Das `has_key`-Flag ist Teil des Zustands des Agenten, sobald man also den Schlüssel greift, gibt es einen ganzen zweiten Raum mit *neuen* Zuständen zu entdecken – genau wie sich im echten Spiel neue Bildschirme öffnen.

Wir trainieren dann zweimal einen einfachen **tabellarischen Q-Learner**:

| Agent | Ergebnis in MiniMontezuma |
|-------|--------------------------|
| **ohne Neugier (Epsilon-Greedy)** | Der Return bleibt über alle 1.500 Episoden bei **0**. Er erreicht nicht einmal den Schlüssel. (Kommt Ihnen das bekannt vor? Das ist DQN im echten Spiel.) |
| **mit Vorhersagefehler-Curiosity-Bonus** | Erreicht den Schatz innerhalb von ~20–25 Episoden und lernt dann die **optimale 15-Schritte-Route**. (Das ist die RND-Idee, geschrumpft auf eine Q-Tabelle.) |

Die Abbildung zeigt die beiden Lernkurven nebeneinander, plus die tatsächliche Route, die der neugierige Agent gelernt hat, eingezeichnet in das Gitter (Start → Schlüssel → Tür → Schatz). Das Skript gibt diese Route auch als ASCII-Frames aus.

## Die Lektion

> **„Spärliche Belohnung“ (Sparse Reward) ist keine Eigenheit eines seltsamen Atari-Spiels – es ist der Standard in jeder Welt, in der Erfolg eine lange, spezifische Abfolge von Aktionen erfordert.** Ein Agent, der nur auf Belohnungen reagiert (Vanilla DQN), kann buchstäblich nicht anfangen: Es gibt keinen Gradienten, dem er folgen könnte. Ein Curiosity-Bonus erzeugt einen – ein dichtes, selbst generiertes „das ist neu, weiter geht's“-Signal – und dieses Signal trägt den Agenten über die Wüste aus Nullen hin zur ersten echten Belohnung. Alles danach (RND, Go-Explore, Agent57) ist eine hochskalierte neuronale Netzwerk-Version desselben Prinzips.

## Wichtige Begriffe zum Merken

| Begriff | Bedeutung |
|------|---------|
| **Hard Exploration** | Probleme, bei denen man nur durch geschickte Erkundung Erfolg hat; zufällige Exploration scheitert |
| **Sparse Reward** | Die Belohnung ist fast überall Null; man erhält sie erst nach einer langen korrekten Sequenz |
| **Montezuma's Revenge** | Das Atari-Spiel, bei dem klassische Deep-RL-Agenten (DQN, A3C) 0 Punkte erzielten – der klassische Hard-Exploration-Benchmark |
| **RND (Random Network Distillation)** | Die Methode von 2018, die Montezuma mit einem Vorhersagefehler-Curiosity-Bonus schlug |
| **Go-Explore** | „Vielversprechende Zustände merken, zu ihnen zurückkehren und von dort aus weiter explorieren“ — ein weiterer Montezuma-Knacker |
| **Maßstabsgetreues Modell (Scale Model)** | Eine kleine, kostengünstige Umgebung, die die *Struktur* eines schwierigen Problems beibehält, um es schnell untersuchen zu können |

---

## Zusammenfassung in einem Satz

> **Montezuma's Revenge ist das Spiel, das RL lehrte: „Belohnungen, die du nie erhältst, können dir nichts beibringen“ – und die Lösung, damals wie heute, ist ein Curiosity-Bonus, der es dem Agenten ermöglicht, sich selbst für das Explorieren zu belohnen, bis er den echten Preis findet.**
