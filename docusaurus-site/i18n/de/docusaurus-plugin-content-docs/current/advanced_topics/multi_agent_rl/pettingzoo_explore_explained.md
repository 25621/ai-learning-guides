# Erkundung von PettingZoo-Umgebungen 🦓

## Was ist PettingZoo?

Wenn Sie sich mit Single-Agent-RL beschäftigt haben, haben Sie wahrscheinlich **Gymnasium** (den Nachfolger von OpenAI Gym) verwendet. Jede Umgebung sieht dort gleich aus: `env.reset()`, `env.step(action) → obs, reward, done, info` — eine neue *Beobachtung* der Welt, ein skalarer *Belohnungswert*, ein *Done*-Flag, das „Spiel vorbei“ signalisiert, und ein *Info*-Dictionary für zusätzliche Debugging-Daten. Diese Einheitlichkeit ermöglicht es RL-Bibliotheken überhaupt erst zu funktionieren.

**PettingZoo** ist genau die gleiche Idee, aber für *mehrere Agenten*. Es ist ein „Zoo“ von Multiagenten-Umgebungen — alle hinter einer klar definierten API:
- **Klassische Spielzeugprobleme**: einfache Umgebungen wie Schere-Stein-Papier, um grundlegende Algorithmen zu testen.
- **Kooperative Gitterwelten**: Agenten navigieren durch ein Gitter, um ein gemeinsames Ziel zu erreichen.
- **Atari-Multiplayer**: klassische kompetitive Spiele wie Pong.
- **MPE (Multi-Particle Environment)**: Umgebungen mit physikbasierten kontinuierlichen Räumen für komplexe Koordination und Wettbewerb.

Wenn Sie Code schreiben können, der in einer PettingZoo-Umgebung funktioniert, können Sie ihn fast ohne Änderungen auf jede andere Umgebung anwenden.

---

## Die zwei API-Stile

Multiagenten-Szenarien sind komplizierter als Einzelschritt-Szenarien, da zwei Agenten gleichzeitig handeln können, oder nacheinander, oder sogar in völlig beliebiger Reihenfolge. PettingZoo löst dies mit zwei parallelen APIs:

### 1) AEC (Agent-Environment-Cycle)

Es handelt immer nur ein Agent gleichzeitig. Die Umgebung durchläuft die Agenten in einer bestimmten Reihenfolge, und jeder erhält:
- eine **Beobachtung (Observation)** — was er *genau jetzt* sieht,
- eine **Belohnung (Reward)** — den Gewinn, der durch die *gemeinsame* Aktion in der letzten vollständigen Runde erzielt wurde (d. h. was als Ergebnis *aller* handelnden Agenten geschah, nicht nur durch dich; beim Schach zum Beispiel spiegelt deine Belohnung den Zustand des Brettes nach dem letzten Zug deines Gegners wider, nicht nur nach deinem eigenen),
- ein **Termination-Flag** — `True`, wenn die Episode *natürlich* endet (z. B. Schachmatt, jemand gewinnt),
- ein **Truncation-Flag** — `True`, wenn die Episode durch ein Zeitlimit *vorzeitig abgebrochen* wird, bevor ein natürliches Ende erreicht ist.

Dies ist natürlich für **zugbasierte Spiele** wie Schach, Go oder Poker.

```python
env.reset()
for agent in env.agent_iter():
    obs, reward, term, trunc, info = env.last()
    if term or trunc:
        env.step(None)
        continue
    action = my_policy(obs, agent)
    env.step(action)
```

### 2) Parallel

Alle Agenten beobachten und handeln in jedem Schritt gleichzeitig. `step()` nimmt ein *Dictionary* von Aktionen entgegen und gibt Dictionaries von Beobachtungen und Belohnungen zurück.

Dies ist natürlich für **Echtzeitspiele** wie MPE (Multi-Particle Environments, in denen sich alle Punkt-Agenten gleichzeitig bewegen) oder Multiagenten-Gitterwelten.

```python
obs, info = env.reset()
while env.agents:
    actions = {a: my_policy(obs[a]) for a in env.agents}
    obs, rewards, terms, truncs, info = env.step(actions)
```

Die beiden Stile sind **isomorph** — strukturell gleichwertig und ineinander umwandelbar: Jede AEC-Umgebung kann automatisch so verpackt (wrapped) werden, dass sie wie eine parallele aussieht und umgekehrt. PettingZoo liefert diese Konvertierungs-Wrapper mit, sodass Sie nur Code für einen Stil schreiben müssen.

---

## Analogie aus dem echten Leben

- **AEC = ein Brettspielabend.** „Alice ist dran. Jetzt Bob. Jetzt Carol. Zurück zu Alice.“ Wer als Nächstes zieht, sieht den aktuellen Zustand des Spielbretts.
- **Parallel = ein Multiplayer-Videospiel.** Alle vier Spieler drücken gleichzeitig Tasten; das Spiel aktualisiert die Welt 60-mal pro Sekunde.
- **Warum einheitliche APIs wichtig sind.** Stellen Sie sich vor, jedes Multiplayer-Videospiel bräuchte seinen eigenen Joystick. PettingZoo ist der „universelle Joystick“ des MARL.

---

## Was unser Code tut

Wir bauen eine Umgebung im **PettingZoo-Stil** von Grund auf neu: das **Iterated Coordination Game**. Zwei Agenten wählen wiederholt den Kanal `0` oder `1`:

- Gleiche Wahl → beide erhalten +1
- Unterschiedliche Wahl → beide erhalten -1

Die **Beobachtung**, die jeder Agent erhält, ist die vorherige *gemeinsame Aktion* — was beide Agenten in der letzten Runde gewählt haben, verpackt in eine einzelne Ganzzahl. Konkret: Die letzte Aktion jedes Agenten ist eine aus `{Start, 0, 1}` (3 Zustände), sodass das Paar als `3 × agent_1_zustand + agent_2_zustand` kodiert wird, was 9 mögliche Ganzzahlen ergibt (0 – 8). Die Ganzzahl 0 ist der „Start“-Zustand — sie signalisiert, dass noch keine Aktion ausgeführt wurde (der Beginn einer Episode). Eine Episode dauert 25 Schritte, sodass der maximale Gesamt-Return +25 pro Agent und der minimale -25 beträgt. **Zufälliges Spiel erzielt ≈ 0**, da bei jedem Schritt zwei unabhängige Zufallsagenten jeweils 0 oder 1 mit gleicher Wahrscheinlichkeit wählen: Sie stimmen in 50 % der Fälle überein (+1) und unterscheiden sich in 50 % der Fälle (-1), was eine erwartete Belohnung pro Schritt von 0,5 × (+1) + 0,5 × (-1) = **0** ergibt. Über 25 Schritte summiert ist der erwartete Episoden-Return ebenfalls 0.

Wir führen dann Folgendes aus:

1. **Demonstration der AEC-Schnittstelle** mit einem zufälligen Durchlauf — dies bestätigt die grundlegende AEC-Schleife: `agent_iter()` liefert den Agenten, der an der Reihe ist, `last()` liest dessen aktuelle Beobachtung und angesammelte Belohnung, und `step()` gibt die gewählte Aktion an die Umgebung zurück.
2. **Training zweier unabhängiger Q-Learner über die Parallel-Schnittstelle**. Jeder Agent führt seine eigene Q-Tabelle, die über die **Beobachtung der gemeinsamen Aktion** indexiert ist, sodass er lernen kann: „Wenn wir beim letzten Mal beide 0 gewählt haben, sollte ich wieder 0 wählen.“
3. **Versuch, die echte `pettingzoo`-Bibliothek zu importieren** und eine ihrer eingebauten Umgebungen (Schere-Stein-Papier) mit einer Zufallsstrategie auszuführen. Falls PettingZoo nicht installiert ist, überspringen wir diesen Schritt mit einer freundlichen Nachricht.

### Was Sie sehen sollten

| Phase | Erwartung |
|-------|----------|
| Zufälliger Durchlauf (AEC) | Mittlerer Episoden-Return nahe **0** — Zufallsagenten wählen Kanäle unabhängig voneinander. |
| Unabhängige Q-Learner (Parallel) — erste 100 Episoden | Etwa **0** — noch weitgehend zufällig, während die Agenten explorieren. |
| Unabhängige Q-Learner — letzte 100 Episoden | Deutlich positiv, **+20 bis +25** — **Koordination ist entstanden**: Beide Agenten haben gelernt, in jeder Runde zuverlässig denselben Kanal zu wählen. |

Das Diagramm `outputs/pettingzoo_coordination.png` zeigt die Returns der einzelnen Episoden (grau) und eine gleitende Durchschnittskurve (**Mean**, blau). Der Durchschnitt glättet verrauschte Episoden, sodass Sie den Trend sehen können: Die Agenten bewegen sich von unkoordiniertem Zufallsspiel nahe ~0 hin zu stabiler **Koordination** nahe ~+25. Die gestrichelte grüne Linie markiert die Obergrenze der perfekten Koordination.

---

## Warum zuerst eine eigene Umgebung bauen?

Weil **die API die Lektion ist.** (Zu verstehen, wie man die Interaktion zwischen mehreren Agenten und der Umgebung strukturiert, ist wichtiger als die spezifischen Spielregeln.) Multiagenten-RL gibt es in vielen Varianten (zugbasiert, Echtzeit, kooperativ, kompetitiv, gemischt), und sie alle passen in das AEC- / Parallel-Muster. Sobald Sie diese beiden Schleifen implementiert haben, ist jede PettingZoo-Umgebung nur noch eine Frage des Einsetzens eines anderen Konstruktors — der Trainingscode bleibt derselbe.

Genau so hat Gymnasium das Single-Agent-RL verändert: Indem es die Umgebung zu einer Black Box hinter einer einheitlichen Schnittstelle gemacht hat.

---

## Wo unabhängiges Q-Learning hilft und wo es schadet

Koordinationsspiele sind *nachsichtig* — die Agenten teilen sich das Belohnungssignal, sodass ihre Interessen gleichgerichtet sind. Unabhängige Lerner können dies problemlos lösen, da jede Verbesserung eines Agenten auch dem anderen hilft.

In **adversarialen** (gegnerischen) Spielen (wie Schere-Stein-Papier) oszilliert unabhängiges Q-Learning ewig (wenn sich ein Agent anpasst, ändert der andere seine Strategie, um gegenzusteuern, was zu einer endlosen Jagd führt). In Spielen mit **unvollständiger Beobachtbarkeit** kann es gar nicht lernen, da die „Beobachtung“ nur ein Teil des Zustands ist. PettingZoo enthält beide Arten von Umgebungen, damit Sie diese Fehlermodi selbst sehen können.

---

## Wichtige Begriffe zum Merken

| Begriff | Bedeutung |
|------|---------|
| **PettingZoo** | Das Gymnasium des Multiagenten-RL — eine Bibliothek standardisierter MARL-Umgebungen |
| **AEC** | Agent-Environment-Cycle: Ein Agent handelt pro Schritt (zugbasiert) |
| **Parallele API** | Alle Agenten handeln gleichzeitig in jedem Schritt |
| **MPE** | Multi-Particle Environment, eine beliebte kooperative/kompetitive Testumgebung in PettingZoo (oft mit beweglichen Punkten). |
| **CTDE** | Centralised Training, Decentralised Execution — Trainieren mit globaler Sicht (Zugriff auf alle Zustände), Einsatz mit nur lokalen Beobachtungen. |
| **Unabhängiges Q-Learning** | Jeder Agent lässt ein Standard-Q-Learning laufen und ignoriert die Existenz anderer Lerner. |

---

## Zusammenfassung in einem Satz

> **PettingZoo verleiht jeder Multiagenten-Umgebung die gleiche Form — sodass der Code, den Sie heute schreiben, morgen auch für ein völlig anderes Spiel funktioniert.**

Sobald Ihnen die beiden API-Stile in Fleisch und Blut übergegangen sind, können Sie zu MADDPG (zentralisierter Critic für Continuous-Control-Agenten), QMIX (Value-Mixing für kooperative Teams), MAPPO (Multiagenten-PPO) oder jedem anderen modernen MARL-Algorithmus übergehen — die Umgebungsseite Ihres Codes muss sich nie wieder ändern.
