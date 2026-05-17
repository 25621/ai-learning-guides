# SARSA vs Q-Learning: Sicherer vs. Optimaler Pfad 🐢 vs 🐇

## Was ist das?

Zwei Roboter müssen an einer **Klippe** entlanglaufen, um das Ziel zu erreichen. Beide Roboter befinden sich noch im *Lernprozess* und machen manchmal zufällige Bewegungen (Hoppla!).

- 🐢 **SARSA-Roboter**: "Ich weiß, dass ich manchmal wackele... also bleibe ich weit weg von der Klippe, um sicher zu sein, auch wenn es länger dauert."
- 🐇 **Q-Learning-Roboter**: "Der kürzeste Weg führt direkt an der Klippe entlang – los geht's! (Fällt während des Lernens manchmal runter, lernt aber schließlich die beste Route.)"

Beide Roboter sind intelligent, aber sie wählen einen **unterschiedlichen Kompromiss**: sicher-aber-langsamer vs. optimal-aber-riskant-während-des-Lernens.

---

## Der Hauptunterschied: Welche "nächste Aktion" wird verwendet?

Beim Aktualisieren der Bewertungen nach jedem Schritt fragen beide Algorithmen:
> "Was ist der Wert des *nächsten Zustands*?"

| Algorithmus | Verwendet die nächste Aktion... | On-policy? |
|-------------|--------------------------------|------------|
| **SARSA** | ...die ich *tatsächlich ausführen werde* (vielleicht zufällig!) | Ja |
| **Q-Learning** | ...die *theoretisch am besten* ist (immer gierig/greedy) | Nein |

**Beispiel aus dem echten Leben:** Zwei Kinder lernen Fahrradfahren.

- **SARSA-Kind**: Bleibt nah am Gras, weil es *weiß*, dass es manchmal zufällig wackelt. Es plant für sein tatsächliches, wackeliges Selbst.
- **Q-Learning-Kind**: Übt in der Mitte des Weges, weil es sich ein perfektes zukünftiges Selbst vorstellt, das niemals wackelt. Es fällt jetzt zwar manchmal hin, lernt aber schneller den besten Pfad.

Beide Kinder lernen schließlich – aber während des Trainings fällt das SARSA-Kind seltener hin!

---

## Was unser Code herausgefunden hat

Beide Algorithmen liefen für **500 Episoden** auf "Cliff Walking" mit ε=0,1 (ε = Epsilon; hier bedeutet es eine 10%ige Chance, eine zufällige Bewegung zu machen):

| Metrik | SARSA | Q-Learning |
|--------|-------|------------|
| Durchschn. Belohnung während des Trainings (letzte 50 Ep.) | **-19,7** | **-51,0** |
| Gierige (greedy) Bewertung (ohne Exploration) | -17 | **-13** |

- **Während des Trainings**: SARSA erhält **viel bessere Belohnungen**, weil es die Klippe vermeidet (indem es seine eigenen zufälligen Bewegungen einkalkuliert).
- **Nach dem Training** (rein gierig/greedy): Q-Learning findet den **kürzeren, optimalen Pfad** (-13)!

Wenn ε gegen 0 geht, konvergieren beide Algorithmen zur gleichen optimalen Policy.

---

## Visuelle Zusammenfassung

```
SARSA-Pfad (während des Trainings):     Q-Learning-Pfad (greedy, nach dem Training):
[ ][→][→][→][→][→][→][→][→][→][→][↓]   [ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][↓]
[↑][→][→][→][→][→][→][→][→][→][→][↓]   [ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][↓]
[ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][↓]   [ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][↓]
[S][C][C][C][C][C][C][C][C][C][C][G]   [S][→][→][→][→][→][→][→][→][→][→][G]
    (sicherer Umweg, obere Reihen)           (optimal, direkt an der Klippe)
```

---

## Beispiele aus dem echten Leben

- **Junger Chirurg vs. erfahrener Chirurg**: Ein junger Chirurg (SARSA) hält sich während des Lernens von riskanten Techniken fern. Ein erfahrener Chirurg (Q-Learning greedy) verwendet nach der Beherrschung die effizienteste Technik.
- **Stadtverkehr vs. Autobahnroute**: Eine SARSA-ähnliche Planung wählt sicherere Wohnstraßen; Q-Learning findet die optimale, aber schmale Autobahn.
- **Lernender Student**: Ein SARSA-Student hält sich während der Übung an gut verstandene Themen. Ein Q-Learning-Student wagt sich an die schwierigsten Probleme (scheitert öfter), lernt aber die optimale Strategie.

---

## Wichtige Begriffe

- **On-policy** (SARSA): Lernt über das, was man *tatsächlich tut*, einschließlich zufälliger Exploration.
- **Off-policy** (Q-Learning): Lernt über das *bestmögliche* Verhalten, getrennt von dem, was man tatsächlich tut.
- **Sicherer Pfad**: Längere Route, die Gefahren vermeidet; wird verwendet, wenn Exploration zu Unfällen führt.
- **Optimaler Pfad**: Kürzeste Route mit der höchsten Belohnung; wird gefunden, wenn keine Exploration stattfindet.
- **Exploration-Exploitation-Abwägung**: Das Gleichgewicht zwischen dem Ausprobieren neuer Dinge und der Nutzung des bereits Bekannten.

Die Kernbotschaft: **SARSA ist während des Trainings sicherer (on-policy), Q-Learning findet den optimalen Pfad schneller (off-policy). Was besser ist, hängt davon ab, ob es schlimm ist, von der Klippe zu stürzen!**
