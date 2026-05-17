# Training eines Weltmodells: Dem Agenten das Träumen beibringen 🌍

## Was ist ein "Weltmodell"?

Ein **Weltmodell** ist die *interne Kopie des Universums* des Agenten. Geben Sie ihm einen Zustand und eine Aktion, und es sagt voraus, was als Nächstes passieren wird:

```
(Zustand, Aktion)  ──►  Neuronales Netz  ──►  (nächster_Zustand, Belohnung)
```

Es ist nicht die reale Welt – es ist ein **Simulator, den der Agent für sich selbst gebaut hat**, indem er die Realität beobachtet und gelernt hat, sie nachzuahmen.

Sobald es trainiert ist, erlaubt das Modell dem Agenten, "Was-wäre-wenn"-Fragen zu stellen, ohne eine echte Aktion auszuführen:

> *"Wenn ich jetzt nach links drücke und dann zweimal nach rechts, wo lande ich dann? Wird der Stab umfallen?"*

Der Agent kann in seinem Modell hundert Pläne durchdenken in der Zeit, die er für einen einzigen echten Zug bräuchte. Das ist der ganze Sinn.

---

## Eine Analogie aus dem echten Leben

Denken Sie darüber nach, wie *Sie* ein Puzzle lösen. Sie bewegen nicht physisch jedes Teil in jede Lücke. Sie **stellen sich vor**, was passiert, wenn Teil A hierhin kommt. Wenn diese mentale Simulation falsch aussieht, verwerfen Sie sie, bevor Sie einen Finger rühren.

Ihr Gehirn hat ein gelerntes Weltmodell – aufgebaut aus Jahren, in denen Sie gesehen haben, wie sich Objekte verhalten –, das es Ihnen ermöglicht, Ergebnisse zu simulieren, bevor Sie sich festlegen.

Andere Beispiele:

- **Ein Schachspieler** stellt sich Züge mehrere Runden im Voraus vor.
- **Ein Autofahrer** denkt: "Wenn ich jetzt bremse, hat das Auto dahinter genug Platz."
- **Ein Kind**, das Klötze stapelt: "Wenn ich den großen nach oben lege, wird der Turm wackeln." (Es hat dieses Modell gelernt, indem es zuvor Türme umgeworfen hat.)

In jedem Fall gilt: **Ein mentales Modell + Vorstellungskraft = bessere Entscheidungen mit weniger Risiko.**

---

## Wie baut der Agent sein Modell auf?

Er **beobachtet** einfach. Speziell:

1. **Daten sammeln.** Lassen Sie eine beliebige Policy (sogar eine zufällige) eine Zeit lang mit der echten Umgebung interagieren. Speichern Sie jeden Übergang:
   ```
   (Zustand, Aktion, Belohnung, nächster_Zustand)
   ```
2. **Trainieren eines neuronalen Netzes**, um den `nächsten_Zustand` und die `Belohnung` aus `(Zustand, Aktion)` vorherzusagen. Dies ist überwachtes Lernen (Supervised Learning): Jeder gespeicherte Übergang ist ein beschriftetes Beispiel, bei dem die Eingabe "was der Agent sah und tat" und das Label "was tatsächlich als Nächstes geschah" ist.
3. **Validieren.** Behalten Sie 10% der Daten zurück und vergleichen Sie die Vorhersagen des Modells mit den realen Ergebnissen. Ein niedriger Fehler bedeutet, dass das Modell die **Dynamik** der Umgebung erfasst hat: wie sich Zustände nach Aktionen ändern.

Der Trick, den wir verwenden: Anstatt den `nächsten_Zustand` direkt vorherzusagen, sagen wir das **Delta** `nächster_Zustand − Zustand` voraus. Die meiste Physik ist inkrementell ("der Wagen bewegte sich ein kleines Stück"), und kleine Zielwerte sind einfacher für neuronale Netze.

---

## Unser Setup

| Wahl | Wert | Warum |
|--------|-------|-----|
| Umgebung | `CartPole-v1` | 4-D Zustand, 2 Aktionen – einfach zu modellieren |
| Daten | 20.000 Übergänge einer zufälligen Policy | Breite Abdeckung des Zustandsraums |
| Netzwerk | MLP, 2 × 128 ReLU hidden | MLP = Multi-Layer Perceptron (standardmäßiges neuronales Netz). Zwei versteckte Schichten mit 128 Neuronen und ReLU-Aktivierungen. Genug Kapazität, schnell zu trainieren. |
| Loss | MSE auf `(delta_state, reward)` | MSE = Mean Squared Error (durchschnittlicher quadratischer Vorhersagefehler). Standardmäßiger Regressionsverlust. |
| Optimizer | Adam, lr = 1e-3, 30 Epochen | Adam = adaptiver Optimierer (passt Lernraten pro Parameter automatisch an). Standardmäßig, kein spezielles Tuning erforderlich. |

Das gesamte Training ist in wenigen Sekunden auf der CPU abgeschlossen.

---

## Wie sieht "gut" aus?

Zwei Diagnosen sind wichtig:

### 1. Ein-Schritt-Genauigkeit (Validierungs-MSE)

Dies ist "wie gut sagt das Modell EINEN Schritt in die Zukunft voraus?" Nach 30 Epochen sollten Sie einen Validierungs-MSE im Bereich von **1e-4 bis 1e-3** sehen. Das ist winzig – Stabwinkel und Wagenpositionen sind bis auf wenige Dezimalstellen genau.

### 2. **Sich aufsummierender Fehler** bei k-Schritt-Rollouts

Dies ist der *echte* Test. Nehmen Sie einen Zustand, geben Sie ihn in das Modell ein, nehmen Sie dann dessen Vorhersage und geben Sie sie wieder in das Modell ein – für `k` Schritte hintereinander. Der Fehler wächst, weil jeder Schritt ein bisschen Rauschen über die vorherige Vorhersage legt.

```
Schritt  1:  L2-Fehler ≈ 0,01   (fast perfekt)
Schritt  5:  L2-Fehler ≈ 0,05
Schritt 10:  L2-Fehler ≈ 0,15
Schritt 20:  L2-Fehler ≈ 0,40   (sichtbares Abweichen)
```

*(L2-Fehler = Euklidischer Abstand zwischen dem vorhergesagten nächsten Zustand und dem echten – stellen Sie es sich als "wie weit liegt die Schätzung des Modells im 4-D-Zustandsraum daneben" vor.)*

**Warum das wichtig ist.** Wenn wir mit dem Modell 15 Schritte vorausplanen, wird der *exakte* Zustand bei Schritt 15 falsch sein – aber wenn das relative Ranking von "guten Plänen vs. schlechten Plänen" erhalten bleibt, funktioniert die Planung trotzdem. (Dies ist es, was `model_based_planning.py` ausnutzt.)

Das Diagramm in `outputs/world_model.png` zeigt beide Diagnosen nebeneinander: Die Trainingsverlustkurve geht auf einer logarithmischen Skala schön nach unten, und die Rollout-Fehlerkurve geht schön nach oben.

---

## Warum das *Delta* vorhersagen?

Vergleichen Sie zwei Arten, dem Netzwerk das gleiche Problem zu stellen:

| Zielwert | Typische Größe | Einfach oder schwer? |
|--------|------------------:|--------------|
| `nächster_Zustand`        | 0–2,4 (Wagenpos.) | Netzwerk muss Position **und** die winzige Änderung reproduzieren |
| `nächster_Zustand - Zustand`| ~0,02            | Netzwerk lernt nur die winzige Änderung |

Das Vorhersagen des Deltas bedeutet auch: Wenn das Netzwerk Nullen ausgibt (wie es ein untrainiertes Netzwerk oft tut), ist die Vorhersage einfach "nichts hat sich bewegt" – ein vernünftiger, sicherer Standard für einen einzelnen Zeitschritt. Im Gegensatz dazu würde das direkte Vorhersagen des absoluten `nächsten_Zustands` anfangs völlig zufällige Werte ausgeben, was das frühe Training sehr instabil machen würde.

---

## Was uns das bringt

Ein trainiertes Weltmodell ist die Grundlage für:

- **Planung** – Suche über vorgestellte Aktionsfolgen (siehe `model_based_planning.py`).
- **Dyna-Style-Erweiterung** – Trainieren eines Q-Netzwerks auf vorgestellten Daten, um die Samplen-Effizienz zu vervielfachen.
- **Neugier / Exploration** – Besuchen von Zuständen, die das Modell nicht gut vorhersagen kann.
- **Dreamer / World-Models Veröffentlichungen** – Trainieren einer *Policy* vollständig innerhalb des Modells mit Null Interaktion mit der realen Welt über die anfängliche Datenerfassung hinaus.

---

## Grenzen und Vorsichtsmaßnahmen

- **Abweichung außerhalb der Verteilung (Out-of-distribution drift).** Das Modell kennt nur den Teil der Welt, den es gesehen hat. Planen Sie zu aggressiv, und Sie landen in Regionen, die das Modell nie besucht hat – Vorhersagen dort sind reine Fantasie.
- **Sich aufsummierender Fehler.** Die Planung über lange **Horizonte** (viele Schritte in die Zukunft) ist aufgrund sich ansammelnder Fehler unzuverlässig, wie das Diagramm zeigt. Moderne Systeme lösen dies durch **probabilistische Ensembles** (Training mehrerer Modelle und Prüfung, ob sie übereinstimmen, wie in PETS oder Dreamer), damit der Planer genau weiß, *wie unsicher* das Modell bei jedem Schritt ist und riskante, unbekannte Pfade vermeiden kann.
- **Stochastische Umgebungen.** Ein standardmäßiger deterministischer Regressor sagt nur das *durchschnittliche* Ergebnis voraus und übersieht die Streuung möglicher Ergebnisse vollständig. Komplexe, reale Umgebungen erfordern probabilistische Modelle (wie solche mit Gaußschen Ausgaben oder **latente stochastische Modelle**), um Unsicherheit und Zufälligkeit genau darzustellen.

---

## Wichtige Begriffe

| Begriff | Einfaches Deutsch |
|------|---------------|
| **Weltmodell** | Ein neuronales Netz, das die Umgebung nachahmt |
| **Dynamik** | Die Funktion `(s, a) → s'` |
| **Belohnungsmodell** | Die Funktion `(s, a) → r` (oft im Paket enthalten) |
| **Ein-Schritt-Vorhersage** | Was das Modell aus einem realen Zustand ausgibt |
| **Rollout** | Wiederholte Ein-Schritt-Vorhersagen, bei denen Ausgaben wieder eingespeist werden |
| **Sich aufsummierender Fehler** | Kleine Fehler, die über einen Rollout wachsen |

---

## Zusammenfassung in einem Satz

> **Ein Weltmodell ist eine winzige neuronale Kopie des Universums, die der Agent konsultieren – und in der er träumen – kann, bevor er eine echte Aktion riskiert.**

Als Nächstes: `model_based_planning.py` setzt dieses Modell für die tatsächliche Entscheidungsfindung ein.
