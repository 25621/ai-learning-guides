# Ein gelerntes Modell für die Planung nutzen (MPC) 🔮

## Die Kernidee

Sie haben ein **Weltmodell** (ein neuronales Netzwerk, das die Zukunft vorhersagt). Was nun?

Die direkteste Anwendung ist die **Planung**: Fragen Sie das Modell in jedem Moment: „Was würde passieren, wenn ich *diesen* Plan ausprobierte? Oder *jenen*? Oder einen *völlig anderen*?“ Führen Sie dann den Plan aus, der am besten aussieht – aber **nur den allerersten Schritt davon**. Da das Modell nicht perfekt ist, führen wir nur eine Aktion aus, beobachten den tatsächlichen neuen Zustand aus der echten Umgebung und planen dann von Grund auf neu.

Dieser Trick hat einen Namen: **Model Predictive Control** (MPC).

---

## Eine Analogie aus dem echten Leben

Sie sind in einem Restaurant und schauen sich die Speisekarte an. Sie legen sich nicht sofort auf eine Bestellung von fünf Gängen fest – Sie bestellen den ersten Gang, sehen, wie satt Sie sich fühlen, und entscheiden dann über den Nachtisch neu.

Oder: Sie fahren auf einer kurvenreichen Straße. Sie legen Ihre Lenkbewegungen nicht für die nächsten 30 Sekunden fest – Sie schauen ständig voraus, planen ein paar Sekunden voraus, führen die nächste Lenkbewegung aus und planen dann neu.

Diese Schleife aus **weit vorausplanen / kurzfristig handeln / neu planen** ist MPC.

---

## Wie „Random Shooting“ funktioniert

Es gibt anspruchsvollere Planer – zum Beispiel:
- **CEM** (Cross-Entropy Method): Verfeinert iterativ eine Verteilung über Pläne, indem in jeder Runde nur die besten k Pläne behalten werden.
- **MCTS** (Monte Carlo Tree Search): Baut einen Suchbaum auf, der durch Simulationsstatistiken geleitet wird (verwendet von AlphaGo und MuZero).
- **Gradientenbasierte Planer**: Differenzieren die Vorhersagen des Modells in Bezug auf die Aktionen und folgen direkt dem Gradienten.

Wir verwenden die einfachste Methode, die funktioniert: **Random Shooting**.

```
Gegeben der aktuelle Zustand s:
    1. Erzeuge N=200 zufällige Aktionssequenzen der Länge H=15.
    2. Simuliere für jede Sequenz den Verlauf durch das Weltmodell ausgehend von s
       und summiere eine geformte (shaped) Belohnung bei jedem Schritt.
       (200 Träume parallel — schnell!)
    3. Finde die Sequenz mit der höchsten vorhergesagten Gesamtbelohnung.
    4. Führe die ERSTE Aktion dieser Sequenz in der realen Umgebung aus.
    5. Beobachte den realen nächsten Zustand. Verwirf den Rest des Plans.
    6. Gehe zu Schritt 1 — plane von Grund auf neu.
```

200 Pläne × 15 Schritte = 3.000 gedachte Übergänge pro realem Schritt. Das Weltmodell führt sie alle in einem einzigen gebündelten (batched) Vorwärtspass des neuronalen Netzwerks aus – normalerweise in wenigen Millisekunden.

---

## Warum bei jedem Schritt neu planen?

Weil das Modell unvollkommen ist. Fehler summieren sich über einen Durchlauf (Rollout) auf (wie in der von `world_model.py` erzeugten Grafik in `outputs/world_model.png` zu sehen). Der Plan bei Schritt 0 ist nur für die ersten paar Züge verlässlich; bei Schritt 15 halluziniert das Modell bereits. Daher vertrauen wir nur dem **ersten Zug** und aktualisieren dann den Plan mit dem neuesten realen Zustand.

Das ist derselbe Grund, warum Menschen keinen Schachplan über 100 Züge schreiben und stur daran festhalten – Umstände ändern sich, und je weiter man rät, desto weniger entspricht es der Realität.

---

## Ein Haken: Die Belohnung muss dem Planer etwas mitteilen

In CartPole ist die reale Belohnung `+1` für jeden Schritt, bis der Stab fällt. Das Modell wird für fast jeden Plan brav `+1, +1, +1, ...` vorhersagen, da zufällige Pläne innerhalb des Modells selten schnell enden – und somit jeder Plan die gleiche Punktzahl erhält. Der Planer hat keine Entscheidungsgrundlage.

Die Lösung: Ersetze die reale Belohnung während der Planung durch einen **glatten Proxy** (Stellvertreter):

```python
reward_proxy(state) = 1
                    - |Stabwinkel| / 0,21          # Stab aufrecht? (1=ja)
                    - 0,1 * |Wagenposition| / 2,4  # Wagen mittig? (1=ja)
```

Nun erhalten Pläne, die mit einem umgefallenen Stab enden *würden*, sichtbar schlechtere Bewertungen als Pläne, bei denen der Stab aufrecht bleibt. Der Planer kann sie in eine Rangfolge bringen.

> **Lektion für das echte Leben:** Ein flaches Belohnungssignal – „du hast eine weitere Sekunde überlebt“ – ist für kurzfristige Planung nutzlos. Dichte, geformte Signale helfen.

---

## Was unser Code tut

`model_based_planning.py`:

1. **Lädt** die von `world_model.py` gespeicherten Gewichte des Weltmodells. (Falls diese fehlen, wird on-the-fly eines trainiert.)
2. **Führt 20 Episoden** MPC auf dem realen CartPole-v1 aus.
3. **Führt ebenfalls 20 Episoden** mit einer gleichmäßig zufälligen Policy als Baseline aus.
4. **Plottet** beide im Vergleich und gibt die Durchschnittswerte aus.

### Was Sie beim Ausführen sehen sollten

| Policy | Durchschnittliche Belohnung (überlebte Schritte) |
|--------|-------------------------------:|
| Random (Zufall) | ~22 (typisch für CartPole — Stab fällt schnell) |
| MPC (unser Agent) | ~150–500 (variiert je nach Seed; viele Episoden nahe 500) |
| Maximum | 500 |

Diese **5- bis 25-fache Verbesserung** wird ohne Policy-Netzwerk, ohne Value-Funktion und ohne weiteres Training erreicht. Nur ein Weltmodell + 200 „Träume“ pro Schritt.

Das Diagramm `outputs/model_based_planning.png` zeigt zwei farbige Balken pro Episode – MPC ist fast immer höher als Random, und viele Episoden erreichen die Obergrenze von 500 Schritten.

---

## Stärken der modellbasierten Planung

- **Daten-effizient (Sample efficient).** Das gesamte Lernen erfolgte aus einem einzigen Batch zufälliger Übergänge. Es war keine weitere Interaktion mit der Umgebung nötig, um eine nützliche Policy abzuleiten.
- **Leicht anpassbar.** Sie möchten den Agenten anders steuern? Ändern Sie einfach den Belohnungs-Proxy – kein erneutes Training nötig. (Versuchen Sie zum Spaß, die Wagengeschwindigkeit zu maximieren.)
- **Interpretierbar.** Sie können die Pläne, die der Agent in Betracht gezogen hat, die vorhergesagten Trajektorien und die Scores inspizieren.

## Schwächen (und wie man damit umgeht)

- **Random Shooting ist simpel.** Es zieht Pläne blindlings. Für höhere Dimensionen wechselt man zu **CEM** (siehe oben), **iLQR** (Iterative Linear-Quadratic Regulator, eine klassische Methode der optimalen Steuerung) oder einem vollständigen **gradientenbasierten** Planer, der Aktionen durch Verfolgen von Gradienten durch ein differenzierbares Modell verbessert.
- **Sich summierende Modellfehler.** Lange Zeithorizonte driften ab. Man verwendet **probabilistische Ensembles** (mehrere Modelle, die auf denselben Daten trainiert wurden, wie in PETS, Chua et al. 2018), damit der Planer Unstimmigkeiten bemerken und Pläne bestrafen kann, bei denen sich das Modell unsicher ist.
- **Die reale Belohnung ist letztlich das, was wir wollen.** Reward Shaping hilft, aber für komplexere Aufgaben lernt man eine **Value-Funktion**, die *innerhalb* des Weltmodells trainiert wird – ein gelernter Critic, der den langfristigen Return von jedem Zustand aus schätzt, ohne dass ein handgefertigter Proxy erforderlich ist. Sowohl **Dreamer** (das einen Actor-Critic vollständig in der latenten Vorstellung trainiert) als auch **MuZero** (das MCTS mit einem gelernten Value-Netzwerk kombiniert) nutzen diese Idee.

---

## Verbindung zu modernen Systemen

Das Rezept, das Sie gerade ausgeführt haben – **gelernte Dynamik + Planung** – ist das Rückgrat einiger der stärksten RL-Systeme in der modernen KI-Forschung:

- **MuZero** (DeepMind): Kombiniert ein gelerntes Weltmodell mit Monte Carlo Tree Search. Es meisterte Go, Schach, Shogi und Atari, ohne die Regeln vorher zu kennen.
- **Dreamer / DreamerV3** (Hafner et al.): Trainiert eine Policy *innerhalb* eines gelernten **latenten** Weltmodells (das Modell komprimiert rohe Bilder oder Zustände in eine kompakte, abstrakte Darstellung, bevor es die Zukunft vorhersagt). Es erreicht State-of-the-Art-Leistungen in über 100 Benchmarks.
- **PETS / PlaNet / TD-MPC**: Dies sind Algorithmen-Familien, die genau diese Idee auf komplexe kontinuierliche Steuerungsaufgaben wie in der Robotik skalieren.

Sie haben – in wenigen hundert Zeilen – das kleinste Mitglied dieser Familie gebaut.

---

## Wichtige Begriffe

| Begriff | Einfache Erklärung |
|------|---------------|
| **MPC** | Model Predictive Control — vorausplanen, einmal handeln, neu planen |
| **Random Shooting** | Viele zufällige Pläne bewerten, den besten wählen |
| **Horizon (H)** | Wie viele Schritte der Plan in die Zukunft blickt |
| **N Samples** | Wie viele Kandidaten-Pläne wir pro Schritt betrachten |
| **Receding Horizon** | Neuplanung bei jedem Schritt statt sturem Festhalten an einem Plan |
| **Reward Proxy / Shaping** | Eine glatte Ersatz-Belohnung, die dem Planer ein nützliches Signal zur Optimierung gibt |

---

## Zusammenfassung in einem Satz

> **Sobald man ein Weltmodell hat, ist Planung nur noch ein „Träumen von hundert Zukünften, Auswahl des besten ersten Schritts und Wiederholen“.**

Das ist das ganze Geheimnis des modellbasierten RL.
