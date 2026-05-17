# Double DQN: Das Problem der Selbstüberschätzung lösen 🤔

## Das Problem: DQN denkt, es sei besser als es ist

Stellen Sie sich vor, Sie werden gefragt: „Was ist das beste Restaurant der Stadt?“

Sie könnten sagen: „Pizza Palace ist fantastisch – es ist definitiv eine 10/10!“ Aber Sie waren erst zweimal dort. Sie wissen eigentlich nicht, ob es *wirklich* eine 10/10 ist. Vielleicht überschätzen Sie es, weil Sie bei diesen zwei Besuchen einfach Glück mit einer besonders guten Pizza hatten.

Dasselbe Problem tritt bei DQN auf: Der Agent **überschätzt die Q-Werte**.

---

## Warum überschätzt DQN?

Wenn DQN das Ziel (Target) berechnet, passiert Folgendes:
> Target = Belohnung + γ × **max** Q(nächster_Zustand)

Das `max` ist das Problem! Wenn man das Maximum aus mehreren verrauschten Schätzungen wählt, wählt man fast immer diejenige mit dem größten zufälligen Fehler nach oben (Upward Bias).

**Beispiel aus dem echten Leben:** Sie lassen 5 Freunde die Höhe eines Gebäudes schätzen. Ihre Schätzungen sind: 40m, 38m, 45m (Glückstreffer!), 39m, 41m. Die wahre Höhe ist 40m. Wenn Sie `max(Schätzungen)` = 45m verwenden, liegen Sie völlig daneben! Das Maximum verrauschter Schätzungen ist fast immer eine Überschätzung.

Über tausende von Updates hinweg trainiert DQN immer weiter in Richtung dieser aufgeblähten Ziele und lernt, dass die Dinge besser sind, als sie in Wirklichkeit sind. Dies kann das Lernen verlangsamen oder dazu führen, dass der Agent überhebliche, schlechte Entscheidungen trifft.

---

## Die Lösung: Double DQN

**Double DQN** (Hasselt et al., 2016) teilt das `max` in zwei Schritte auf:

**Schritt 1 — Welche Aktion?** Verwende das **Online-Netzwerk**, um die beste Aktion auszuwählen:
> beste_Aktion = argmax Q_online(nächster_Zustand)

**Schritt 2 — Welchen Wert hat sie?** Verwende das **Target-Netzwerk**, um diese Aktion zu bewerten:
> Target = Belohnung + γ × Q_target(nächster_Zustand, beste_Aktion)

```
Normales DQN:  Target = r + γ × max_a Q_target(s', a)
                                 ↑ das gleiche Netzwerk wählt aus UND bewertet → voreingenommen (biased)

Double DQN:    beste_a = argmax_a Q_online(s', a)   ← Online-Netzwerk wählt
               Target = r + γ × Q_target(s', beste_a) ← Target-Netzwerk bewertet
                                 ↑ verschiedene Netzwerke → weniger voreingenommen
```

**Beispiel aus dem echten Leben:** In einem Vorstellungsgespräch lassen Sie den Bewerber seinen eigenen Leistungstest nicht selbst benoten (das ist das Problem des normalen DQN!). Stattdessen *benennt* der Kandidat seine beste Arbeit, und ein *separater* Prüfer bewertet sie. Zwei verschiedene Personen = fairere Bewertung!

---

## Warum hilft die Trennung?

Die beiden Netzwerke (Online und Target) haben unterschiedliche Gewichte, da das Target-Netzwerk seltener aktualisiert wird. Sie haben unterschiedliche „Meinungen“ darüber, welche Aktion die beste ist.

Wenn sie sich uneinig sind:
- Online sagt: „Aktion A sieht toll aus!“
- Target sagt: „Eigentlich ist Aktion A nur okay – etwa 7 wert, nicht 10.“

Indem wir die WERt-Schätzung des Target-Netzwerks für die vom Online-Netzwerk GEWÄHLTE Aktion verwenden, erhalten wir eine ehrlichere, weniger aufgeblähte Zahl.

---

## Code-Unterschied: Nur eine Zeile!

Die einzige Code-Änderung von normalem zu Double DQN liegt in der Target-Berechnung:

```python
# Normales DQN:
q_next = target_net(s_next).max(dim=1).values

# Double DQN:
best_actions = online_net(s_next).argmax(dim=1, keepdim=True)   # Wahl durch Online-Netzwerk
q_next = target_net(s_next).gather(1, best_actions)              # Bewertung durch Target-Netzwerk
```

Es ändern sich nur zwei Zeilen – aber die Auswirkungen auf Stabilität und Genauigkeit sind erheblich!

---

## Was der Vergleich zeigt

Wenn Sie `double_dqn_cartpole.py` ausführen, sehen Sie zwei Diagramme:

**Diagramm 1: Lernkurven**
- Sowohl das normale als auch das Double DQN sollten CartPole lösen.
- Double DQN konvergiert oft schneller und stabiler.
- CartPole ist einfach genug, dass der Unterschied moderat ist; bei Atari ist er viel deutlicher.

**Diagramm 2: Q-Wert-Schätzungen**
- Normales DQN: Die Q-Werte driften im Laufe der Zeit nach oben (Überschätzung).
- Double DQN: Die Q-Werte bleiben moderater und genauer.

Das Diagramm der Q-Wert-Überschätzung ist die wichtigste Erkenntnis – es zeigt, wie das normale DQN aufgeblähte Werte lernt, die letztendlich die Leistung beeinträchtigen.

---

## Wie viel besser ist Double DQN?

| Metrik | Normales DQN | Double DQN |
|--------|--------------|------------|
| Q-Wert-Genauigkeit | Überschätzt | Genauer |
| Lernstabilität | Mehr Varianz | Weniger Varianz |
| CartPole-Leistung | Gut | Etwas besser |
| Atari-Leistung (50 Spiele) | Baseline | +2,6× mehr Spiele nahe menschlichem Niveau |

Bei komplexen Atari-Spielen machte Double DQN einen deutlich größeren Unterschied als bei CartPole, weil Atari wesentlich verrauschtere Q-Wert-Schätzungen hat.

---

## Die Familie der DQN-Verbesserungen

Double DQN ist nur eine von vielen Verbesserungen des ursprünglichen DQN. Das „Rainbow“-Paper (2017) kombinierte 6 Verbesserungen:

1. **Double DQN** (behebt Überschätzung) ← dieses Skript!
2. **Prioritized Replay** (lernt mehr aus überraschenden Erfahrungen)
3. **Dueling Networks** (trennt „Wie gut ist dieser Zustand?“ von „Was ist die beste Aktion?“)
4. **Multi-Step Returns** (blickt weiter in die Zukunft)
5. **Distributional RL** (lernt die vollständige Verteilung der Returns, nicht nur den Durchschnitt)
6. **NoisyNets** (gelerntes Erforschen statt [ε-greedy](../foundations/multi_armed_bandit_explained.md#die-epsilon-greedy-strategie))

Rainbow kombinierte ALLE diese Ansätze und erreichte die beste Atari-Leistung seiner Zeit!

---

## Wichtige Begriffe

| Begriff | Bedeutung |
|------|---------|
| **Überschätzung** | Q-Werte sind höher als die wahren Werte (zu optimistisch) |
| **Double DQN** | Verwendet das Online-Netzwerk zur Aktionswahl und das Target-Netzwerk zur Bewertung |
| **Entkopplung** | Trennung von zwei Aufgaben, die zuvor vom selben Netzwerk erledigt wurden |
| **Bias** | Ein systematischer Fehler in eine Richtung (immer zu hoch oder immer zu niedrig) |
| **Rainbow** | Eine DQN-Variante, die 6 Verbesserungen für maximale Leistung kombiniert |

---

## Zusammenfassung: Die Reise durch Phase 3

Sie haben nun die gesamte Progression von Phase 3 abgeschlossen:

| Algorithmus | Was er hinzufügt | Warum es hilft |
|-----------|-------------|-------------|
| Linear Q | Neuronales Netz → einfache Formel | Bewältigt kontinuierliche Zustände |
| Naives DQN | Vollständiges neuronales Netz | Lernt komplexe Muster |
| + Replay Buffer | Zufällige Stichproben aus dem Speicher | Bricht Korrelationen auf |
| + Target-Netzwerk | Eingefrorene Kopie für Ziele | Stabilisiert das Lernziel |
| Atari DQN | CNN + Frame Stacking | Lernt direkt aus Pixeln! |
| Double DQN | Getrennte Wahl/Bewertung | Reduziert Überschätzung |

Jeder Schritt hat ein spezifisches Problem gelöst. So funktioniert echte Forschung – eine sorgfältige Verbesserung nach der anderen!
