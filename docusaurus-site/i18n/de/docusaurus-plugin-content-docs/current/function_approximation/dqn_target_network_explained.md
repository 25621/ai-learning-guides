# Target-Netzwerk: Das Lernziel stabilisieren 🎯

## Das Problem des beweglichen Ziels

Stellen Sie sich vor, Sie versuchen, mit Pfeil und Bogen das Schwarze einer Zielscheibe zu treffen. Sie schießen, schauen, wo der Pfeil gelandet ist, und passen Ihr Zielen für das nächste Mal an. Einfach, oder?

Stellen Sie sich nun vor, die Zielscheibe BEWEGT SICH jedes Mal, wenn Sie schießen! Jeder Pfeil, den Sie abfeuern, verändert die Position der Zielscheibe für den nächsten Schuss ein wenig. Sie würden sich nie verbessern – Sie würden einem Ziel hinterherjagen, das ständig wegläuft.

Genau das ist das Problem bei DQN ohne ein Target-Netzwerk!

---

## Warum Q-Ziele sich ständig bewegen

Bei DQN ist das Ziel (Target) für jedes Update:
> Target = Belohnung + γ × max(Q(nächster_Zustand))

Hier ist **γ (Gamma)** der **Diskontierungsfaktor** – eine Zahl zwischen 0 und 1 (typischerweise 0,99), die steuert, wie sehr sich der Agent um *zukünftige* Belohnungen im Vergleich zu *sofortigen* Belohnungen kümmert.

**Beispiel aus dem echten Leben:** Stellen Sie sich vor, jemand bietet Ihnen jetzt einen Keks an oder morgen zwei Kekse. Wenn Sie die Kekse unbedingt jetzt wollen, ist Ihr γ niedrig (Sie diskontieren die Zukunft stark). Wenn Sie geduldig sind und gerne warten, ist Ihr γ hoch (zukünftige Belohnungen zählen fast so viel wie jetzige). Im RL bedeutet γ = 0,99: „Eine Belohnung im nächsten Schritt ist 99 % einer Belohnung zum jetzigen Zeitpunkt wert.“

Die Q-Werte auf der rechten Seite kommen von... demselben Netzwerk, das wir gerade trainieren!

Jedes Mal, wenn wir also das Netzwerk aktualisieren (um die Q-Werte zu verbessern), ändern wir auch die Ziele. Es ist eine Rückkopplungsschleife:

1. Netzwerk aktualisieren → Q-Werte ändern sich.
2. Q-Werte ändern sich → Ziele ändern sich.
3. Ziele ändern sich → Netzwerk wird anders aktualisiert.
4. Endlose Wiederholung — instabil!

**Beispiel aus dem echten Leben:** Versuchen Sie, sich auf einer Waage zu wiegen, die ihre Anzeige jedes Mal ändert, wenn Sie darauf steigen. Sie würden nie Ihr wahres Gewicht erfahren!

---

## Die Lösung: Das Ziel einfrieren! ❄️

Das **Target-Netzwerk** ist eine KOPIE des Haupt-Q-Netzwerks, das vorübergehend eingefroren wird.

- **Online-Netzwerk** (`qnet`): Wird bei jedem Trainingsschritt aktualisiert – lernt schnell.
- **Target-Netzwerk** (`target_net`): Eingefrorene Kopie – wird nur alle 100 Schritte aktualisiert.

Wir verwenden das EINGEFRORENE Target-Netzwerk, um die Ziele zu berechnen:
> Target = Belohnung + γ × max(Q_TARGET(nächster_Zustand))

Das Ziel bleibt für 100 Schritte unbeweglich! Das gibt dem Online-Netzwerk ein stabiles Ziel, auf das es hinarbeiten kann. Danach kopieren wir die Gewichte des Online-Netzwerks in das Target-Netzwerk, frieren es wieder ein und wiederholen das Ganze.

**Beispiel aus dem echten Leben:** Denken Sie an einen Schüler und einen Lehrer. Der Lehrer gibt Hausaufgaben auf (das Ziel). Der Schüler lernt und verbessert sich. Nach 100 Lektionen aktualisiert der Lehrer die Hausaufgaben, um sie schwieriger zu machen. Der Lehrer ändert sich nicht jede einzelne Minute – das wäre zu chaotisch!

---

## Das vollständige DQN-Rezept 🍕

Der vollständige DQN-Algorithmus (Experience Replay + Target-Netzwerk) sieht so aus:

```
1. Initialisiere Online-Netzwerk Q und Target-Netzwerk Q_target (gleiche Gewichte).
2. Erstelle einen Replay Buffer (Erinnerungsbox).

In jedem Schritt der Umgebung:
  a. Wähle Aktion mit ε-greedy basierend auf Q.
  b. Speichere (Zustand, Aktion, Belohnung, nächster Zustand) im Buffer.

Alle 4 Schritte:
  c. Wähle einen zufälligen Mini-Batch aus dem Buffer.
  d. Berechne Ziele mit Q_TARGET (eingefroren!).
  e. Aktualisiere Q, um den Verlust (Loss) zu minimieren.

Alle 100 Schritte:
  f. Kopiere Q-Gewichte nach Q_TARGET (Synchronisierung).
```

Dies ist genau der Algorithmus aus dem wegweisenden DQN-Paper von DeepMind (2015)!

---

## Was der Vergleich zeigt

Wenn Sie `dqn_target_network.py` ausführen, werden Sie sehen:

**Ohne Target-Netzwerk (nur DQN + Replay):**
- Das Training kann „okay“ verlaufen, bricht aber periodisch zusammen.
- Q-Werte können divergieren (explodieren oder schwanken).
- Das Lernen ist weniger vorhersehbar.

**Vollständiges DQN (Replay + Target-Netzwerk):**
- Beständigerer Lernfortschritt nach oben.
- Q-Werte bleiben in einem vernünftigen Bereich.
- Schnellere Konvergenz zum gelösten Schwellenwert (195+ bei CartPole).

---

## Die „tödliche Triade“ (Deadly Triad) ☠️

Im Reinforcement Learning erzeugt die Kombination von drei Dingen Instabilität:

1. **Funktionsapproximation** (neuronales Netzwerk statt Tabelle) ← wir verwenden dies.
2. **Bootstrapping** (Verwendung von Q-Werten zur Schätzung von Q-Werten) ← wir verwenden dies.
3. **Off-Policy-Lernen** (Q-Learning verwendet das Maximum, nicht die tatsächliche Policy) ← wir verwenden dies.

Alle drei zusammen ergeben die „tödliche Triade“. DQN bändigt dies mit:
- Experience Replay → bricht Korrelationen auf.
- Target-Netzwerk → bricht die Rückkopplungsschleife auf.

Es löst das Problem nicht vollständig, aber es macht es handhabbar!

---

## Wichtige Begriffe

| Begriff | Bedeutung |
|------|---------|
| **Target-Netzwerk** | Eine eingefrorene Kopie des Q-Netzwerks, die nur zur Berechnung der Ziele verwendet wird |
| **Online-Netzwerk** | Das Q-Netzwerk, das aktiv trainiert wird |
| **Sync** | Kopieren der Gewichte des Online-Netzwerks in das Target-Netzwerk |
| **Rückkopplungsschleife** | Wenn die Ausgabe eines Systems die Eingabe verändert (kann Instabilität verursachen) |
| **Deadly Triad** | Kombination aus Funktionsapproximation + Bootstrapping + Off-Policy-Lernen |

---

## Auswirkungen auf die reale Welt

Im Jahr 2015 veröffentlichte DeepMind sein DQN-Paper und zeigte eine KI, die 49 Atari-Spiele auf übermenschlichem Niveau spielen konnte – allein durch diese zwei Tricks (Replay + Target-Netzwerk).

Zuvor dachte man, man könne neuronale Netzwerke wegen der Instabilität nicht mit RL trainieren. DeepMind bewies das Gegenteil und löste damit die Deep-RL-Revolution aus!

Als Nächstes werden wir dieses vollständige DQN-Rezept auf Atari Pong anwenden – ein echtes Videospiel mit rohen Pixeln als Eingabe!
