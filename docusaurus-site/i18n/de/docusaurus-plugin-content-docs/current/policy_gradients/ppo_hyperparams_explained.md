# PPO-Hyperparameter-Sensitivität: Was zählt am meisten?

## Warum Hyperparameter wichtig sind

Stellen Sie sich vor, Sie backen einen Schokoladenkuchen. Das Rezept verlangt:
- 2 Eier
- 200g Mehl
- 1 Teelöffel Backpulver
- 35 Minuten bei 180 °C

Wenn Sie 10 Eier verwenden, explodiert der Kuchen. Wenn Sie nur 0,1 Teelöffel Backpulver nehmen, geht er nicht auf. Wenn Sie ihn 10 Minuten lang bei 300 °C backen, verbrennt er außen und ist innen noch roh.

**Hyperparameter in PPO sind wie Zutaten und Ofeneinstellungen.** Die richtige Kombination funktioniert wunderbar; falsche Einstellungen können das Lernen gänzlich verhindern.

Dieses Skript testet systematisch drei zentrale Hyperparameter, indem jeweils nur EIN Parameter geändert wird. Jede Einstellung wird mit drei verschiedenen Zufallssaaten (Seeds) ausgeführt und die Ergebnisse werden verglichen.

---

## Die drei Experimente

### Experiment 1: Clip-Epsilon (ε)

```
ε = 0.05   (sehr konservativ — nur winzige Policy-Änderungen erlaubt)
ε = 0.2    (Standard — ausgewogenes Verhältnis zwischen Sicherheit und Geschwindigkeit)
ε = 0.4    (aggressiv — erlaubt große Policy-Änderungen)
```

**Was steuert ε?**

ε ist die Größe des „Sicherheitsfensters“ um die alte Policy herum:
```
Verhältnis (Ratio) muss im Bereich [1 - ε, 1 + ε] bleiben
ε=0.05: Ratio zwischen [0.95, 1.05]  ← winzige Änderungen
ε=0.2:  Ratio zwischen [0.80, 1.20]  ← Standard
ε=0.4:  Ratio zwischen [0.60, 1.40]  ← große Änderungen
```

**Beispiel aus dem echten Leben:** Betrachten Sie ε als die Angabe, „wie weit man das Lenkrad mit einer Bewegung einschlagen darf“.
- ε=0.05: Wie Fahren auf Glatteis — nur minimale Korrekturen.
- ε=0.2:  Normales Fahren — angemessene Kurven.
- ε=0.4:  Rennfahrer — aggressives Lenken, Risiko des **Ausbrechens** (Verlust der Kontrolle, da die Änderung zu drastisch ist).

**Erwartete Ergebnisse:**
- ε=0.05: Langsames, aber stabiles Lernen (zu vorsichtig).
- ε=0.2:  Gute Balance (der **„Goldlöckchen-Wert“** — nicht zu klein, nicht zu groß, genau richtig).
- ε=0.4:  Kann schnell lernen, neigt aber zum **Überschießen und Oszillieren** (Überschießen = über die optimale Policy hinausgehen; Oszillieren = wie ein Pendel wild um das Ziel herumschwingen, ohne zur Ruhe zu kommen).

---

### Experiment 2: Lernrate (Learning Rate)

```
lr = 1e-4  (langsam aber stabil)
lr = 3e-4  (Standard)
lr = 1e-3  (schnell aber riskant)
```

**Was steuert die Lernrate?**

Die Lernrate ist wie die „Schrittweite“ beim Besteigen eines Hügels (jeder Schritt entspricht einem Update der Gewichte des neuronalen Netzwerks in die Richtung, die die Belohnung verbessert):
- Zu klein: Es dauert ewig, bis man oben ankommt (konvergiert langsam).
- Zu groß: Man schießt über den Gipfel hinaus und fällt auf der anderen Seite hinunter (**divergiert** — die Belohnung bricht zusammen oder schwankt wild).
- Genau richtig: Stetiger Fortschritt in Richtung Gipfel.

**Beispiel aus dem echten Leben:** Das Stimmen einer Gitarrensaite.
- lr=1e-4: Winzige Drehungen am Wirbel — dauert ewig, ist aber präzise.
- lr=3e-4: Normales Stimmen — den richtigen Ton in wenigen Drehungen finden.
- lr=1e-3: Grobe Rucke am Wirbel — die Saite könnte **reißen** (das Training bricht irreversibel ab)!

**Erwartete Ergebnisse:**
- lr=1e-4: Schließlich gut, aber sehr langsam.
- lr=3e-4: Beste Gesamtleistung.
- lr=1e-3: Schneller Anfangsfortschritt, dann Instabilität.

---

### Experiment 3: Update-Epochen (K)

```
K = 3   (konservativ — wenige Durchgänge durch jeden Batch)
K = 10  (Standard)
K = 20  (aggressiv — viele Durchgänge durch jeden Batch)
```

**Was steuern die Update-Epochen?**

Nachdem ein **Rollout** gesammelt wurde (= das Spiel für eine gewisse Zeit gespielt wurde, um neue Erfahrungen zu sammeln), verpackt PPO diese Erfahrungen in einen **Batch** (Satz von Tupeln aus Zustand, Aktion, Belohnung). Es führt dann K Durchgänge (Sweeps) über dieselben Daten aus, wobei das Netzwerk jedes Mal aktualisiert wird.
Mehr Epochen bedeuten, dass mehr Lerneffekt aus jedem Batch gepresst wird, aber es besteht das Risiko des **Overfittings auf veraltete Daten** (= Muster werden gelernt, die unter der alten Policy wahr waren, aber nicht mehr gelten, nachdem die Policy aktualisiert wurde).

**Beispiel aus dem echten Leben:** Ein Schüler, der mit 20 Matheaufgaben übt.
- K=3:  Jede Aufgabe 3-mal machen → man lernt noch, kein Overfitting auf den Übungssatz.
- K=10: Jede Aufgabe 10-mal machen → solide Beherrschung dieser spezifischen Aufgaben.
- K=20: Jede Aufgabe 20-mal machen → **Lösungen auswendig lernen, ohne die Mathematik dahinter wirklich zu verstehen** (= das Modell passt perfekt auf den spezifischen Batch, verliert aber die Fähigkeit zur Generalisierung)!

> ⚠️ **„Aber die Ergebnisse für K=20 sehen doch gut aus — warum sollte mich das kümmern?“**
> Der Clipping-Trick von PPO begrenzt, wie stark sich die Policy pro Durchgang ändern kann, daher wird K=20 keinen plötzlichen Absturz verursachen.
> Dennoch passt sich der Agent im Stillen zu sehr an Daten an, die nicht mehr widerspiegeln, was die aktuelle Policy tatsächlich erleben würde.
> Dies **verlangsamt das langfristige Lernen**: Jeder Rollout lehrt den Agenten weniger, als er sollte, weil spätere Durchgänge zunehmend veraltete Informationen recyceln. Der Schaden ist schleichend, nicht dramatisch — genau deshalb wird er in kurzen Experimenten oft übersehen.

Das Clipping verhindert katastrophales Overfitting, aber zu viele Epochen können das Lernen dennoch insgesamt verlangsamen.

**Erwartete Ergebnisse:**
- K=3:  Weniger effizient (etwas Lernpotenzial pro Batch wird verschenkt).
- K=10: Gute Balance.
- K=20: Risiko, dass die Policy **zu sicher auf veralteten Daten** wird (= die Updates des Netzwerks werden von Erfahrungen getrieben, die nicht mehr dem entsprechen, was die aktuelle Policy antreffen würde, was die Dateneffizienz untergräbt).

---

## Wie man die Ergebnisse liest

Das Diagramm zeigt drei Grafiken, in denen jeweils ein Hyperparameter variiert wird:

```
Linke Grafik:   Clip-Epsilon — welches ε lernt am schnellsten?
Mittlere Grafik: Lernrate — welche lr ist am stabilsten?
Rechte Grafik:  Update-Epochen — welches K findet die beste Policy?
```

Jede Linie stellt die **durchschnittliche Belohnung über 3 Seeds** dar (um den Zufallseinfluss zu reduzieren).

**Worauf Sie achten sollten:**
1. **Lerngeschwindigkeit:** Welche Linie erreicht am schnellsten eine hohe Belohnung?
2. **Endleistung:** Welche Linie erreicht die höchste finale Belohnung?
3. **Stabilität:** Welche Linie weist die geringsten Schwankungen (Oszillationen) auf?

Ein guter Hyperparameter balanciert alle drei Aspekte aus!

---

## Methodik: Wissenschaftliches Experimentieren

Dieses Experiment nutzt ein **Ablationsstudien-Design** (= eine Methode, bei der man eine Komponente nach der anderen entfernt oder variiert, um ihren individuellen Einfluss zu messen):
1. Standardwerte wählen: ε=0.2, lr=3e-4, K=10.
2. Jeweils nur EINEN Parameter ändern.
3. Alles andere konstant halten.
4. Ergebnisse vergleichen.

Dies zeigt uns den Effekt JEDES Parameters isoliert auf.

---

## Typische Erkenntnisse aus der Praxis

| Hyperparameter | Zu klein | Idealbereich | Zu groß |
|----------------|-----------|------------|-----------|
| **ε (clip)** | Langsame Konvergenz | ε ≈ 0.2 | Instabilität |
| **lr** | Zu langsam | 2.5e-4 bis 3e-4 | Divergenz |
| **K (Epochen)** | **Datenverschwendung** | K = 4-10 | Overfitting auf alte Rollout-Daten |
| **n_steps** | Zu verrauscht | 128-2048 | **OOM-Speicherfehler** (RAM voll) |
| **batch_size** | Zu verrauscht | 32-256 | **OOM-Speicherfehler** (RAM voll) |

Diese „Idealbereiche“ können sich je nach Umgebung verschieben!

---

## Die zentrale Erkenntnis: PPO ist relativ robust

Im Vergleich zu früheren Algorithmen (wie DQN ohne Target-Netzwerke) ist PPO relativ unempfindlich gegenüber der Wahl der Hyperparameter. Der Clipping-Mechanismus bietet ein natürliches Sicherheitsnetz.

**Beispiel aus dem echten Leben:** Ein Auto mit **ABS** (Antiblockiersystem) beim Bremsen im Vergleich zu einem ohne:
- Ohne ABS (DQN): Eine falsche Bewegung (schlechter Hyperparameter) und man dreht sich.
- Mit ABS (PPO): Das Auto korrigiert sich selbst — vernünftige Hyperparameter funktionieren alle halbwegs gut.

Diese Robustheit ist ein Hauptgrund, warum PPO in der Praxis der beliebteste RL-Algorithmus ist!

---

## Wichtige Begriffe

| Begriff | Einfache Erklärung |
|---------|---------------|
| **Ablationsstudie** | Eine Sache nach der anderen ändern, um den Effekt zu sehen |
| **Clip-Epsilon ε** | Sicherheitsgrenze — 0.2 ist meist am besten |
| **Lernrate** | **Schrittweite** — wie stark die Gewichte des Netzwerks nach jedem Batch angepasst werden. **2.5e-4 bis 3e-4** entspricht 0,00025 bis 0,0003 |
| **Update-Epochen K** | Wie oft jeder Batch wiederverwendet wird — 4 bis 10 ist Standard |
| **Random Seeds** | Jedes Experiment wird mit verschiedenen Startwerten für den Zufallszahlengenerator wiederholt, um zu sehen, ob die Ergebnisse konsistent sind |

---

## Zusammenfassung: Policy-Gradient-Methoden im Überblick

```
REINFORCE              A2C                    PPO
     │                  │                      │
Ganze Episoden    n-Step-Updates         n-Step + Clipping
Einfach, verrauscht  Schneller, instabil    Stabil + effizient
Beste für einfache   Mittelschwere          Schwere Umgebungen
Umgebungen           Umgebungen             (Industriestandard)
```

**Wenn Sie nur EINEN Algorithmus aus dieser Phase lernen, dann lernen Sie PPO.** Er ist das Fundament für:
- Das ChatGPT-Training von OpenAI (RLHF nutzt PPO).
- Die Nachfolge-Projekte von DeepMinds AlphaGo.
- Den Großteil der modernen Robotik-Forschung.
- KIs, die Videospiele spielen.
