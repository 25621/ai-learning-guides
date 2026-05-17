# PPO für Continuous Control: Dem BipedalWalker das Laufen beibringen

## Diskrete vs. kontinuierliche Aktionen

Bisher hatte jede Umgebung, die wir gelöst haben, **diskrete** Aktionen:
- CartPole: schiebe nach LINKS oder RECHTS (2 Möglichkeiten).
- LunarLander: zünde nichts / links / Haupttriebwerk / rechts (4 Möglichkeiten).

Aber Roboter in der realen Welt benötigen **kontinuierliche** Aktionen:
- Ein humanoider Roboter: „wie stark soll jedes Gelenk gedrückt werden“ (jeder Wert von -1 bis +1).
- Ein Auto: „wie genau soll das Lenkrad eingeschlagen werden“ (jeder Winkel von -30° bis +30°).
- Ein Roboterarm: „übe exakt 2,3 Newton in diese Richtung aus“.

**Beispiel aus dem echten Leben:** Tippen auf einer Tastatur = diskret (drücke A, B, C...).
Schreiben mit einem Bleistift = kontinuierlich (bewege die Hand 2,3 cm nach rechts, drücke mit 40g Kraft...).

---

## Die Gauß-Policy für kontinuierliche Aktionen

Für kontinuierliche Aktionen verwenden wir anstelle einer kategorialen Verteilung (Wahl aus N Kategorien) eine **Normalverteilung (Gauß-Verteilung)**:

```
Aktion ~ Normal(μ, σ)
```

Dabei ist:
- **μ (Mu, Mittelwert)**: Das Zentrum der Verteilung — der Aktionswert, den das Netzwerk „anstrebt“.
- **σ (Sigma, Standardabweichung)**: Die Streuung — wie viel Zufälligkeit / Exploration hinzugefügt werden soll.

```
        Wahrscheinlichkeit
             │
        0.4 ─┤      ██████
             │    ████████████
        0.2 ─┤  ██████████████████
             │████████████████████████
             └──────────────────────── Aktionswert
           -1  -0.5   0   0.5   1
                      ↑
                  Mittelwert μ
```

**Beispiel aus dem echten Leben:** Ein geschickter Bogenschütze zielt auf die Mitte der Scheibe (μ). Seine Pfeile landen nicht alle exakt an derselben Stelle — es gibt eine gewisse Streuung (σ). Mit zunehmender Übung wird er genauer (σ sinkt), während er weiterhin auf das Schwarze zielt.

---

## Unser Gauß-Actor-Critic-Netzwerk

```
Zustand (24 Zahlen) → [256 Neuronen] → [256 Neuronen] →
    ├── Actor: 4 Mittelwerte (μ₁, μ₂, μ₃, μ₄)
    │          + 4 log_std Parameter (gemeinsam für alle Zustände!)
    └── Critic: 1 Wert (V(s))
```

Der `log_std` (Logarithmus der **Standardabweichung** — ein Maß für die Streuung oder Unsicherheit) ist ein **lernbarer Parameter** — er ist nicht vom Zustand abhängig. Dies hält das Modell einfach und ermöglicht es dennoch, dass sich die Exploration während des Trainings ändert.

**Warum log_std statt std?** Die Standardabweichung muss positiv sein. Durch die Verwendung von `log_std` kann das Netzwerk jede reelle Zahl (positiv oder negativ) ausgeben. Wir wenden dann `exp(log_std)` an — die Exponentialfunktion, die die Umkehrfunktion des Logarithmus ist —, um eine garantiert positive Standardabweichung zu erhalten. Dies verhindert, dass die Standardabweichung jemals negativ oder Null wird.

---

## Berechnung der Log-Wahrscheinlichkeit für kontinuierliche Aktionen

Bei diskreten Aktionen gilt: `log_prob = log(P(Aktion=LINKS))`.

Bei kontinuierlichen Aktionen beschreibt die **Normalverteilung** eine glatte Glockenkurve um den Mittelwert. Ein einzelner exakter Wert hat in der kontinuierlichen Mathematik die Wahrscheinlichkeit Null. Daher verwenden wir die Kurvenhöhe an diesem Wert, die als **pdf** (Probability Density Function, Wahrscheinlichkeitsdichtefunktion) bezeichnet wird:
```
log_prob = Σᵢ log[Normal(μᵢ, σᵢ).pdf(aᵢ)]
```

`log` steht für den natürlichen Logarithmus. Er verwandelt winzige Dichtewerte in stabile Zahlen, die für neuronale Netzwerke leichter zu optimieren sind. Wir summieren über alle Aktionsdimensionen (4 beim BipedalWalker), da die vollständige Aktion ein Vektor aus 4 Zahlen ist.

**Beispiel aus dem echten Leben:** Wie groß ist die Wahrscheinlichkeit, dass es morgen exakt 5,732...°C warm wird? Für das kontinuierliche Wetter würde man sich die Kurve der Normalverteilung ansehen und feststellen, wie hoch sie an diesem exakten Punkt ist. Wahrscheinlichere Temperaturen (nahe dem Mittelwert) haben eine höhere Dichte.

---

## BipedalWalker: Eine Herausforderung beim Laufen

BipedalWalker-v3 ist ein 2D-Roboter, der lernen muss zu laufen, ohne umzufallen:

```
          O (Kopf)
         /│\
        / │ \
       /  │  \
      L   │   R   ← zwei Beine, jedes mit einem Kniegelenk
     / \  │  / \
    ●   ● │ ●   ●  ← 4 Motoren (Hüfte/Knie für jedes Bein)
```

**Zustandsraum (24 Zahlen):**
- Rumpf (Hull): Winkel, Winkelgeschwindigkeit, horizontale Geschwindigkeit, vertikale Geschwindigkeit (4 Zahlen).
- Gelenke: 4 Motoren (2 Hüften, 2 Knie), die jeweils Winkel und Geschwindigkeit liefern, plus 2 Bodenkontaktsensoren (einer für jedes Bein) (10 Zahlen).
- 10 LIDAR-Entfernungssensoren (Entfernungsmessungen, die den Boden vor dem Roboter erfassen) (10 Zahlen).

**Aktionsraum (4 kontinuierliche Werte, jeweils in [-1, 1]):**
Die Aktionswerte steuern das **Drehmoment** (die von den Motoren ausgeübte Rotationskraft) für genau 4 Gelenke (es werden keine Aktionen direkt auf den Rumpf angewendet):
- Hüft-Drehmoment Bein 1, Knie-Drehmoment Bein 1, Hüft-Drehmoment Bein 2, Knie-Drehmoment Bein 2.

**Belohnungen:**
- +300 für das Erreichen des Ziels (rechte Seite).
- -100 für das Umfallen (Berühren des Bodens mit dem Körper).
- Kleine Belohnung pro Schritt für Vorwärtsbewegung.
- Kleine Strafe für jeden Einsatz der Motoren (Belohnung für Effizienz).

**Gelöst, wenn:** Die durchschnittliche Belohnung über 100 Episoden > 300 ist.

---

## Hauptunterschied zum diskreten PPO

Alles ist gleich, AUSSER:

| | Diskretes PPO | Kontinuierliches PPO |
|---|---|---|
| **Policy** | Categorical(logits) | Normal(μ, σ) |
| **Sample** | Aktion = Stichprobe aus {0,1,...,N} | Aktion = μ + σ × Rauschen |
| **log_prob** | log P(Aktion=k) | Σ log Normal(μᵢ, σᵢ).pdf(aᵢ) |
| **Clamp** | Nicht erforderlich | Aktionen auf [-1, 1] begrenzen (Clamp) |

**Logits** sind rohe, nicht normalisierte Scores für diskrete Aktionen. Eine kategoriale Policy wandelt diese mit **Softmax** in Wahrscheinlichkeiten um — eine Funktion, die eine beliebige Menge von Zahlen nimmt und sie in eine gültige Wahrscheinlichkeitsverteilung staucht (alle Werte positiv, Summe ergibt 1). Beispielsweise werden aus den Logits [2,0, 1,0, 0,5] die Wahrscheinlichkeiten [0,59, 0,24, 0,17]. Das kontinuierliche PPO verwendet **kein** Softmax für die Aktion selbst, da die Aktion nicht aus einem festen Menü gewählt wird. Stattdessen gibt die Policy den Mittelwert und die Standardabweichung einer Normalverteilung aus und zieht daraus reellwertige Drehmomente.

**Clamp** bedeutet, einen Wert in einen gültigen Bereich zu zwingen. Der Code verwendet `action.clamp(-1, 1)`, damit die Umgebung niemals einen Motorbefehl außerhalb der erlaubten Grenzen erhält.

**Clip** bedeutet bei PPO etwas anderes: PPO „clippt“ das Wahrscheinlichkeitsverhältnis innerhalb der Loss-Funktion, wie im Abschnitt [PPO Clipping](./ppo_scratch_explained.md#der-clipping-trick) erklärt. Das Clamping der Aktion schützt die Schnittstelle zur Umgebung; das Clipping bei PPO schützt das Policy-Update.

---

## Laufen lernen von Grund auf: Was der Agent lernt

**Frühes Training (negative Belohnungen):** Der Roboter zappelt zufällig herum und fällt sofort um. Jede Episode endet innerhalb von Sekunden mit einem Absturz.

**Mittleres Training:** Der Roboter entdeckt, dass abwechselnde Beinbewegungen eine Vorwärtsbewegung erzeugen. Er beginnt, kleine, unbeholfene Schritte zu machen — die Belohnung wird weniger negativ.

**Spätes Training:** Ein flüssiger, effizienter **Gang (Gait)** entsteht. Ein Gang ist ein sich wiederholendes Bewegungsmuster, wie abwechselnde linke und rechte Schritte. Der Roboter passt sich dynamisch an unebenes Gelände an, indem er seine LIDAR-Sensoren nutzt, um seine Schritte in Echtzeit anzupassen.

**Beispiel aus dem echten Leben:** Ein Baby lernt laufen:
1. Fällt sofort um (negative Belohnung).
2. Macht einen Schritt, fällt hin (etwas weniger negativ).
3. Macht ein paar Schritte (kleine positive Belohnung).
4. Läuft durch den Raum (große positive Belohnung!).

---

## Warum BipedalWalker PPO benötigt (und nicht REINFORCE)

- **BipedalWalker-Episoden** können bis zu 1600 Schritte dauern (viel länger als CartPole!).
- **Belohnungen sind spärlich** — die Belohnungen für Vorwärtsbewegung sind pro Schritt winzig.
- **REINFORCE würde tausende** kompletter Episoden benötigen, um ein brauchbares Signal zu erhalten.

PPOs n-Step-Updates mit [GAE (Generalized Advantage Estimation)](./ppo_scratch_explained.md#gae-intelligentere-vorteilsschatzungen) ermöglichen es dem Roboter, aus unvollständigen Episoden zu lernen:
> „Auch wenn ich nach 50 Schritten hingefallen bin, zeigten diese Schritte EINIGEN Vorwärtsfortschritt. Lass mich eine 50-Schritt-Return-Schätzung verwenden, anstatt auf das Ende der Episode zu warten.“

---

## Ergebnisse

Nach 500 Updates (≈ 1 Million Schritte in der Umgebung):
- Der Roboter macht sichtbare Fortschritte vom zufälligen Zappeln hin zu einer gewissen Vorwärtsbewegung.
- Beständige Verbesserung in der Lernkurve.
- Die vollständige Konvergenz zu einer Belohnung > 300 erfordert mehr Training (5-10 Millionen Schritte).

Die Lernkurve zeigt die charakteristische „S-Kurve“ der kontinuierlichen Steuerung:
1. Langsamer Anfangsfortschritt (Lernstabilität).
2. Schnelle Verbesserung (Entdeckung des Gangmusters).
3. Allmähliche Verfeinerung (Optimierung des Gangs).

---

## Wichtige Erkenntnisse

| Konzept | Einfache Erklärung |
|---------|---------------|
| **Gauß-Policy** | Anstatt aus einem Menü zu wählen, wirf einen Dartpfeil auf einen Wertebereich |
| **μ (Mittelwert)** | Wohin die Policy „zielt“ |
| **σ (Standardabweichung)** | Wie viel Zufälligkeit / Exploration die Policy verwendet |
| **log_std als lernbarer Parameter** | Eine globale Explorationsrate, die durch gradientenbasierte Optimierung aktualisiert wird — genau wie jedes andere Gewicht im Netzwerk |
| **Kontinuierliche Steuerung** | Steuerung von reellwertigen Ausgaben (Drehmomente, Kräfte, Winkel) |

---

## Was kommt als Nächstes?

PPO hat viele **Hyperparameter** — Einstellungen, die Sie vor Beginn des Trainings wählen (im Gegensatz zu *Parametern* wie Netzwerkgewichten, die automatisch gelernt werden). Beispiele sind `clip_eps`, Lernrate, Anzahl der Epochen und Batch-Größe.

Wie empfindlich reagiert PPO auf diese Entscheidungen? `ppo_hyperparams.py` führt Experimente durch, bei denen jeder Hyperparameter systematisch variiert wird, und zeigt die Auswirkungen auf Lerngeschwindigkeit und Stabilität.
