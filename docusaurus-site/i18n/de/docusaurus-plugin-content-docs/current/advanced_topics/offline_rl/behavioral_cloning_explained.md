# Behavioral Cloning (BC) 🐒

## Was ist das?

Stellen Sie sich vor, Sie möchten Tennis spielen lernen. Sie schauen sich hunderte Stunden aufgezeichneter Wimbledon-Matches an und **kopieren einfach, was die Spieler tun**. Sie überlegen nicht, ob ihr Schlag der *beste* Schlag war – Sie passen einfach Ihre Körperposition an ihre an und schwingen auf die gleiche Weise.

Das ist Behavioral Cloning (Verhaltensklonen). **Keine Belohnung. Keine Planung. Nur Imitation.**

In RL-Begriffen: Nehmen Sie den Datensatz von `(Zustand, Aktion)`-Paaren und trainieren Sie ein neuronales Netzwerk darauf, die Aktion aus dem Zustand vorherzusagen, genau wie ein Bildklassifizierungsmodell Katze gegen Hund vorhersagt. Das „Label“ (Etikett) ist die Aktion, die der Datensammler ausgeführt hat.

---

## Wie es sich von „echtem“ Offline RL unterscheidet

| Ansatz | Verwendet Belohnungen? | Kann die Daten schlagen? |
|----------|---------------|---------------------|
| **BC**   | ❌ Nein         | ❌ Nein — bestenfalls erreicht es die durchschnittliche Datenqualität |
| **CQL** (und ähnliche) | ✅ Ja | ✅ Ja — kann gute Übergänge aus gemischten Daten zusammenfügen |

BC ist die „Supervised-Learning-Sicht“ auf RL. Es ist unglaublich einfach, oft überraschend stark und der universelle Baseline-Wert. **Wenn ein Offline-RL-Algorithmus BC auf demselben Datensatz nicht schlagen kann, hat er nichts erreicht.**

---

## Beispiele aus dem echten Leben

- **Autofahren lernen aus Dashcam-Aufnahmen.** Schauen Sie auf die Straße, sagen Sie den Lenkradwinkel voraus, den der Mensch benutzt hat. Zwei wegweisende Beispiele:
  - **ALVINN (1989)** — der allererste neuronale Netzwerk-Fahrer; ein winziges 3-Schicht-Netzwerk, das auf Kamera- und Lasereingaben trainiert wurde, um einen Van über Autobahnen zu steuern.
  - **NVIDIA PilotNet (2016)** — ein modernes tiefes CNN, das End-to-End auf Dashcam-Aufnahmen trainiert wurde; es lernte Spurhalten und grundlegendes Lenken rein durch Nachahmung menschlicher Fahrer, ohne handgefertigte Regeln.
- **Lehrling kopiert einen Meisterkoch.** „Was auch immer der Koch tut, tue ich auch.“ Funktioniert hervorragend, wenn der Koch großartig ist; bringt einen schlechten Koch hervor, wenn der Koch schlecht ist.
- **GitHub Copilot.** Auto-Complete ist darauf trainiert, vorherzusagen: „Welchen Code würde ein Mensch als Nächstes tippen?“ – reine Imitation von Quellcode-Logs.
- **Nachahmung des älteren Geschwisters.** Kinder tun dies jahrelang, bevor sie anfangen darüber nachzudenken, *warum* das ältere Geschwisterkind tut, was es tut.

---

## Die Mathematik (eine Zeile)

Minimieren Sie für jedes `(s, a)` im Datensatz:

```
loss = -log π(a | s)        (Cross-Entropy)
```

Das ist alles. Die Policy `π` ist nur ein MLP, das Aktions-Logits ausgibt; das Training ist identisch mit MNIST. Lassen Sie uns den Fachjargon aufschlüsseln:
- **`π` (Pi):** Das Standardsymbol für „Policy“ (Strategie) – die Regel oder das neuronale Netzwerk, das entscheidet, was zu tun ist.
- **MLP (Multi-Layer Perceptron):** Ein einfaches, standardmäßiges neuronales Netzwerk.
- **Logits:** Die rohen, nicht normalisierten Werte, die das Netzwerk ausspuckt, bevor wir sie in Wahrscheinlichkeiten umwandeln.
- **Cross-Entropy:** Die Standardformel zur Bestrafung eines Modells, wenn es der richtigen Antwort eine niedrige Wahrscheinlichkeit zuweist.
- **MNIST:** Der berühmte Anfänger-Datensatz für handgeschriebene Ziffern.

Einen Agenten darauf zu trainieren, ein Spiel über BC zu spielen, ist buchstäblich identisch mit dem Training eines Netzwerks zur Erkennung handgeschriebener Ziffern in MNIST. In MNIST ist die Eingabe ein Bild und die Ausgabe eine Ziffer (0-9). In BC ist die Eingabe der Spielzustand und die Ausgabe die Aktion (z. B. „nach links bewegen“).

---

## Was unser Code tut

Das Skript `behavioral_cloning.py`:

1. **Lädt alle vier Datensätze**, die durch `d4rl_dataset.py` erstellt wurden
   (`random`, `medium`, `expert`, `medium-replay`).
2. **Trainiert für jeden Datensatz eine separate BC-Policy** für 10.000 Gradientenschritte der Cross-Entropy. Die Belohnungsspalte wird komplett ignoriert.
3. **Evaluiert** alle 2.500 Schritte die aktuelle Policy, indem sie gierig (greedy) in der echten CartPole-v1-Umgebung ausgeführt wird (20 Episoden, gemittelt).
4. Erstellt Diagramme:
   - Ein Balkendiagramm: finaler BC-Return pro Datensatz.
   - Ein Lernkurven-Diagramm: wie jede BC-Variante im Laufe des Trainings steigt.

---

## Was Sie sehen sollten

Ein typischer Durchlauf gibt Folgendes aus:

```
Final evaluation returns:
  BC on random          ->    ~20  ± ein paar   (≈ zufälliges Spiel)
  BC on medium          ->   ~150  ± groß       (≈ die Medium-Policy)
  BC on expert          ->   ~480  ± klein      (≈ die Expert-Policy)
  BC on medium-replay   ->    ~60  ± groß       (≈ der DURCHSCHNITT der gemischten Daten)
```

Das Balkendiagramm macht die Geschichte deutlich: **Der Return von BC folgt dem durchschnittlichen Return des Datensatzes.** Er kann nicht über diese Obergrenze hinausgehen, da BC keine Möglichkeit hat, die „guten“ Teile eines gemischten Datensatzes gegenüber den „schlechten“ Teilen zu bevorzugen – beide sind gleichermaßen gültige Imitationsziele.

Das ist die Kernaussage: **BC erbt die Obergrenze der Daten.**

---

## BC vs CQL — Der klarste Vergleich

Auf dem **medium-replay** Datensatz (der realistischste Fall mit gemischter Qualität):

| Methode | Ungefährer finaler Return | Warum? |
|--------|--------------------:|------|
| BC     | ~60   | Imitiert den *Durchschnitt* aus frühen fehlgeschlagenen Versuchen + späteren guten Durchläufen |
| CQL    | ~400+ | Verwendet Belohnungen, um Übergänge mit hohen Q-Werten zu bevorzugen; fügt eine gute Policy aus gemischten Daten zusammen |

CQL **schlägt also die Daten**, BC **entspricht den Daten**. Das ist der ganze Grund, warum Offline RL ein Forschungsfeld ist und nicht nur „Imitation Learning“. Wenn die Daten von gemischter Qualität sind (was reale Logs immer sind), holen belohnungsbewusste Methoden mehr heraus.

Bei **Expert-Daten** dreht sich der Vergleich um: BC erreicht Expert-Niveau (~480). Sie fragen sich vielleicht, warum CQL hier „gleichzieht“ anstatt zu verlieren. Da CQL so konzipiert ist, dass es *konservativ* ist und Aktionen bestraft, die nicht im Datensatz enthalten sind, tut es am Ende genau das, was der Experte getan hat. Es kann den Experten nicht schlagen (da die maximal mögliche Punktzahl bereits erreicht ist), aber es bricht die Strategie des Experten auch nicht aktiv. Es zieht einfach mit der Leistung von BC gleich.

Dies ist der berühmte „Datenqualität vs. Algorithmus“-Abgleich:

```
                            EXPERT-Daten  →  BC gewinnt, CQL zieht gleich
   Algorithmus-Komplexität   ↑         
                            GEMISCHTE Daten → CQL schlägt BC deutlich
                            
                            ZUFÄLLIGE Daten → Alle scheitern; Exploration erforderlich
```

---

## Wo BC im modernen RL zu finden ist

- **Pre-Training für Online RL.** Viele moderne Systeme (RT-1, Voyager, Spiele-Bots) beginnen mit BC auf Demonstrationen und führen dann ein Online-Feintuning mit PPO/SAC durch.
- **RLHF.** Schritt 1 von InstructGPT ist Supervised Fine-Tuning – reines BC auf von Menschen geschriebenen Antworten. PPO + Belohnungsmodell kommen später.
- **DAgger (Ross et al., 2011).** Eine clevere Erweiterung, um das Problem des **Fehler-Compounding** (Fehleranhäufung) zu beheben.
  *Warum ist Fehler-Compounding ein Problem, wenn BC perfekt klont?* Selbst wenn ein BC-Modell zu 99 % genau ist, passiert dieser 1 %-Fehler irgendwann. Wenn das passiert, gerät der Agent in einen Zustand, den er im perfekt gefahrenen Datensatz nie gesehen hat. Weil er verwirrt ist, macht er einen größeren Fehler und entfernt sich noch weiter von den bekannten Daten, was sich zu einem totalen Scheitern summiert (wie das Herunterfahren von einer Klippe).
  *Die Lösung:* Wir könnten den Experten einfach ewig fahren lassen, aber Expertenzeit ist teuer. Stattdessen lässt DAgger die BC-Policy fahren. Wenn die Policy einen Fehler macht und in einen seltsamen Zustand gerät, halten wir inne, fragen den Experten „was würdest du *genau hier* tun?“ und fügen das dem Datensatz hinzu. Wir „befragen den Experten nur erneut zu Zuständen, die die BC-Policy besucht“, weil wir den Experten nur brauchen, um uns zu lehren, wie wir uns von unseren eigenen spezifischen Fehlern erholen, anstatt ihn immer zu befragen.
- **Decision Transformer (Chen et al., 2021).** Ein „smartes“ BC, das die Aktionsvorhersage von einem gewünschten *Return-to-go* abhängig macht und so Offline RL im Wesentlichen wieder in eine Next-Token-Vorhersage verwandelt.

---

## Wichtige Begriffe zum Merken

| Begriff | Bedeutung |
|------|---------|
| **Imitation Learning** | Oberbegriff für „kopiere den Demonstrator“; BC ist das einfachste Mitglied |
| **Compounding Error** | Ein kleiner BC-Fehler führt dich in Zustände, die der Datensatz nie gesehen hat, wo sich Fehler anhäufen |
| **Demonstrationsdaten** | Trajektorien, die von einem Experten erstellt wurden und als BC-Trainingssatz dienen |
| **Data Ceiling** | Der Return von BC ist durch den durchschnittlichen Return im Datensatz begrenzt |
| **DAgger** | Eine interaktive Lösung für den Compounding Error |

---

## Zusammenfassung in einem Satz

> **Behavioral Cloning ist nur Supervised Learning auf (Zustand, Aktion)-Paaren – stark, wenn die Daten gut sind, hilflos, wenn die Daten gemischt sind.**
