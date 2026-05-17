# Monte Carlo Control für Blackjack 🃏

## Was ist das?

Haben Sie jemals ein Kartenspiel gespielt, bei dem Sie entscheiden mussten: **„Nehme ich noch eine Karte oder bin ich zufrieden mit dem, was ich habe?“**

**Blackjack** (auch „17 und 4“ genannt) ist genau das! Sie möchten, dass Ihre Karten in der Summe so nah wie möglich an 21 herankommen, ohne diese Zahl zu überschreiten. Wenn Sie über 21 kommen, haben Sie sich „überkauft“ (Bust) und verloren!

**Monte Carlo Control** ist die Methode, mit der ein Roboter lernt, Blackjack zu spielen – indem er *tausende vollständige Spiele* spielt und sich merkt, was funktioniert hat und was nicht.

---

## Die Kernidee: Aus vollständigen Geschichten lernen

Das Wort „Monte Carlo“ stammt von dem berühmten Casino in Monaco. In der Mathematik bedeutet es: **Verwende Zufallsexperimente, um etwas zu lernen**.

So funktioniert es:

1. **Spiele ein vollständiges Spiel** (eine komplette Episode) mit deiner aktuellen Strategie.
2. **Schau dir an, was passiert ist**: Hast du gewonnen? Verloren? Unentschieden gespielt?
3. **Gehe rückwärts vor**: War es eine gute Idee, bei 17 noch eine Karte zu ziehen (Hit)? Was war bei 14?
4. **Aktualisiere dein Gedächtnis**: Merke dir, ob jede Entscheidung zum Sieg oder zur Niederlage geführt hat.

Wiederholen Sie dies für **500.000 Spiele**, und Sie werden sehr gut darin!

**Beispiel aus dem echten Leben:** Stellen Sie sich vor, Sie lernen kochen, indem Sie 500.000 Mahlzeiten zubereiten. Jedes Mal merken Sie sich genau, was Sie getan haben – und ob das Essen geschmeckt hat. Nach genügend Versuchen wissen Sie: „Zu viel Salz in diesem Schritt hat es immer schlecht gemacht.“ Monte Carlo funktioniert genauso!

---

## Hauptunterschied zu SARSA und Q-Learning

SARSA und Q-Learning aktualisieren ihr Wissen **nach jedem einzelnen Schritt** (sogar mitten in einer Episode). Monte Carlo wartet, bis die **gesamte Episode beendet ist**, und blickt dann auf alles zurück.

| Methode | Wann wird aktualisiert? | Benötigt vollständige Episode? |
|--------|---------------|------------------------|
| **TD (SARSA, Q-Learning)** | Nach jedem Schritt | Nein |
| **Monte Carlo** | Nach jeder vollständigen Episode | Ja |

Dies macht Monte Carlo einfacher zu verstehen, aber man kann erst lernen, wenn eine Episode endet.

---

## Der Blackjack-Zustand (State)

Der Roboter betrachtet in jeder Runde drei Dinge:
1. **Meine Kartensumme** (12 bis 21).
2. **Welche Karte zeigt der Dealer?** (Ass bis 10).
3. **Habe ich ein nutzbares Ass?** (Ein Ass kann als 1 oder 11 zählen).

Basierend auf diesen 3 Informationen entscheidet er: **Karte ziehen (Hit) oder stehen bleiben (Stick)**?

---

## Was unser Code herausgefunden hat

Nach **500,000 Spielen** Blackjack:

| Ergebnis | Prozentsatz |
|---------|------------|
| **Gewonnen** | **43,1 %** |
| **Unentschieden** | 8,9 % |
| **Verloren** | 48,0 % |

Dies liegt nahe an der mathematisch optimalen „Basisstrategie“ (ca. 42–43 % Siege)! Der Roboter hat gelernt, wann er ziehen und wann er stehen bleiben muss – allein durch das Spielen von Partien und das Erinnern.

Die gelernte Strategie zeigt:
- **Ziehen (Hit)**, wenn die Summe niedrig ist (man überkauft sich unwahrscheinlich).
- **Stehen bleiben (Stick)**, wenn die Summe hoch ist (man könnte sich überkaufen, wenn man noch eine Karte nimmt).
- Ein **nutzbares Ass** ermöglicht es, aggressiver zu sein (es kann bei Bedarf von 11 auf 1 wechseln).

---

## Beispiele aus dem echten Leben

- **Wettervorhersage**: Monte-Carlo-Simulationen lassen tausende „Was-wäre-wenn“-Szenarien laufen, um das Wetter von morgen vorherzusagen.
- **Aktienmarktmodellierung**: Analysten simulieren tausende möglicher Zukünfte, um Risiken einzuschätzen.
- **Schach lernen**: Ein Spieler analysiert ganze Partien (nicht nur einzelne Züge), um zu verstehen, welche Strategie zum Sieg geführt hat.

---

## Wichtige Begriffe zum Merken

- **Episode**: Ein vollständiges Spiel vom Anfang bis zum Ende.
- **Return (G)**: Die gesamte gesammelte Belohnung von einem Punkt im Spiel bis zum Ende.
- **Every-visit MC**: Den Score für einen Zustand jedes Mal aktualisieren, wenn er in einer Episode besucht wird.
- **Kein Bootstrapping**: Monte Carlo verwendet keine Schätzungen zukünftiger Werte – es wartet auf das echte Ergebnis!
- **ε-soft Policy** (ε = Epsilon): Meistens die beste bekannte Aktion wählen, aber manchmal zufällig explorieren.

Die Kernbotschaft: **Monte Carlo lernt durch das Spielen vieler vollständiger Spiele. Es ist wie das Lernen aus Erfahrung – du merkst dir alles, was passiert ist, und findest heraus, was zum Sieg geführt hat!**
