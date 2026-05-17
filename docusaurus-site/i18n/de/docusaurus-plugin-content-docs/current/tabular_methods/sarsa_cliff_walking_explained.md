# SARSA für Cliff Walking 🏔️

## Was ist das?

Stellen Sie sich einen **sehr langen Flur** mit einer **schrecklichen Klippe** an einer Seite vor. Wenn Sie von der Klippe stürzen, müssen Sie den ganzen Weg zurück zum Start gehen! Ihr Ziel ist es, so schnell wie möglich von einem Ende zum anderen zu gelangen, ohne zu fallen.

**SARSA** ist ein Roboter, der durch Üben lernt, diesen Flur entlangzugehen. Er lernt, einen *sicheren Pfad* zu nehmen, der die Klippe meidet – auch wenn dieser ein wenig länger ist –, weil er weiß, dass er beim Erkunden (Exploration) versehentlich nah an den Rand rutschen könnte!

---

## Die Kernidee: Aus dem lernen, was man tatsächlich tut

SARSA steht für: **S**tate → **A**ction → **R**eward → **S**tate → **A**ction
(Zustand → Aktion → Belohnung → Zustand → Aktion)

Dies sind die fünf Informationen, die SARSA zum Lernen nutzt:

1. **S** — Wo bin ich gerade? (aktueller Zustand)
2. **A** — Welche Aktion habe ich tatsächlich ausgeführt?
3. **R** — Welche Belohnung habe ich erhalten?
4. **S** — Wo bin ich gelandet?
5. **A** — Welche Aktion werde ich *als Nächstes tatsächlich ausführen*?

Das letzte „A“ ist das, was SARSA so besonders macht! Es aktualisiert seine Werte unter Verwendung der Aktion, die es *tatsächlich als Nächstes ausführen wird* (auch wenn das ein zufälliger Erkundungsschritt ist), und nicht unter Verwendung der perfekten idealen Aktion.

**Beispiel aus dem echten Leben:** Denken Sie an das Fahrradfahren lernen. Wenn Sie wissen, dass Sie manchmal wahllos wackeln (Exploration), halten Sie etwas mehr Abstand zu parkenden Autos — weil Sie wissen, dass Sie in Ihrem wackeligen Zustand ausscheren könnten! SARSA macht genau das: Es lernt einen sicheren Pfad, weil es seine eigenen zufälligen Fehler einkalkuliert.

---

## Die Cliff-Walking-Karte

```
[ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]
[ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]
[ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]
[S][C][C][C][C][C][C][C][C][C][C][G]
   ← ← ← ← KLIPPE ← ← ← ← ←
```

- **S** = Start (unten links)
- **G** = Ziel (unten rechts)
- **C** = Klippe (Cliff) — hier hineinzutreten bedeutet -100 Belohnung und Neustart!
- Jeder andere Schritt = -1 Belohnung.

---

## Was unser Code herausgefunden hat

Nachdem SARSA über 500 Episoden trainiert wurde:

| Ergebnis | Wert |
|--------|-------|
| Durchschnittliche Belohnung (letzte 50 Ep.) | **-21,6** |
| Belohnung auf dem optimalen (riskanten) Pfad | -13 |

Die von SARSA gelernte Strategie führt **oben am Gitter entlang** — ein sicherer Umweg! Das kostet ein paar zusätzliche Schritte (-21,6 statt -13), aber der Agent fällt während des Trainings fast nie von der Klippe.

---

## Beispiele aus dem echten Leben

- **Pflegekraft bei der Medikamentengabe**: Verwendet das bewährte Sicherheitsprotokoll (sicherer Pfad), auch wenn eine etwas schnellere Methode existiert, da kleine Fehler (Exploration) gefährlich sein könnten.
- **Flugzeugpiloten**: Folgen strengen Checklisten (sichere Pfade), selbst wenn Abkürzungen schneller erscheinen könnten, um menschliches Versagen einzukalkulieren.
- **Kochen lernen**: Mit bewährten Rezepten beginnen (sicher), anstatt riskante Abkürzungen zu nehmen.

---

## Wichtige Begriffe zum Merken

- **On-Policy**: Lernt über die Strategie, die tatsächlich gerade verwendet wird (einschließlich ihrer zufälligen Fehler).
- **SARSA-Update**: Nutzt die *tatsächliche* nächste Aktion, nicht die theoretisch beste.
- **Sicherer Pfad**: Ein längerer Weg, der Gefahren meidet und Explorationsfehler berücksichtigt.
- **TD (Temporal Difference) Control**: Aktualisierung der Werte nach jedem einzelnen Schritt (kein Warten auf das Ende der Episode).

Die Kernbotschaft: **SARSA ist ehrlich — es lernt aus dem, was es tatsächlich tut, nicht aus dem, was es sich wünschen würde zu tun. Das macht es vorsichtig und sicher in der Nähe von Gefahren!**
