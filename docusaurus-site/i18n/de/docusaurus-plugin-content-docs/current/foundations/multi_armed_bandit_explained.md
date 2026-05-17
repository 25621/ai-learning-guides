# Das Multi-Armed Bandit Problem 🎰

## Was ist das?

Stellen Sie sich vor, Sie sind auf einer Geburtstagsparty und dort stehen **10 verschiedene Gläser mit Süßigkeiten**. In jedem Glas sind Süßigkeiten, aber manche Gläser enthalten *leckere* Naschereien und andere eher *weniger leckere*. Sie wissen nicht, welches Glas das beste ist – Sie müssen sie ausprobieren!

Jedes Mal, wenn Sie in ein Glas greifen, bekommen Sie eine Süßigkeit. Ihre Aufgabe ist:

> **Holen Sie sich so viele leckere Süßigkeiten wie möglich!**

Das ist das Multi-Armed Bandit Problem! Anstatt von Süßigkeitsgläsern sprechen Wissenschaftler von „Armen“ (wie die Hebel bei einem Spielautomaten, dem „einarmigen Banditen“). Jeder Arm gibt Ihnen einen Preis, aber die Preise sind jedes Mal anders.

---

## Die große Frage: Soll ich neue Gläser probieren oder bei meinem Favoriten bleiben?

Das ist der schwierigste Teil! Nehmen wir an, Sie haben Glas Nr. 3 probiert und es war ziemlich gut. Nun haben Sie die Wahl:

- **Exploit (Ausnutzen)**: Greifen Sie weiterhin in Glas Nr. 3, weil Sie bereits wissen, dass es gut ist.
- **Explore (Erkunden)**: Probieren Sie ein neues Glas aus – vielleicht ist Glas Nr. 7 sogar *noch besser*!

Wenn Sie immer nur das erste Glas nehmen, das Ihnen geschmeckt hat, verpassen Sie vielleicht das super-leckere Glas. Aber wenn Sie *immer* neue Gläser ausprobieren, nutzen Sie nie das aus, was Sie bereits gelernt haben!

**Beispiel aus dem echten Leben:** Denken Sie an Ihr Lieblingsrestaurant. Sie bestellen immer Chicken Nuggets (Exploit!), aber vielleicht ist die Pizza dort noch viel besser. Wenn Sie nie etwas Neues probieren, werden Sie es nie erfahren!

---

## Die Epsilon-Greedy-Strategie

Eine kluge Art, dies zu lösen, nennt man **Epsilon-Greedy** (Epsilon ist einfach der griechische Buchstabe ε):

1. **Meistens (sagen wir zu 90 %)**: Wähle das Glas, von dem du *denkst*, dass es das beste ist.
2. **Manchmal (sagen wir zu 10 %)**: Wähle ein *zufälliges* Glas, um zu erkunden!

Die 10 % Erkundungstrips helfen Ihnen, bessere Gläser zu entdecken. Die 90 % Ausnutzungstrips ermöglichen es Ihnen, Ihr gelerntes Wissen zu nutzen.

---

## Was unser Code herausgefunden hat

Wir haben 10 Arme (Süßigkeitsgläser) mit 200 verschiedenen Kindern getestet, jeweils 1000 Versuche:

| Strategie | % der Zeit, in der das beste Glas gewählt wurde |
|----------|----------------------------------|
| **Niemals erkunden (ε=0)** | 14,5 % — früh steckengeblieben, nie das beste gefunden! |
| **Zu 1 % erkunden (ε=0.01)** | 37,6 % — das beste Glas wurde langsam gefunden |
| **Zu 10 % erkunden (ε=0.10)** | **74,2 %** — schnell gelernt, meistens das beste gewählt! |

**Lektion**: Ein kleines bisschen Erkundung bringt einen sehr weit!

---

## Beispiele aus dem echten Leben

- **Netflix-Empfehlungen**: Sollte Netflix einen Film zeigen, der dir wahrscheinlich gefällt (Exploit), oder etwas Neues vorschlagen (Explore)?
- **Ärzte bei der Behandlungswahl**: Die Behandlung anwenden, die normalerweise funktioniert (Exploit), oder eine neue ausprobieren, die vielleicht noch besser hilft (Explore)?
- **Eine Biene bei der Blumensuche**: Sollte sie weiterhin die Blumen besuchen, von denen sie weiß, dass sie Nektar haben, oder zu einem neuen Feld fliegen?

---

## Wichtige Begriffe zum Merken

- **Arm**: Eine der Wahlmöglichkeiten (wie ein Süßigkeitsglas)
- **Belohnung (Reward)**: Was du bekommst, wenn du einen Arm wählst (wie eine Süßigkeit)
- **Exploit (Ausnutzen)**: Das nutzen, von dem du bereits weißt, dass es gut ist
- **Explore (Erkunden)**: Etwas Neues ausprobieren, um mehr zu lernen
- **Epsilon (ε)**: Die Chance, dass du erkundest, anstatt auszunutzen

Die Kernbotschaft: **Du musst ein Gleichgewicht finden zwischen dem Ausprobieren neuer Dinge und dem Nutzen dessen, was du bereits weißt!**
