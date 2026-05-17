# Het meerarmige bandietenprobleem 🎰

## Wat is het?

Stel je voor dat je op een verjaardagsfeestje bent en dat er **10 verschillende snoeppotten** zijn.
In elke pot zit snoep, maar sommige potten bevatten *lekkere* snoepjes en sommige potten hebben dat wel
*niet zo lekker* snoep. Je weet niet welke pot het beste is, je moet ze proberen!

Elke keer dat je in een pot reikt, krijg je wat snoep. Jouw taak is:

> **Krijg zoveel mogelijk lekkere snoepjes!**

Dat is het probleem van de meerarmige bandieten! In plaats van snoeppotten bellen wetenschappers
ze "armen" (zoals armen op een gokautomaat). Elke arm levert je een prijs op, maar de
prijzen zijn elke keer anders.

---

## De grote vraag: moet ik nieuwe potten proberen of bij mijn favoriet blijven?

Dit is het moeilijkste deel! Laten we zeggen dat je pot nr. 3 hebt geprobeerd en dat het behoorlijk goed was.
Nu heb je een keuze:

- **Exploit**: blijf pot nr. 3 kiezen omdat je al weet dat het goed is.
- **Verkennen**: Probeer een nieuwe pot — misschien is pot nr. 7 zelfs *beter*!

Als je alleen het eerste potje kiest dat je lekker vindt, mis je misschien het superlekkere potje
pot. Maar als je *altijd* nieuwe potten probeert, gebruik je nooit wat je al hebt geleerd!

**Voorbeeld uit de praktijk:** Denk aan uw favoriete restaurant. Jij bestelt altijd
kipnuggets (exploit!), maar misschien is de pizza nog lekkerder. Als je nooit
probeer iets nieuws, je zult het nooit weten!

---

## De Epsilon-Greedy-strategie {#the-epsilon-greedy-strategy}

Een slimme manier om dit op te lossen heet **epsilon-greedy** (epsilon is slechts de
Griekse letter ε, gezegd als "ep-sih-lon"):

1. **Meestal (zeg 90%)**: Kies de pot waarvan jij *denkt* dat deze de beste is.
2. **Soms (zeg 10%)**: kies een *willekeurige* pot om te verkennen!

De 10% ontdekkingsreizen helpen je betere potten te ontdekken. De 90% die uitbuit
Met reizen kun je gebruiken wat je al hebt geleerd.

---

## Wat onze code heeft gevonden

We hebben 10 armen (snoeppotten) getest met 200 verschillende kinderen, elk 1000 plectrums:

| Strategie | % van de tijd Het kiezen van de beste pot |
|----------|--------------------------------|
| **Nooit verkennen (ε=0)** | 14,5% — vroeg vastgelopen, nooit de beste gevonden! |
| **Verken 1% van de tijd (ε=0,01)** | 37,6% — vond langzaam de beste pot |
| **Verken 10% van de tijd (ε=0,10)** | **74,2%** — snel geleerd, meestal de beste gekozen! |

**Les**: met een beetje onderzoek kom je al een heel eind!

---

## Voorbeelden uit het echte leven

- **Netflix-aanbevelingen**: als Netflix je een film laat zien, zul je dat waarschijnlijk doen
  leuk vinden (uitbuiten) of iets nieuws voorstellen (verkennen)?
- **Arts kiest een behandeling**: Gebruik de behandeling die gewoonlijk werkt (exploit)
  of een nieuwe proberen die misschien nog beter is (verkennen)?
- **Een bij die bloemen zoekt**: Moet hij de bloemen blijven bezoeken waarvan hij weet dat ze die hebben?
  nectar, of naar een nieuw veld vliegen?

---

## Sleutelwoorden om te onthouden

- **Arm**: een van de keuzes (zoals een snoeppot)
- **Beloning**: wat je krijgt als je een arm kiest (zoals snoep)
- **Exploit**: gebruik wat je al weet dat goed is
- **Verkennen**: probeer iets nieuws om meer te leren
- **Epsilon (ε)**: de kans die je verkent in plaats van exploiteert

Het grote idee: **Je moet het proberen van nieuwe dingen combineren met het gebruiken van wat je weet!**