# Matrix Games: de eenvoudigste wereld met meerdere agenten 🎲

## Wat is een matrixspel?

Stel je voor dat jij en een vriend elk een handteken kiezen – **steen, papier of schaar** —
*tegelijkertijd*. Je ziet elkaars keuze niet. De winnaar wordt bepaald door
een kleine tafel:

|        | Rock | Paper | Scissors |
|--------|:----:|:-----:|:--------:|
| Steen     |  0,0  | -1,+1 | +1,-1 |
| Papier    | +1,-1 |  0,0  | -1,+1 |
| Schaar | -1,+1 | +1,-1 |  0,0  |

Die tafel is de *hele wereld* van het spel. Geen beweging, geen tijd, geen kaart.
Gewoon een beslissing in één keer. We noemen dit een **matrixspel** vanwege de uitbetaling
matrix is de gehele omgeving.

Matrix-spellen zijn de schoonste plek om **multi-agent RL** te bestuderen, omdat de
Het enige dat tijdens de training kan veranderen, is het *beleid* van elke speler – de
waarschijnlijkheid dat elke actie wordt gekozen.

---

## Waarom het "Multi-Agent" is

Bij single-agent RL ligt de omgeving vast: de wind waait altijd hetzelfde
Zo beweegt de vloer nooit. De agent verbetert en wint uiteindelijk.

In een matrixspel is je "omgeving" *een ander leermiddel*. Zoals ze krijgen
slimmer, wat voor jou als een goede zet geldt *veranderingen*. Dit wordt genoemd
**niet-stationariteit**, en dit is het centrale probleem van multi-agent RL.

> Als je Rock blijft spelen, zal je tegenstander uiteindelijk Paper gaan spelen
> altijd. Je stapt dus over op Schaar. Dus schakelen ze over naar Rock. Dus je stapt over naar
> Papier... enzovoort. De ‘beste zet’ blijft nooit op zijn plaats.

De klassieke oplossing is **gemengde strategieën**: kies niet één actie
deterministisch – willekeurig maken op een manier die de tegenstander niet kan exploiteren.

---

## De drie spellen die we spelen

### 1) Steen-papier-schaar (nulsom)
- De winst van de ene speler is het verlies van de andere.
- Het **Nash-evenwicht** is: elke speler kiest elke actie met waarschijnlijkheid
  ⅓.  Elke afwijking is exploiteerbaar.
- We verwachten dat onze twee Q-leerlingen rond ⅓-⅓-⅓ wiebelen – nooit perfect
  stabiel, want elke keer dat de één afdrijft, reageert de ander.

### 2) Gevangenendilemma (algemene som)
Twee verdachten worden afzonderlijk verhoord:

|           | Cooperate | Defect |
|-----------|:---------:|:------:|
| Samenwerken |   3, 3    |  0, 5  |
| Defect    |   5, 0    |  1, 1  |

- 'Defect' verslaat 'Samenwerken', ongeacht wat de ander doet - het is een
  **dominante strategie**.
- Beide spelers zijn rationeel → beide defect → beiden krijgen 1, ook al
  (Samenwerken, samenwerken) was elk 3. Egoïstisch beste antwoord vernietigt de groep
  welzijn.
- We verwachten dat Q-learning netjes convergeert naar (Defect, Defect).

### 3) Hertenjacht (coördinatie)
Twee jagers kunnen samen een hert neerhalen (enorme prijs), of elk genoegen nemen met een
haas (kleine maar veilige prijs):

|       | Stag | Hare |
|-------|:----:|:----:|
| Hert  | 4, 4 | 0, 3 |
| Haas  | 3, 0 | 2, 2 |

- (Stag, Stag) is **uitbetalingsdominant** — het beste voor beide.
- (Hare, Hare) is **risicodominant** — veilig als u uw partner niet vertrouwt.
- De uitkomst hangt af van de initiële voorwaarden: onafhankelijke Q-leerlingen komen vaak terecht
  in het *slechtere* (Haas, Haas) evenwicht omdat hazen veiliger zijn om te leren.

---

## Voorbeelden uit het echte leven

- **Prijzen in een duopolie.** Twee coffeeshops in dezelfde straat kiezen elk een
  Prijs elke ochtend. De vorm van de uitbetalingsmatrix bepaalt of ze
  eindigen op een hoge ‘coöperatieve’ prijs (goed voor hen, slecht voor klanten) of
  een lage, scherpe prijs.
- **Netwerkprotocollen.** Routers en zenders kiezen timingstrategieën; de
  De congestie-uitkomst van het netwerk wordt bepaald door de matrixspelachtige uitbetaling
  van er doorheen komen versus terugtrekken.
- **Bieden in een veiling.** Elke bieder kiest een bod zonder de anderen te kennen;
  de uitbetalingen zijn afhankelijk van de gehele vector. Het Nash-evenwicht is een *bieding
  strategie*, geen enkel getal.

---

## Wat onze code doet

Voor elk spel:
1. Creëer twee staatloze Q-leerlingen (Q is slechts één getal per actie – daar
   zijn geen staten in een 1-shot-spel).
2. Loop voor 20.000 stappen. Bij elke stap kiezen beide agenten een ε-hebzuchtige actie
   tegelijkertijd een beloning ontvangen en hun Q-waarden bijwerken.
3. Volg de **empirische actiefrequentie** van elke agent in stappen van 500
   venster. In plaats van alleen naar abstracte waarschijnlijkheden te kijken, tellen we welke acties ze recentelijk hebben gekozen (bijvoorbeeld: "in de laatste 500 rondes speelden ze 40% van de tijd Rock"). Dit geeft ons een real-time, praktisch beeld van hun veranderende strategie.
4. Teken de frequenties in de loop van de tijd, sla ze op in `outputs/<game>.png` en druk ze af
   de uiteindelijke Q-waarden.

### Wat je zou moeten zien

| Game | Expected outcome of the plot |
|------|----------------------------|
| **steen-papier-schaar** | Beide spelers zweven in de buurt van ⅓-⅓-⅓ maar trillen zichtbaar. De bochten volgen elkaar op: klassiek fietsgedrag. |
| **Gevangenendilemma** | De "Defect"-frequentie van beide spelers stijgt snel naar ~1,0. 'Samenwerken' wordt verpletterd. |
| **Hertenjacht** | De meeste willekeurige zaden nestelen zich op (Haas, Haas). Sommige gelukszaden bereiken (Stag, Stag) - probeer het zaad in het script te veranderen en zie hoe het omdraait. |

---

## Waar zelfstandig leren breekt

Onze agenten zijn *onafhankelijk*; ze zien alleen hun eigen beloning, nooit de
actie van de tegenstander of Q-waarden. Dit is de eenvoudigste basislijn en die is er ook
limieten:

- Het **kan geen convergentie garanderen** in spellen met een algemene som.
- Het kan vastlopen in **slechte evenwichten** (hertenjacht).
- Het **kan de tegenstander niet modelleren**.

Echte multi-agentalgoritmen lossen dit op door expliciet over de ander te redeneren
leerling. Dit is wat iedereen doet, in gewoon Engels:

| Algoritme | Kern idee                                                                                                                                                                                                                                                                               | Analogie uit het echte leven |
|-----------|--------------------------------------------------------------- --------------------------------------------------------------- --------------------------------------------------------------- ----------------------------------------------------------------|-------------------|
| **Fictief toneelstuk** | Houd bij hoe vaak je tegenstander elke actie heeft gekozen. Stel dat ze morgen zullen doen wat ze altijd hebben gedaan – kies dan je eigen beste reactie op die overtuiging.                                                                                                          | Het observeren van de gewoonten van een tegenstander tijdens veel schaakspellen en uw opening dienovereenkomstig aanpassen. |
| **CFR (Contrafeitelijke spijtminimalisatie)** | Vraag na elke ronde: "Hoeveel heb ik er spijt van gehad dat ik niet voor elkaar een actie heb gekozen?"* Verplaats geleidelijk de waarschijnlijkheid naar acties waarvan je spijt hebt dat je ze hebt overgeslagen. Wordt gebruikt in poker omdat het spellen met **onvolmaakte informatie** verwerkt (je ziet de kaarten van de tegenstander niet).                                  | Na een pokerhand, die je opnieuw speelt en denkt: *"Ik had meer moeten inzetten — dat zal ik de volgende keer doen."* |
| **LOLA (Leren met bewustzijn van de tegenstander)** | Je gradiëntstap houdt rekening met het feit dat de tegenstander *ook* gradiëntstappen maakt. Je optimaliseert je eigen update terwijl je anticipeert op de volgende update van de tegenstander: twee stappen vooruit in plaats van één.                                                                                    | Onderhandelen over een deal terwijl je denkt: *"Als ik X aanbiedt, reageren ze met Y, dus ik moet met Z beginnen."* |
| **MADDPG (Multi-Agent Deep Deterministic Policy Gradient)** | De *criticus* (waardeschatter) van elke agent is getraind met het **algemene beeld**: hij ziet ieders observaties en acties. De *actor* (het beleid dat wordt geïmplementeerd) gebruikt nog steeds alleen lokale informatie: dit is het CTDE-patroon (Centralized Training with Decentralized Execution). | Een basketbalcoach die het hele veld in de gaten houdt (gecentraliseerde criticus), maar elke speler leert alleen te reageren op wat hij kan zien (gedecentraliseerde acteur). |

Maar zelfstandig Q-learning is de juiste eerste stap. Je ziet de
Het niet-stationariteitsprobleem sloeg je in het gezicht en de oplossingen zijn logisch
daarna.

---

## Sleutelwoorden om te onthouden

| Word | Meaning |
|------|---------|
| **Uitbetalingsmatrix** | De tabel die een 1-shot multi-agentspel definieert |
| **Nash-evenwicht** | Een beleidsprofiel waarbij geen enkele agent kan verbeteren door af te wijken |
| **Gemengde strategie** | Een beleid dat randomiseert over meerdere acties |
| **Niet-stationariteit** | De omgeving (= andere agenten) blijft veranderen terwijl ze leert |
| **Onafhankelijke leerling** | Een agent die het bestaan ​​van andere leerlingen negeert |
| **Nulsom** | De winst van de ene agent is precies het verlies van de andere |
| **Algemeen bedrag** | Beide agenten kunnen winnen, beide kunnen verliezen, of iets daartussenin |

---

## Samenvatting van één zin

> **In matrixspellen is de "omgeving" een andere leerling - dus de beste zet
> blijft in beweging.**

Dit is het basisidee achter elk multi-agentalgoritme dat je tegenkomt
later, van zelfspel tot MADDPG tot MARL met communicatie.