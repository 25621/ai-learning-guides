# Optie-kritische architectuur

## Het grote idee: werken in hoofdstukken, niet woord voor woord {#the-big-idea-working-in-chapters-not-word-by-word}

Stel je voor dat je een roman schrijft. Je plant niet elk woord voordat je begint. In plaats daarvan denk je in **hoofdstukken**: "Hoofdstuk 1 introduceert de held. Hoofdstuk 2 is de zoektocht. Hoofdstuk 3 is de krachtmeting." Binnen elk hoofdstuk ontdek je gaandeweg de details.

Dat is precies hoe de Option-Critic-architectuur over beslissingen denkt.

---

## Wat is een "platte" agent? {#what-is-a-flat-agent}

Een normale RL-agent (zoals die uit Fase 3 en 4 van het curriculum) beslist één actie tegelijk, elke afzonderlijke stap. Het is als een GPS die de hele route opnieuw berekent telkens wanneer u één meter beweegt. Het werkt, maar het is vermoeiend en langzaam om te leren.

---

## Wat is een "optie"? {#what-is-an-option}

Een **optie** is een **benoemde vaardigheid**: een minibeleid dat de agent meerdere stappen achter elkaar kan uitvoeren voordat hij de controle teruggeeft.

Zie het als een manager die delegeert aan specialisten:

| Who | What they do |
|-----|-------------|
| **Manager (metabeleid)** | Bepaalt *welke* specialist op een klus wordt gestuurd |
| **Specialist A** | Expert in het navigeren door de kamer linksboven |
| **Specialist B** | Expert in het oversteken van deuropeningen |
| **Specialist C** | Expert in het rennen naar het doel |
| **Specialist D** | Reserve-generalist |

De manager kiest een specialist. De specialist werkt totdat hij besluit dat hij klaar is (dit heet **beëindiging**). Dan kiest de manager opnieuw.

---

## De drie bewegende delen {#the-three-moving-parts}

Elke optie bestaat uit drie componenten – beschouw ze als de **taakomschrijving** van de specialist:

1. **Inleiding**: Wanneer kan deze specialist ingeschakeld worden? *(Bijvoorbeeld: "Specialist A wordt alleen geactiveerd in de buurt van de kamer linksboven.")*
2. **Intra-optiebeleid**: Wat doet de specialist terwijl hij of zij aan het werk is? *(Bijvoorbeeld: "Loop naar de linkerbovenhoek.")*
3. **Beëindiging**: Wanneer draagt de specialist de controle terug? *(Bijvoorbeeld: "Stop wanneer je een deuropening hebt bereikt.")*

Het mooie van Option-Critic is dat ze alle drie **automatisch worden geleerd** — je hoeft de specialisten niet met de hand te maken. Het algoritme komt erachter dat het handig is om voor elke kamer één optie te hebben, of één om naar het doel te rennen, helemaal alleen.

---

## Een dag uit het leven van een optiecriticus {#a-day-in-the-life-of-an-option-critic-agent}

1. Agent komt een nieuwe kamer binnen (staat).
2. **Manager** kijkt naar de kamer en kiest een optie, bijvoorbeeld optie 2.
3. **De specialist van optie 2** neemt het over: loopt stap voor stap naar de deuropening.
4. Op een gegeven moment zegt Optie 2: "Ik ben hier klaar" (beëindiging).
5. **Manager** wordt wakker en kiest een nieuwe optie voor de nieuwe situatie.
6. Herhaal.

Vergelijk dit met de platte agent: de platte agent piekert over elke stap. Option-Critic delegeert hele gedragslijnen, waardoor elke specialist goed kan worden in zijn beperkte taak.

---

## Waarom helpt dit? {#why-does-this-help}

In een doolhof moet de agent een doel bereiken dat 30 tot 50 stappen verwijderd kan zijn. Met vlak leren is elke stap op het pad even ‘onzichtbaar’ totdat de beloning uiteindelijk aan het einde arriveert – dat signaal moet tientallen stappen achteruit gaan.

Bij opties wordt het pad opgesplitst in **subtaken**. Elke subtaak krijgt zijn eigen minibeloningssignaal (het bereiken van de deuropening, het betreden van de volgende kamer). Leren plant zich voort via kortere segmenten. **De agent leert sneller over problemen waarvoor veel stappen nodig zijn.**

Dit is het kernidee achter alle Hiërarchische RL – en Option-Critic is een van de schoonste implementaties ervan.

---

## Wat onze code doet {#what-our-code-does}

Het script `option_critic.py` plaatst een Option-Critic-agent in een **7x7 rasterwereld** met een vast doel. De agent begint ergens in het raster en moet naar de doelcel navigeren.

De agent heeft vier opties en moet tegelijkertijd leren:

- Een beleid voor elke optie (waar te lopen)
- Wanneer elke optie beëindigen (beëindigingsvoorwaarde)
- Een metabeleid voor het kiezen tussen opties

De beloning maakt gebruik van **potentieelgebaseerde vormgeving**: de agent krijgt een kleine bonus voor elke stap die hij dichter bij het doel komt, bovenop +1 voor het bereiken ervan. Deze uitgebreide feedback maakt het leren stabiel genoeg om de opties binnen 2500 afleveringen te zien werken.

Geen mens vertelt hem ooit wat elke optie zou moeten doen. Het algoritme ontdekt in welke gebieden van het raster elke optie gespecialiseerd is.

---

## Wat de grafieken laten zien {#what-the-charts-show}

![Optie-kritische leercurven](outputs/option_critic.png)

**Links – Shaped Return:** Een hoger rendement betekent dat de agent het doel betrouwbaarder bereikt *en* kortere paden bewandelt (de vormgeving geeft een bonus per stap dichterbij). De curve die stijgt en vervolgens stabiliseert, laat zien dat de opties leren coördineren.

**Rechts: stappen naar doel:** Minder stappen betekent dat de agent een efficiënter pad heeft gevonden. De neerwaartse trend laat zien dat de opties uitgroeien tot samenhangende vaardigheden die de agent directer naar het doel leiden.

De afgevlakte curven tonen de algemene trend over vensters van 100 afleveringen: enige ruis is normaal in RL, vooral wanneer meerdere componenten (opties, beëindiging, metabeleid) tegelijkertijd leren.

---

## Samenvatting van één zin {#one-sentence-summary}

> **Option-Critic leert een agent te werken in vaardigheden in plaats van in afzonderlijke stappen: een manager kiest welke specialist aan het werk is, elke specialist doet zijn werk en het hele systeem leert samen van hetzelfde beloningssignaal.**