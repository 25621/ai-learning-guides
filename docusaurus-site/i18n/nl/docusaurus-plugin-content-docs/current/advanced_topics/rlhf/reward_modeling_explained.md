# Beloningsmodellering: een computer leren wat mensen verkiezen

## Het grote idee

Een beloningsmodel is een kleine rechter. Je laat het twee antwoorden op hetzelfde zien
vraag, vertel hem welke iemand leuker vond, en na verloop van tijd leert hij het
om een hogere score te geven aan antwoorden die mensen liever zouden hebben.

Waarom hebben we zo’n rechter nodig? Omdat het meeste van wat we van een taal verwachten
model is moeilijk op te schrijven als een wiskundige formule. Er is niet één enkele vergelijking
voor 'behulpzaam', 'beleefd' of 'goed geschreven'. Maar mensen kunnen dat bijna altijd
wijzen op de beste van twee opties. Het beloningsmodel maakt dit eenvoudig
'Deze is beter' stemt in een score die een leeralgoritme kan gebruiken.

## Een analogie uit het echte leven

Stel je voor dat je een vriend leert brownies te bakken.

Je geeft ze geen regelboek van vijftig pagina's over 'wat een goede brownie is'.
In plaats daarvan proef je twee batches en zeg je:

"Deze is beter."

Na een paar rondes begint je vriend patronen op te merken. Misschien
wint de smeuïge batch altijd. Misschien verliest de overgebakken batch altijd. Je
vriend bouwt een mentaal scoresysteem op basis van jouw vergelijkingen.

Een beloningsmodel doet precies dit, maar dan met cijfers. Dat is niet nodig
weet *waarom* het gekozen antwoord beter is. Het heeft gewoon veel "deze beats" nodig
that' voorbeelden en geleidelijk leert het een partituur die overeenkomt met de
voorkeuren.

## Hoe het leren werkt (alleen intuïtie)

Elk voorbeeld is een drievoud: een prompt, een **gekozen** antwoord en a
**afgewezen** reactie. We willen dat het model een hogere score geeft aan de
uitverkorene dan de afgewezene – met welke marge dan ook.

De trainingsnudge is eenvoudig van geest:

- Score van gekozen te laag? Duw het omhoog.
- Score afgewezen te hoog? Duw het naar beneden.
- Al in de juiste volgorde met een duidelijke opening? Laat ze met rust.

Dat duwtje in de rug wordt het Bradley-Terry-verlies genoemd en is het standaardrecept
in moderne RLHF-systemen.

## Wat het experiment laat zien

We hebben een beloningsmodel getraind op 2.000 synthetische voorkeursparen. Het perceel
Hieronder ziet u drie weergaven van dezelfde trainingsrun.

![Beloningsmodeltraining](outputs/reward_modeling.png)

- **Links** - het verlies daalt snel. Het model wordt steeds zelfverzekerder
  over zijn ranglijst.
- **Midden** - de nauwkeurigheid van de voorkeuren stijgt tot bijna 100%. Op bijna elke
  paar krijgt het gekozen antwoord een hogere score dan het afgewezen antwoord.
- **Rechts** - de scoreverdelingen voor gekozen versus afgewezen antwoorden trekken
  uit elkaar. In het begin overlapten ze elkaar; na de training, gekozen reacties
  zit duidelijk rechts.

Die scheiding is het hele punt. Een latere stap (PPO of DPO) kunt u nu gebruiken
deze score als doel om naartoe te optimaliseren.

## Waar dit zich bevindt in de RLHF-pijpleiding

De routekaart beschrijft RLHF als ‘het afstemmen van modellen op menselijke voorkeuren’.
Het beloningsmodel is stap één van drie:

1. **Beloningsmodel (dit bestand)** - zet voorkeurstemmen om in een score.
2. **PPO-verfijning** - duw het taalmodel naar hogere scores
   terwijl hij dicht bij zijn oorspronkelijke gedrag blijft.
3. **DPO** - een nieuwere snelkoppeling die het beloningsmodel volledig overslaat.

Beloningsmodellering is dus de brug tussen *menselijk oordeel* en
*machine-optimalisatie*. Begrijp deze brug verkeerd en elke stroomafwaartse stap
raakt uit koers.

## Waarom dit belangrijk is buiten het laboratorium

Hetzelfde idee duikt op veel plaatsen op:

- **Aanbevelingssystemen** leren wat u leuk vindt door klikken, overslaan en
  tijd besteed aan kijken.
- **Zoekmachines** leren de rangschikking van 'op welk resultaat heeft u geklikt'.
- **Restaurants** leren de populaire gerechten van nabestellingen, niet van
  klanten die essays schrijven over wat ze leuk vonden.

Wanneer het gemakkelijker is om te *vergelijken* dan te *beoordelen*, is er sprake van een beloningsmodel
juiste gereedschap.

## Samenvatting van één zin

**Een beloningsmodel is een geleerde rechter die zegt: "Deze is beter"
voorkeuren omzetten in een numerieke score die de rest van RLHF kan optimaliseren.**