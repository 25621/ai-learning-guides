# PPO-fijnafstelling: een model polijsten zonder het te breken

## Het grote idee

Zodra we een beloningsmodel hebben dat reacties scoort, willen we onze taal
model om antwoorden met hogere scores te produceren. PPO (Proximaal beleid
Optimalisatie) doet precies dit, maar voegt een veiligheidsgordel toe aan het model
achtervolgt de partituur niet en vergeet hoe je normale tekst moet schrijven.

Zie het als een poetsstap. Het model spreekt al vloeiend; wij gewoon
spoor het aan om te spreken op een manier die het beloningsmodel beloont, terwijl het dat blijft
stem herkenbaar.

## Een analogie uit het echte leven

Stel je een chef-kok voor die al goed kookt, maar nu leert om iemand tevreden te stellen
specifieke voedselcriticus.

Na elk gerecht geeft de criticus een score. De chef-kok heeft twee drukpunten:

1. **Behaal een hogere score.** Kook op een manier die de criticus leuk vindt.
2. **Niet onherkenbaar worden.** Als de chef de eigen stijl loslaat
   volledig - zout bij de beker gooien alleen maar om een score na te jagen - de
   eten wordt raar. Klanten komen niet meer.

PPO vangt beide vormen van druk op:

- Het deel **beloning** stimuleert het model in de richting van reacties die de rechter leuk vindt.
- Het **KL-straf**-gedeelte trekt het model terug naar hoe het voorheen sprak
  opleiding begonnen. KL is slechts een manier om te meten "hoe anders is de
  nieuw gedrag van het oude gedrag."

Samen zeggen ze: *word beter, maar blijf jezelf*.

## Hoe het leren werkt (alleen intuïtie)

Elke trainingsronde ziet er als volgt uit:

1. Volg enkele aanwijzingen. Laat het huidige model antwoorden produceren.
2. Scoor de antwoorden met het beloningsmodel.
3. Vergelijk met het **referentiemodel** - een bevroren kopie van het model
   van vóór de training. Als de nieuwe reacties heel anders zijn,
   trek een KL-straf af van de beloning.
4. Geef het model een duwtje in de richting van antwoorden die goed scoorden.

Het "Proximaal" in PPO betekent *neem geen grote sprongen*. Elke update is een
kleine, voorzichtige stap. Grote sprongen in beleidstraining veroorzaken crashes
waarom eerdere methoden zoals vanilla policy gradiënt zo onstabiel waren.

## Wat het experiment laat zien

We beginnen met een fris beleid en een getraind beloningsmodel. PPO loopt voor 150
iteraties, het nemen van monsters van batches reacties en het bijwerken van het beleid.

![PPO-training](outputs/ppo_fine_tuning.png)

- **Links** - de gemiddelde score van het beloningsmodel stijgt gestaag. Het beleid is
  leren reacties te produceren die de rechter leuk vindt.
- **Midden** - De KL-divergentie van het referentiemodel neemt toe. Het beleid is
  weggaan van waar het begon. Dit wordt verwacht, maar als het groeit
  Als er niets aan gedaan wordt, zou het model in onzin vervallen.
- **Rechts** - de gevormde beloning (onbewerkte beloning minus de KL-straf) volgt
  de rauwe beloning komt eerst dichtbij, maar raakt dan achterop als KL klimt. De
  De boete doet zijn werk: het model laten ‘betalen’ voor het te ver afdrijven.

In een echt RLHF-systeem stem je de KL-coëfficiënt af tot de score stilstaat
stijgt, maar het model blijft coherent. Een te kleine boete en het model
hackt de beloning door vreemde herhalende zinnen uit te zenden. Te groot en de
model verbetert nooit.

## Waar dit zich bevindt in de RLHF-pijpleiding

Dit is stap twee van het klassieke RLHF-recept:

1. Train een beloningsmodel vanuit voorkeuren.
2. **Verfijn het taalmodel met PPO met behulp van dat beloningsmodel.**
3. (Optioneel) Sla stap 2 met DPO over als u een eenvoudiger pad wilt.

PPO is het werkpaard dat bedrijven als OpenAI en Anthropic hiervoor gebruikten
eerste golf van uitgelijnde modellen, inclusief InstructGPT en het origineel
ChatGPT.

## Waarom dit belangrijk is buiten het laboratorium

Het patroon van ‘verbeteren, maar niet afdrijven’ komt overal voor:

- Een pianist die een hard stuk oefent, verandert niet zijn hele techniek
  om één passage te pakken te krijgen – dat zou de rest van het recital verbreken.
- Een bedrijf dat een website aanpast om meer aanmeldingen te genereren, moet nog steeds het merk behouden
  herkenbaar voor bestaande gebruikers.
- Een fabriek die één knop in een proces aanpast, houdt de andere dicht bij de knop
  bekend-goede instellingen.

PPO is slechts een zorgvuldige versie van dit universele idee, geschreven in wiskunde.

## Samenvatting van één zin

**PPO-verfijning duwt een model naar een hogere beloning terwijl er een KL-straf wordt opgelegd
houdt het dicht bij zijn oorspronkelijke gedrag - verbeter, maar blijf jezelf.**