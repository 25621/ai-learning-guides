# DPO: de rechter overslaan en rechtstreeks naar de bron gaan

## Het grote idee

Klassieke RLHF kent twee fasen: train eerst een beloningsmodel en gebruik vervolgens PPO
achtervolg zijn scores. DPO (Direct Preference Optimization) vraagt een slimme
vraag:

*Als het beloningsmodel slechts een tussenstap is, kunnen we deze dan overslaan?*

Het blijkt: ja. DPO traint het taalmodel rechtstreeks vanuit voorkeur
paren, zonder afzonderlijke rechter, zonder PPO-bemonsteringslus en zonder KL-coëfficiënt
afstemmen. Het gebruikt één elegante formule en gedraagt ​​zich als begeleid leren.

Dit maakt DPO eenvoudiger uit te voeren, stabieler en sneller - en dat is ook de reden
is snel de standaardkeuze geworden voor veel op open source afgestemde modellen.

## Een analogie uit het echte leven

Stel dat je een student coacht bij het schrijven van essays.

De PPO-aanpak is: huur een leraar in om essays te beoordelen, en laat de student dat doen
schrijf essay na essay en pas het aan op basis van de cijfers van de leraar.

De DPO-aanpak is: laat de student twee essays tegelijk zien en zeg:
"Deze is beter - neig naar schrijven zoals deze, weg van dat
één." Geen leraar in het midden. De leerling past direct aan
vergelijkingen.

Beide kunnen werken. DPO is meestal sneller klaar omdat niemand hoeft te trainen en
een aparte leraar behouden.

## Hoe het leren werkt (alleen intuïtie)

DPO gebruikt dezelfde voorkeursparen als beloningsmodellering: prompt, gekozen,
afgewezen. Voor elk paar worden twee vragen gesteld:

1. Is de kans groter geworden dat het model het gekozen antwoord oplevert?
   dan het referentiemodel zou zijn geweest?
2. Is de kans kleiner geworden dat het model het afgewezen antwoord oplevert?
   dan het referentiemodel zou zijn geweest?

De training duwt beide getallen tegelijk in de goede richting. Cruciaal is dat
het referentiemodel is altijd aanwezig in de vergelijking - het speelt hetzelfde
rol als de KL-straf in PPO. Het model mag verschuiven, maar de
verschuivingen zijn altijd *ten opzichte van* het startpunt.

Een subtiel en mooi resultaat uit het DPO-papier is dat dit enkel verlies is
functie is wiskundig gelijkwaardig aan 'een beloningsmodel trainen en vervolgens uitvoeren'
PPO met een KL-straf." Dezelfde bestemming, eenvoudiger reis.

## Wat het experiment laat zien

We hebben een beleid rechtstreeks getraind op 2.000 voorkeursparen voor 300 tijdvakken.

![DPO-training](outputs/dpo_implementation.png)

- **Links** - het DPO-verlies neemt af naarmate het model leert de voorkeur te geven aan 'verkozen'
  afgewezen reacties.
- **Midden** - nauwkeurigheid van voorkeuren (hoe vaak het beleid een hogere waarde toekent).
  impliciete beloning voor het gekozen antwoord) stijgt tot ongeveer 99%.
- **Juist** - de impliciete beloningsmarge groeit. DPO noemt nooit een ‘beloning’
  maar de kloof tussen log-kansen van gekozen versus afgewezen, geschaald met
  bèta, kan als één geheel worden gelezen. Het wordt gestaag breder, wat betekent dat het model wordt
  meer vertrouwen in zijn voorkeuren.

Merk op hoe schoon dit eruit ziet vergeleken met PPO. Er is geen bemonsteringslus, nee
verkenningsgeluid en er loopt geen apart beloningsmodel. Elk tijdperk is een
pure update onder toezicht van de voorkeursdataset.

## Waar dit zich bevindt in de RLHF-pijpleiding

DPO is een *alternatief* voor stap twee van de klassieke pijplijn:

- **Klassiek:** voorkeuren → beloningsmodel → PPO → afgestemd model.
- **DPO:** voorkeuren → afgestemd model. (Klaar.)

Het addertje onder het gras is dat DPO traint op een dataset met vaste voorkeuren. PPO, omdat
het bemonstert elke ronde nieuwe antwoorden en kan in principe verder verkennen.
In de praktijk wint DPO het meeste gebruik van "afstemmen op een samengestelde voorkeursdataset".
gevallen.

## Waarom dit belangrijk is buiten het laboratorium

Het patroon "sla de middelste meting over" is overal:

- Een coach die de vorm van een zwemmer corrigeert door zij aan zij te demonstreren
  dan elke ronde te scoren met een stopwatch.
- Een fotograaf die twee foto's tegelijk bewerkt en de beste kiest,
  in plaats van een 'goede foto'-scorerubriek op te stellen.
- Een rekruteringsmanager die twee cv's vergelijkt in plaats van ze allemaal te beoordelen
  tegen een checklist van 30 punten.

Als u alleen *rang* hoeft te geven, heeft u geen absolute schaal nodig. DPO is
dat inzicht werd toegepast op taalmodellen.

## Samenvatting van één zin

**DPO verandert voorkeursparen rechtstreeks in een beter model, zonder beloning
model in het midden - eenvoudiger dan PPO, en vaak net zo goed.**