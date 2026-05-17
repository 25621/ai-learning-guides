# Zelf spelen: een agent lesgeven door hem zelf te laten spelen ♟️

## Wat is zelfspelen?

Stel je een kind voor dat heel goed wil worden in schaken, maar niemand heeft om te spelen.
Ze speelt dus zichzelf. Linkerhand versus rechterhand. Bij elk spel proberen *beide* partijen het
winnen. Bij elk spel leren *beide* partijen wat werkte.

Dat is **zelfspel**: één agent fungeert als beide spelers en bij elke zet
wordt een les voor degene die als volgende stap zet. Geen leraar, geen deskundige tegenstander.
Gewoon een leerling die ook zijn eigen ladder is.

Zelfspelen klinkt als een truc – je hebt toch zeker een echte tegenstander nodig? – maar dat is zo
de motor achter de beroemdste RL-mijlpalen van het afgelopen decennium:
**AlphaGo Zero**, **AlphaZero**, **MuZero**, **OpenAI Five**. Ze gebruiken allemaal
zelf spelen. De reden is simpel: naarmate jij verbetert, verbetert je tegenstander ook
hetzelfde bedrag. De uitdaging komt altijd overeen met jouw vaardigheden.

---

## Waarom het werkt

Drie dingen maken zelfspelen speciaal:

1. **Eindeloze tegenstanders.** Je komt nooit zonder games te zitten. De tegenstander is altijd
   aanwezig en gratis.
2. **Curriculum dat met je meegroeit.** Een beginner kan alleen andere spelen
   beginners. Naarmate jij beter wordt, wordt jouw schaduw dat ook – automatisch.
3. **Symmetrie.** In een nulsomspel (de winst van de ene speler is het verlies van de andere),
   één set Q-waarden beschrijft beide kanten; je draait gewoon het bord om als het zover is
   is de andere speler aan de beurt. Een *enkele* Q-table kan zichzelf dus leren.

Boter-kaas-en-eieren is het perfecte testbed: klein genoeg om in een woordenboek te passen, maar
complex genoeg dat het willekeurig kiezen van zetten bijna altijd zal leiden tot verlies tegen een strategische speler.

---

## Een analogie uit het echte leven

- **Tennis oefenen tegen een muur.** Je kunt niet verliezen tegen een muur, maar jij wel
  kunt uw services oefenen. Zelfspelen doet dit aan beide kanten: dat doe jij
  de muur *en* de speler, en je schakelt heen en weer.
- **Een debatclub die beide kanten bepleit.** Er komen betere debaters tevoorschijn
  altijd het tegenovergestelde standpunt verdedigen van wat zij persoonlijk geloven.
  Elk argument traint zowel aanval als verdediging.
- **AlphaGo Zero.** Het heeft geleerd van nul menselijke spelletjes. Beginnen met willekeurig
  bewegingen speelde het miljoenen spellen tegen zichzelf; binnen een paar dagen was het zo
  beter dan elk vorig Go-programma, inclusief het programma dat Lee versloeg
  Sedol.

---

## Wat onze code doet

We leren één Q-tafel voor de *huidige speler om te bewegen*:

```
Q[(board, player_to_move)][action] = expected return for that player
```

De trainingslus is:

1. Begin een leeg bord.  `player = X` .
2. Beide spelers handelen met **dezelfde agent**, waarbij ze ε-greedy gebruiken.
3. Loop na elk spel achteruit door elke (bord, speler, actie)
   verdriedubbel in de geschiedenis en pas de Q-learning update toe.
4. Het beloningsbord draait om beurten: als X wint, krijgt elke zet die X heeft gedaan
   +1 (of bootstrapwaarde van een toekomstige winnende staat); elke zet die O maakte krijgt -1.
5. We verlagen (verval) langzaam onze verkenningssnelheid (ε) van 0,2 → 0,02, zodat de agent zich inzet voor zijn beste spel
   laat in de training in plaats van willekeurige bewegingen uit te proberen.

Elke 2.500 afleveringen evalueren we de agent tegen een **willekeurige tegenstander**
(we bevriezen het leerproces, zodat er tijdens de evaluatie geen nieuwe updates aan de Q-tafel worden aangebracht en beide partijen gretig spelen). De agent moet winnen of gelijkspelen
~100% van die games na voldoende zelfspel.

### Wat je zou moeten zien

Na 50.000 zelfgespeelde afleveringen:

| Match-up | Verwacht resultaat |
|----------|-----------------|
| Getrainde agent versus willekeurige tegenstander (1000 spellen) | **~95-99% winst of gelijkspel**, vrijwel 0% verlies |
| Getrainde agent versus zichzelf (200 hebzuchtige spellen) | **Alle 200 trekkingen**. Boter-kaas-en-eieren is een spel dat altijd eindigt in een gelijkspel (gelijkspel) als beide spelers perfect spelen. Het feit dat zelfspel elke game trekt, is een teken van convergentie. |

De plot `outputs/self_play_tic_tac_toe.png` toont de winst/gelijkspel/verlies van de agent
fracties versus een willekeurige tegenstander in de loop van de tijd:
- Het winstpercentage begint met ~60% (wanneer beide spelers willekeurig spelen, heeft de eerste speler een inherent voordeel omdat hij meer markeringen op het bord mag plaatsen, wat leidt tot een basiswinstpercentage van ongeveer 60% voor speler X).
- Klimt tot >90%.
- Het verliespercentage daalt tot bijna 0%.

Het script drukt aan het einde ook een voorbeeldspel, zet voor zet, af, zodat u het kunt zien
de agent speelt.

---

## Pas op voor deze subtiliteiten

- **Sign flips zijn belangrijk.** Een veel voorkomende bug: vergeten dat "de tegenstander
  het maximaliseren van hun waarde" betekent *het minimaliseren van de onze* in het bootstrap-doel.
  De update in onze code gebruikt `target = reward - gamma * max(Q[next, opponent])` .
- **Symmetrie wordt hier niet uitgebuit.** Een echte implementatie zou heilig verklaren
  borden (wat betekent dat ze elke bordstatus zouden roteren of weerspiegelen in een standaard, unieke 'normale vorm' zodat de agent identieke bordsituaties herkent) om Q-waarden te delen over 8
  symmetrieën. We slaan dit over: de toestandsruimte is klein genoeg voor brute kracht.
- **De Q-tafel groeit.** Na 50.000 zelfspeelspellen zul je er een paar zien
  duizend staatsspelersleutels. Dat is hier prima; voor schaken of Go zou je dat doen
  hebben in plaats daarvan een neuraal netwerk nodig. Daarom vervangt **AlphaZero de
  tabel met een CNN + MCTS**.

---

## Waar zelfspel breekt

- **Niet-nul-som-spellen.** "Beide partijen blij" is incompatibel met
  symmetrisch spel; je kunt niet zomaar een bord omdraaien.
- **Asymmetrische rollen.** Als "aanvaller" en "verdediger" verschillende actie hebben
  ruimtes, heb je twee afzonderlijke netwerken nodig.
- **Strategie fietsen.** Puur zelfspel kan blijven hangen
  steen-papier-schaar-achtige cycli. AlphaStar heeft dit opgelost door een grote aan te houden
  *pool* (of "competitie") van opgeslagen eerdere versies van de agent en selectie
  tegenstanders uit die pool willekeurig, zodat de agent leert er veel te verslaan
  verschillende speelstijlen in plaats van alleen de huidige.
- **Beloning hacken.** Door zelf te spelen worden beide partijen slimmer, maar alleen aan de
  spel *zoals je het hebt gedefinieerd*. Als je beloningssysteem onbedoelde mazen in de wet bevat (zoals het belonen van een speler alleen voor het langer overleven in plaats van winnen), zullen beide partijen de maas in de wet wederzijds exploiteren, wat leidt tot bizar, nutteloos gedrag in plaats van het beheersen van het eigenlijke spel.

---

## Sleutelwoorden om te onthouden

| Word | Meaning |
|------|---------|
| **Zelfspelen**      | Dezelfde agent speelt beide kanten van een spel |
| **Nulsom**       | De winst van de ene speler = het verlies van de andere |
| **Symmetrie**       | Eén Q-table kan beide kanten bedienen als je de bordjes omdraait |
| **Bevolkingsspel**| Speel zelf met *veel* eerdere versies van jezelf als tegenstanders (AlphaStar) |
| **Leerplan**     | Een natuurlijke moeilijkheidsgraad: zelf spelen krijgt het gratis |
| **MCTS**           | Monte-Carlo Tree Search - het planningsalgoritme AlphaZero combineert met zelfspel |

---

## Samenvatting van één zin

> **Zelfspel verandert verbetering in zijn eigen ladder: elke keer dat je het krijgt
> beter, je tegenstander doet dat ook — automatisch.**

Dit idee, opgeschaald met **neurale netwerken** (op de hersenen geïnspireerde wiskunde
functies die patronen uit gegevens leren) en het zoeken naar bomen, verslaan de beste mensen
bij Go, schaken, shogi, Dota 2 en StarCraft.