# Nieuwsgierigheidsbonus (intrinsieke motivatie) 🧭

## Wat is het? {#what-is-it}

Stel je voor dat een peuter in een nieuwe kamer valt. Niemand betaalt ze, niemand klapt –
toch rennen ze naar de kast die ze nog niet hebben geopend, de knop
ze hebben niet ingedrukt, het luidruchtige speeltje in de hoek. Ze draaien op een
**interne beloning**: *"Dat ziet er nieuw uit. Ga het eens bekijken."*

Een **nieuwsgierigheidsbonus** geeft een versterkende leeragent hetzelfde
interne schijf. De echte beloning van het milieu (de ‘extrinsieke’ beloning –
punten, geld, het winnen van het spel) blijft precies zoals het is. Wij voegen gewoon een
ten tweede, zelf gegenereerde beloning voor het bezoeken van dingen die de agent vindt *nieuw*
of *verrassend*, en train op de som:

```
reward the agent learns from  =  real reward  +  beta * curiosity bonus
```

`beta` is een knop die groot begint (wees nieuwsgierig!) en in de loop van de tijd kleiner wordt
(stop met treuzelen, ga verzilveren wat je hebt geleerd).

## Waarom moeite doen? Het schaarse-beloningsprobleem {#why-bother-the-sparse-reward-problem}

Normale RL-agenten leren van beloningen die ze daadwerkelijk ontvangen. Dat werkt
geweldig als er overal beloningen zijn ("+1 voor elke stap die je overeind blijft" in
CartPole). Het valt uiteen als de beloning **schaars** is: nul, nul,
nul, ..., nul, en dan eindelijk een +1 na een lange, zeer specifieke
opeenvolging van correcte acties.

Echte voorbeelden van schaarse beloningen:

- **Montezuma's Revenge** (het Atari-spel): je eerste punt komt pas binnen
  na ongeveer 100 precieze bewegingen: klim een ladder af, ontwijk een schedel, klim omhoog,
  pak een sleutel. Tot die tijd is de score een vlakke nul.
- **Een cijferslot.** 9.999 verkeerde codes leveren niets op; men geeft
  jij de prijs.
- **Ontdekking van medicijnen / wetenschappelijke experimenten.** Duizenden mislukte onderzoeken,
  dan eentje die werkt.
- **Een lang proefdruk of programma schrijven.** Geen gedeeltelijke credit tot het geheel
  ding uitcheckt.

Een agent die alleen maar beloningen biedt, is in deze omstandigheden als een student die weigert
studeren, tenzij ze betaald worden per juist antwoord op de finale – dat krijgen ze nooit
begonnen. Nieuwsgierigheid is de bonus die zegt: *"Verkennen is zijn eigen beloning,"*
dus de agent blijft rondneuzen totdat hij over de echte prijs struikelt.

## Twee smaken van nieuwsgierigheid (beide geïmplementeerd in `curiosity_bonus.py` ) {#two-flavours-of-curiosity-both-implemented-in-curiosity_bonuspy}

### 1. Op tellingen gebaseerde nieuwigheid: "Ik ben hier nauwelijks geweest" {#1-count-based-novelty-ive-barely-been-here}

Het eenvoudigst mogelijke nieuwigheidssignaal. Houd bij hoeveel `N(s, a)` er zijn
keer dat je actie hebt ondernomen `a` in staat `s` en geef jezelf een bonus die
krimpt naarmate dat aantal groeit:

```
curiosity bonus  =  1 / sqrt( N(s, a) + 1 )
```

Eerste keer dat je iets probeert: bonus = 1,0. Na 100 pogingen: bonus = 0,1.
Na 10.000 pogingen: 0,01. De agent wordt beloond als hij ergens naartoe gaat waar hij niet is geweest
geweest, en het kunstaas vervaagt op natuurlijke wijze van platgetreden grond.

**Echte analogie:** een toerist met een lijst met 'plaatsen die ik niet heb bezocht'.
Gloednieuwe buurt? Topprioriteit. Het café waar je vijftig bent geweest
keer? Niet spannend meer.

Dit is het oudste idee in het boek (MBIE-EB, UCB). Zijn zwakte: in a
grote of aaneengesloten wereld, je bezoekt *exact* dezelfde staat nooit twee keer, dus
het ruwe aantal is altijd 1 – daarom bestaat de volgende smaak.

### 2. Nieuwe voorspellingsfout: "Ik zag *dat* niet aankomen" {#2-prediction-error-novelty-i-didnt-see-that-coming}

Dit is het idee achter de beroemde **ICM** (Intrinsic Curiosity Module,
Pathak et al. 2017) en zijn neef **RND** (Random Network Distillation,
Burda et al. 2018). In plaats van te tellen, houdt de agent een beetje bij
**model dat probeert te voorspellen wat er daarna gebeurt** — "als ik hier ben en dat doe ik ook
dit, waar kom ik terecht?" – en beloont zichzelf door **hoe verkeerd het model is
was**:

```
curiosity bonus  =  surprise  =  -log P( the state I actually reached | where I was, what I did )
```

- Een situatie die het model nog nooit heeft gezien → het voorspelt slecht → grote verrassing
  → grote bonus → "ga daar op ontdekkingstocht!"
- Een situatie die het model al honderd keer heeft gezien → het voorspelt perfect →
  nul verrassing → nul bonus → "was daar, begrepen, ga verder."

**Echte analogie:** een kind dat spelenderwijs leert hoe de wereld werkt.
De *eerste* keer een glas van de tafel duwen is fascinerend (het
verbrijzeld!). De honderdste keer wist je al dat het zou versplinteren – niet
interessant. Nieuwsgierigheid = de kloof tussen wat je verwachtte en wat
gebeurde.

In onze tabelcode is het "model" slechts een tabel met overgangsaantallen, en
"hoe fout het was" is de verrassing `-log P` . Echte ICM/RND gebruikt neuraal
netwerken, dus hetzelfde idee werkt op onbewerkte pixels, maar het principe is hetzelfde
identiek.

> **Waarom twee versies?** Op tellen gebaseerd is doodeenvoudig en een geweldige basislijn.
> Voorspellingsfouten schalen zich naar grote, nooit herhalende werelden en geven a
> *scherper* signaal: in een deterministische omgeving, als je eenmaal een
> overgang zodra de verrassing onmiddellijk naar ~0 daalt, terwijl een telling
> bonus vervaagt slechts langzaam als `1/sqrt(N)` . In onze experimenten is de
> voorspellingsfoutagent lost MiniMontezuma op in een paar dozijn afleveringen;
> de telagent komt daar ook, alleen langzamer en minder betrouwbaar.

## Wat onze code doet {#what-our-code-does}

`curiosity_bonus.py` leidt een eenvoudige **tabelvormige Q-leerling** op
 `MiniMontezumaEnv` — een kleine rasterwereld met twee kamers waar je naar een
**sleutel**, pak hem op (nu gaat de **deur** open), loop er doorheen en bereik de
**schat**. Beloning (+1) verschijnt *alleen* bij de schat, na ~15
perfecte bewegingen. Het bestuurt drie agenten en plot ze:

| Agent | What it does on MiniMontezuma |
|-------|-----------------------------|
| **epsilon-hebzuchtig (geen nieuwsgierigheid)** | Dwaalt naar de start, * bereikt nooit * de sleutel, de score blijft voor altijd 0. |
| **op tellingen gebaseerde bonus** | Vindt betrouwbaar de sleutel; verbindt de hele keten met de schat, misschien ~ 40% van de afleveringen. Werkt - alleen een beetje luidruchtig. |
| **voorspellingsfoutbonus** | Bereikt eerst de sleutel *en* de schat binnen ~20-25 afleveringen; Naarmate `beta` vervalt, convergeert het naar het oplossen ervan in elke aflevering. |

De figuur laat zien:
- een leercurve: *P(bereik de schat)* tijdens training,
- een tweede curve voor de tussenliggende mijlpaal *P(sleutel ophalen)*,
- en **hittekaarten bij staatsbezoeken** van het elektriciteitsnet – de niet-nieuwsgierigheidsagent
  is een strakke klodder bij de start; de nieuwsgierige agenten zetten *beide* kamers onder water.

## Het mechanisme in één foto {#the-mechanism-in-one-picture}

```
            no curiosity                       with curiosity bonus
   reward:  0 0 0 0 0 0 0 0 ... 0  (+1?)        0 0 0 0 0 0 0 0 ... 0  (+1!)
            └──── nothing to learn from ──┘     └ + 0.4 0.3 0.9 0.2 ... ┘  (self-made,
                                                  dense, points "toward the new stuff")
   result:  random walk, never finds +1         systematically sweeps the world,
                                                 trips over +1, then the bonus fades
```

De nieuwsgierigheidsbonus verandert *"Ik heb dit niet gezien"* in een beloning, dus de
agent **dringt opzettelijk op onontgonnen terrein** in plaats van
willekeurig zwaaien. En omdat de bonus kleiner wordt naarmate de zaken zich ontwikkelen
bekend (en `beta` vervalt), zodra de agent de echte beloning ervoor heeft gevonden
stopt natuurlijk met treuzelen en begint met uitbuiten.

## Een paar eerlijke kanttekeningen {#a-few-honest-caveats}

- **Het "noisy-TV-probleem".** Een agent die een voorspellingsfout maakt, kan gehypnotiseerd raken
  door een bron van pure willekeur (een tv waarop ruis te zien is, terwijl er dobbelstenen worden gegooid) – het
  kan het *nooit* voorspellen, zodat de verrassing nooit vervaagt. De echte truc van ICM is
  om te voorspellen in een *geleerde functieruimte* die dingen van de agent negeert
  kan geen controle uitoefenen; RND omzeilt het op een andere manier. Onze deterministische
  gridworld heeft geen luidruchtige tv, dus hier komen we niet tegen.
- **Nieuwsgierigheid is een middel, geen doel.** Daarom vervalt `beta`. Een agent
  dat voor altijd maximaal nieuwsgierig blijft en nooit echt wordt
  *winnen*.
- **Het opschalen van diepgaande verkenning is nog steeds moeilijk.** Een bonus in de beloning helpt,
  maar Q-learning in eenvoudige tabelvorm verspreidt het resulterende optimisme maar langzaam
  langs een lange keten (zie `compare_exploration.py` ). Krakende pixel
  Montezuma had extra vuurkracht nodig – RND met een neuraal net, bootstrapped
  DQN, Go-Explore.

## Sleutelwoorden om te onthouden {#key-words-to-remember}

| Word | Meaning |
|------|---------|
| **Intrinsieke beloning** | Een beloning die de agent voor zichzelf genereert, los van de beloning van de omgeving |
| **Extrinsieke beloning** | De echte beloning van de omgeving (punten, winnen/verliezen) |
| **Schaarse beloning** | De beloning is bijna overal nul; je krijgt het pas na een lange correcte reeks |
| **Nieuwigheid / verrassing** | Hoe nieuw of onverwacht een toestand (of transitie) is – dat is wat nieuwsgierigheid beloont |
| **Op tellingen gebaseerde bonus** | Nieuwigheid ≈ `1/sqrt(visit count)` — de klassieke verkenningsbonus |
| **ICM** | Intrinsieke nieuwsgierigheidsmodule: nieuwigheid = de voorspellingsfout van een voorwaarts model (in een geleerde kenmerkruimte) |
| ** `beta` ** | Het gewicht op de nieuwsgierigheidsbonus; meestal uitgegloeid richting 0, zodat de agent uiteindelijk misbruik maakt |

## Samenvatting van één zin {#one-sentence-summary}

> **Een nieuwsgierigheidsbonus is een zelf gegeven beloning voor nieuwigheid – het produceert
> een compact signaal van 'ga daarheen verkennen' dat de agent erdoorheen sleept
> werelden met schaarse beloningen die het anders nooit zou oplossen en vervolgens beleefd vervaagt
> weg zodra alles bekend is.**