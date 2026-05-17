# Verkenningsstrategieën vergelijken 🔦

## Het probleem van één zin {#the-one-sentence-problem}

Een RL-agent moet twee dingen doen die in tegengestelde richtingen werken:

- **Exploit**: doe datgene wat tot nu toe het beste heeft gewerkt.
- **Verkennen**: probeer iets nieuws, voor het geval het nog beter is.

Leun te ver in de richting van exploiteren en je zult genoegen nemen met middelmatigheid
routine voor altijd. Leun te ver in de richting van ontdekken en je verdient er nooit geld mee. *Hoe*
je onderzoekt – en niet alleen *of* – is wat een agent scheidt die oplost
Montezuma's Revenge van een die nul scoort.

Dit script zet **vijf** verkenningsstrategieën tegenover elkaar
twee zware taken, zodat je hun persoonlijkheid kunt zien.

## Analogie uit het echte leven: een lunchplek kiezen {#real-life-analogy-picking-a-lunch-spot}

Je bent net verhuisd naar een nieuwe stad met 200 restaurants.

- **ε-greedy** = "Ga naar mijn huidige favoriet, maar eens in de tien dagen
  een dobbelsteen en kies een *totaal willekeurig* restaurant." Je zult op grote schaal samplen, maar
  *doelloos* — en je blijft plaatsen herrollen waar je al een hekel aan had.
- **Optimistische initialisatie** = "Ga ervan uit dat *elk* restaurant dat niet is
  geprobeerd is de beste in de stad, totdat het tegendeel bewezen is." Je zult methodisch werken
  doorloop alle 200 en streep ze allemaal af omdat de realiteit je teleurstelt –
  en je zult snel de echt geweldige vinden.
- **UCB (Upper Confidence Bound)** = "Geef de voorkeur aan mijn favoriet, maar geef een *bonus* aan plaatsen waar ik nauwelijks ben geweest
  geprobeerd – hoe minder ik ervan weet, hoe groter de bonus. Dit is slim
  over *welke* onbekende plek om vandaag te proberen, maar elke beslissing is lokaal:
  het kiest *nu* de best uitziende optie zonder een route te plannen
  door hele onontdekte buurten. Het zal niet denken: "Ik moet oversteken
  stad aan de oostkant, omdat er twintig onbeproefde plekken zijn geclusterd
  daar" — elk restaurant wordt stap voor stap afzonderlijk geëvalueerd.
- **Op tellingen gebaseerde beloningsbonus** = zoals UCB, maar u *geniet ook van de nieuwigheid
  zelf* — een maaltijd op een gloednieuwe plek is intrinsiek bevredigend, en
  die tevredenheid bepaalt uw langetermijnplan voor welke buurten
  binnendwalen.
- **Voorspellingsfout beloningsbonus** = "Ik krijg een kick van een maaltijd die
  *verrast* mij – iets wat ik niet had kunnen voorspellen." Een nieuwe plek dat
  verloopt precies zoals verwacht? Meh. Eén die heel anders is dan
  jouw mentale model? Fascinerend, en je werkt je plan bij om meer te zoeken
  vind het leuk.

## De vijf strategieën (allemaal in `compare_exploration.py` ) {#the-five-strategies-all-in-compare_explorationpy}

### 1. ε-greedy — de standaard, en het is *dithering*, niet verkennen {#1-ε-greedy--the-default-and-its-dithering-not-exploring}

Handel hebzuchtig, maar onderneem met waarschijnlijkheid ε een gelijkmatig willekeurige actie. Het is
de standaardbasislijn in DQN en vrienden. De fatale fout bij moeilijke taken:
**elke stap is een onafhankelijke muntworp.** Om langs een keten van `N` te struikelen 
Met de juiste zetten heb je de munt nodig om `N` keer op rij goed te landen – dat is het
exponentieel onwaarschijnlijk. ε-hebzuchtig is *wiebelen*, niet *verkennen*.

### 2. Optimistische initialisatie — "onschuldig totdat bewezen saai is" {#2-optimistic-initialisation--innocent-until-proven-boring}

Begin *elke* Q-waarde met het grootste rendement dat zelfs maar mogelijk is,
 `R_max / (1 − γ)` . Nu lijkt een actie die de agent nog nooit heeft geprobeerd op de
het beste ter wereld, dus het **hebzuchtige** beleid wordt gedwongen het te gaan proberen;
pas na een bezoek zakt de waarde richting de waarheid. Het optimisme
over *niet*geprobeerde regio's **wordt automatisch door de waarde doorgegeven
functie** (via de bootstrap van Q-learning), zodat de agent stap voor stap wordt opgehaald
stap, richting de delen van de wereld die het nog niet heeft gezien. Bijna gratis, geen extra's
boekhouden – en, zoals je zult zien, de sterkste *diepe* ontdekkingsreiziger in een klein
tabellarische wereld.

### 3. Actieselectie in UCB-stijl – bonus in de *keuze*, niet in de *beloning* {#3-ucb-style-action-selection--bonus-in-the-choice-not-the-reward}

Kies `argmax_a [ Q(s,a) + c·√(ln t / N(s,a)) ]`: geef de voorkeur aan hoge waarde
acties, maar blaas de acties op die je zelden hebt geprobeerd. Beroemd van meerarmig
bandieten. Het addertje onder het gras: de bonus leeft alleen in de **actieselectieregel**,
nooit in de beloning - dus het stroomt *niet* door de waardefunctie.
UCB is geweldig in "zorg ervoor dat ik elke actie in *deze* staat heb geprobeerd", maar
zwak in 'een route plannen naar een verre, onontdekte regio'.

### 4. Op tellingen gebaseerde **beloningsbonus** – nieuwsgierigheid, de klassieke versie {#4-count-based-reward-bonus--curiosity-the-classic-version}

Voeg `1/√(N(s,a))` toe aan de **beloning** (met een gewicht `beta` dat vervalt).
Omdat het in de beloning zit, propageert Q-learning het *: stelt dat
leiden naar nieuwe regio’s waardevol worden. Dit is de MBIE-EB / klassieker
"verkenningsbonus"-idee — en de eerste helft van werkitem 1.

### 5. Voorspellingsfout **beloning** bonus — nieuwsgierigheid, de ICM/RND-versie {#5-prediction-error-reward-bonus--curiosity-the-icmrnd-version}

Voeg `−log P(s'|s,a)` van een klein geleerd voorwaarts model toe aan de beloning
(opnieuw met verval van `beta` ). Het scherpste nieuwigheidssignaal van de vijf: in
In een deterministische wereld zakt de verrassing van een transitie op dit moment naar ~0
je hebt het één keer gezien, in plaats van langzaam te vervagen zoals `1/√N` . Het tabellarisch
neef van ICM / RND — de tweede helft van werkitem 1.

## De twee testtaken {#the-two-test-tasks}

- **Taak A — MiniMontezuma**: de sleutel → deur → schatrasterwereld, alleen beloning
  bij de schat (~15 perfecte zetten verwijderd). Tests "kun je lang overleven
  überhaupt een schaarse beloningsketen?"
- **Taak B — DeepSea(N)**: de diepgaande verkenningsketen uit het boekje, uitgevoerd op
  lengtes `N = 5, 8, 11, 14` . De beloning gaat schuil achter `N` correcte zetten,
  elk met een kleine directe prijs – zodat een bijziende agent leert de gevolgen ervan te vermijden
  kost en vindt nooit de prijs. Tests "werkt jouw strategie nog steeds als
  wordt de ketting *langer*?"

## Wat er feitelijk gebeurt (voer het uit en zie) {#what-actually-happens-run-it-and-see}

**Taak A — MiniMontezuma:**

| Strategy | First treasure | Final solve rate |
|----------|---------------:|-----------------:|
| ε-hebzuchtig | nooit | 0,00 |
| optimistisch begin | ~ aflevering 1 | 1.00 |
| UCB-actieselectie | ~aflevering 3 | ~0,95 |
| tel beloningsbonus | ~ aflevering 82 | ~0,41 |
| voorspelling beloningsbonus | ~ aflevering 23 | 1.00 |

**Taak B — DeepSea, fractie van de zaden die de beloning hebben gevonden:**

| Strategie | N=5 | N=8 | N=11 | N=14 |
|----------|----:|----:|-----:|-----:|
| ε-hebzuchtig | 0 | 0 | 0 | 0 |
| optimistisch begin | 1,0 | 1,0 | 1,0 | 1,0 |
| UCB-actieselectie | 1,0 | 1,0 | 0,0 | 0,0 |
| beloningsbonus tellen | 1,0 | 1,0 | ~0,1 | 0,0 |
| voorspelling beloningsbonus | ~0,9 | ~0,8 | ~0,9 | ~0,2 |

*(De cijfers wiebelen een beetje door de willekeurige zaden, maar de vorm is rotsachtig
stevig.)*

## De lessen {#the-lessons}

1. **ε-hebzuchtig is geen verkenning.** Het lost nooit *beide* moeilijke taken op.
   Willekeurig ditheren levert eenvoudigweg geen lange, correcte reeksen op. (Toch
   het is nog steeds de standaard in veel code, omdat dat bij *gemakkelijke* taken zo is
   goed genoeg en doodeenvoudig.)

2. **Echte verkenning betekent optimistisch zijn over het onbekende – in één richting
   of iets anders.** Of je nu het optimisme in de *beginwaarden* verankert
   (strategie 2), in de *actiekeuze* (strategie 3), of in a
   *zelf gegenereerde beloning* (strategieën 4–5), de rode draad is: *make
   het onontdekte ziet er aantrekkelijk uit*, laat het leren je daarheen brengen.

3. **Op een schaars beloningsrooster werken alle vier de ‘echte’ strategieën – en de
   voorspellingsfoutbonus komt er het snelst**, omdat het de
   helderste "dit is nieuw" signaal.

4. **In een *diepe* keten, waar het optimisme een lange weg moet afleggen, zal de
   De eenvoudige kampioen is optimistische initialisatie.** Het propageert optimisme
   via de waardefunctie gratis. UCB valt als eerste uiteen (de bonus
   komt nooit in de waardefunctie terecht, dus het kan niet diep plannen). Beloning
   bonussen doen het beter – ze verspreiden zich *wel* – maar eenvoudige Q-learning in tabelvorm
   is traag om dat optimisme helemaal door een lange keten te duwen vóór de
   bonus vervalt.

5. **Dat laatste punt is precies de reden waarom diepgaande verkenning naar pixels wordt geschaald
   had extra vuurkracht nodig** - DQN, RND opgestart met een echt neuraal net
   (dus optimisme *generaliseert* over soortgelijke staten in plaats van
   één cel tegelijk voortplanten), Go-Explore (onthoud letterlijk en
   terugkeer naar veelbelovende staten). Het speelgoed in tabelvorm hier laat je de
   *principes*; de echte systemen zijn dezelfde principes plus een netwerk
   dat generaliseert.

## Sleutelwoorden om te onthouden {#key-words-to-remember}

| Word | Meaning |
|------|---------|
| **Afweging tussen exploratie en exploitatie** | Probeer nieuwe dingen versus contant geld met wat je weet – de centrale spanning in RL |
| **Dithering** | "Verkenning" door willekeurige ruis aan acties toe te voegen (ε-hebzuchtige, Gaussiaanse ruis) - zwak bij moeilijke taken |
| **Optimisme ondanks onzekerheid** | Het overkoepelende principe: behandel het onbekende alsof het geweldig is totdat je het hebt gecontroleerd |
| **Optimistische initialisatie** | Implementeer dat principe door alle waarden te starten met het maximaal mogelijke rendement |
| **UCB** | Bovenste vertrouwensgrens: kies `argmax (value + bonus that shrinks with visit count)` |
| **Diepe verkenning** | Een verkenning die een lange, samenhangende reeks ‘ongebruikelijke’ acties vereist, en niet slechts één |
| ** `beta` uitgloeien** | Het nieuwsgierigheidsgewicht in de loop van de tijd verkleinen, zodat de agent uiteindelijk stopt met verkennen en exploiteren |

## Samenvatting van één zin {#one-sentence-summary}

> **ε-hebzuchtig is alleen maar lawaai; elke echte verkenningsstrategie werkt door te maken
> het onontdekte uiterlijk aantrekkelijk – via optimistische waarden, een actiekeuze
> bonus, of een zelf gegenereerde nieuwigheidsbeloning – en de juiste keuze hangt ervan af
> of uw beloning slechts *schaars* is (zoals het vinden van een enkele verborgen prijs
> in een vlak veld) of echt *diep* (zoals een cijferslot waarvoor een
> lange, precieze reeks specifieke keuzes om te kraken).**