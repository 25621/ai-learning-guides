# Tutkimusstrategioiden vertailu 🔦

## Yhden lauseen ongelma {#the-one-sentence-problem}

RL-agentin on tehtävä kaksi asiaa, jotka vetäytyvät vastakkaisiin suuntiin:

- **Hyödynnä**: tee se, mikä on toiminut parhaiten tähän mennessä.
- **Tutki**: kokeile jotain uutta, jos se on vielä parempaa.

Nojaa liian pitkälle hyväksikäyttöön, niin tyytyät onnellisesti keskinkertaiseen
rutiinia ikuisesti. Nojaa liian pitkälle tutkiaksesi, etkä koskaan ansaitse rahaa. *Kuinka*
tutkit - ei vain *onko* - on se, mikä erottaa agentin, joka ratkaisee
Montezuman kosto sellaisesta, joka saa nollan.

Tämä skripti yhdistää **viisi** tutkimusstrategiaa
kaksi kovaa tehtävää, jotta näet heidän persoonallisuutensa.

## Analogia tosielämästä: lounaspaikan valinta {#real-life-analogy-picking-a-lunch-spot}

Olet juuri muuttanut uuteen kaupunkiin, jossa on 200 ravintolaa.

- **ε-greedy** = "Siirry nykyiseen suosikkiini, mutta rullaa kerran kymmenessä päivässä
  kuole ja valitse *täysin satunnainen* ravintola." Saat näytteitä laajasti, mutta
  *päämäärättömästi* — ja jatkat paikkoja, joita jo vihasit.
- **Optimistinen alustus** = "Oletetaan, että *jokaisessa* ravintolassa en ole
  kokeiltu on kaupungin paras, kunnes toisin todistetaan." Teet järjestelmällisesti
  käy läpi kaikki 200 ja rajaa jokainen pois, koska todellisuus pettää sinut -
  ja löydät todella upeat nopeasti.
- **UCB (Upper Confidence Bound)** = "Pidä suosikkini, mutta anna *bonus* paikkoihin, joissa olen tuskin
  kokeillut - mitä vähemmän tiedän siitä, sitä suurempi bonus." Tämä on fiksua
  *mitä* tuntemattomia paikkoja kokeilla tänään, mutta jokainen päätös on paikallinen:
  se valitsee parhaimman näköisen vaihtoehdon *tällä hetkellä* suunnittelematta reittiä
  läpi kokonaisten tutkimattomien kaupunginosien. Se ei ajattele: "Minun pitäisi ylittää
  kaupungin itäpuolella, koska siellä on kaksikymmentä kokeilematonta paikkaa
  tuolla" — jokainen ravintola arvioidaan erikseen, askel askeleelta.
- **Lukuperusteinen palkkiobonus** = kuten UCB, mutta myös *nautit uutuudesta
  itse* — ateria upouudessa paikassa on luonnostaan tyydyttävä, ja
  että tyytyväisyys muokkaa pitkän aikavälin suunnitelmasi siitä, mihin lähiöihin
  vaeltaa sisään.
- **Prediction-error reward bonus** = "Saan jännitystä ateriasta, joka
  *yllätti* minut - jotain, jota en olisi voinut ennustaa." Uusi paikka, joka
  tuleeko juuri odotetusti? meh. Sellainen, joka on hurjasti erilainen
  henkinen mallisi? Kiehtovaa, ja päivität suunnitelmasi saadaksesi lisää
  pidä siitä.

## Viisi strategiaa (kaikki osoitteessa `compare_exploration.py`) {#the-five-strategies-all-in-compare_explorationpy}

### 1. ε-ahne — oletusarvo, ja se on *haittailua*, ei tutkimista {#1-ε-greedy--the-default-and-its-dithering-not-exploring}

Toimi ahneesti, mutta todennäköisyydellä ε suorita tasaisen satunnainen toiminto. Se on
DQN:n ja ystävien vakioperusviiva. Sen kohtalokas puute vaikeissa tehtävissä:
**jokainen askel on itsenäinen kolikonheitto.** Kompastua ketjun alas `N`
oikeilla liikkeillä tarvitset kolikon laskeutuaksesi oikealle `N` kertaa peräkkäin - se on
eksponentiaalisesti epätodennäköistä. ε-ahne on *jigglee*, ei *tutkimista*.

### 2. Optimistinen alustus – "syytön, kunnes osoittautuu tylsäksi" {#2-optimistic-initialisation--innocent-until-proven-boring}

Aloita *jokainen* Q-arvo suurimmalla mahdollisella tuotolla,
`R_max / (1 − γ)`. Nyt toimi, jota agentti ei ole koskaan kokeillut, näyttää tältä
paras asia maailmassa, joten **ahne**-politiikka on pakotettu kokeilemaan sitä;
vasta käytyään siellä arvo putoaa kohti totuutta. Optimismi
noin *kokeilemattomista alueista **etenee automaattisesti arvon kautta
toiminto** (Q-learningin bootstrapin kautta), joten agenttia vedetään askel kerrallaan
askel kohti maailman osia, joita se ei ole nähnyt. Melkein ilmainen, ei ylimääräistä
kirjanpito – ja, kuten näet, vahvin *syvän* tutkimusmatkailija pienessä
taulukkomainen maailma.

### 3. UCB-tyylinen toimintavalinta – bonus *valinnassa*, ei *palkinto* {#3-ucb-style-action-selection--bonus-in-the-choice-not-the-reward}

Valitse `argmax_a [ Q(s,a) + c·√(ln t / N(s,a)) ]`: mieluummin arvokasta
toimia, mutta paisuta niitä, joita olet harvoin kokeillut. Kuuluisa monikätisestä
rosvot. Saalis: bonus on voimassa vain **toimintojen valintasäännössä**,
ei koskaan palkkiossa — joten se *ei* virtaa arvofunktion läpi.
UCB on loistava "varmista, että olen kokeillut jokaista toimintaa *tässä* tilassa", mutta
heikko "suunnittelemaan reittiä kaukaiselle tutkimattomalle alueelle".

### 4. Lukuun perustuva **palkkio** -bonus – uteliaisuus, klassinen versio {#4-count-based-reward-bonus--curiosity-the-classic-version}

Lisää `1/√(N(s,a))` **palkkioon** (painolla `beta` joka hajoaa).
Koska se kuuluu palkkioon, Q-learning *edistää sitä: toteaa sen
johtavista uusille alueille tulee arvokkaita. Tämä on MBIE-EB / klassikko
"etsintäbonus" -idea - ja työkohdan 1 ensimmäinen puolisko.

### 5. Ennustusvirhe **palkkio** -bonus – uteliaisuus, ICM/RND-versio {#5-prediction-error-reward-bonus--curiosity-the-icmrnd-version}

Lisää `−log P(s'|s,a)` pienestä opitusta eteenpäin mallista palkkioon
(taas rappeutumisen kanssa `beta`). Viidestä terävin uutuussignaali: sisään
Deterministinen maailma, siirtymän yllätys putoaa ~0:aan tällä hetkellä
olet nähnyt sen kerran sen sijaan, että hiipuisit hitaasti kuten `1/√N`. Taulukko
ICM:n / RND:n serkku – työkohdan 1 toinen puolisko.

## Kaksi testitehtävää {#the-two-test-tasks}

- **Tehtävä A – MiniMontezuma**: avain→ovi→aarreverkkomaailma, vain palkinto
  aarteen luona (~15 täydellistä siirtyy pois). Testit "Voitko selvitä pitkään
  harvoin palkitseva ketju ollenkaan?"
- **Tehtävä B — DeepSea(N)**: oppikirjan syvätutkimusketju, suoritetaan klo.
  pituudet `N = 5, 8, 11, 14`. Palkinto piiloutuu taakse `N` oikeat liikkeet,
  jokaisella pienellä välittömällä hinnalla - joten likinäköinen agentti oppii välttämään
  maksaa eikä koskaan löydä palkintoa. Testit "toimiiko strategiasi edelleen
  ketju *pitenee*?"

## Mitä todella tapahtuu (suorita se ja katso) {#what-actually-happens-run-it-and-see}

**Tehtävä A – MiniMontezuma:**

| strategia | Ensimmäinen aarre | Lopullinen ratkaisunopeus |
|----------|---------------:|-----------------:|
| ε-ahne | ei koskaan | 0.00 |
| optimistinen alku | ~ jakso 1 | 1.00 |
| UCB-toimintojen valinta | ~ jakso 3 | ~0.95 |
| laske palkintobonus | ~jakso 82 | ~0.41 |
| ennustuspalkkiobonus | ~jakso 23 | 1.00 |

**Tehtävä B – Syvämeri, osa siemenistä, jotka löysivät palkinnon:**

| strategia | N = 5 | N = 8 | N = 11 | N = 14 |
|----------|----:|----:|-----:|-----:|
| ε-ahne | 0 | 0 | 0 | 0 |
| optimistinen alku | 1.0 | 1.0 | 1.0 | 1.0 |
| UCB-toimintojen valinta | 1.0 | 1.0 | 0.0 | 0.0 |
| laske palkintobonus | 1.0 | 1.0 | ~0.1 | 0.0 |
| ennustuspalkkiobonus | ~0.9 | ~0.8 | ~0.9 | ~0.2 |

*(Numerot heiluvat hieman satunnaisten siementen kanssa, mutta muoto on kivi
kiinteä.)*

## Oppitunnit {#the-lessons}

1. **ε-ahneus ei ole tutkimista.** Se ei koskaan ratkaise *kumpaakaan* vaikeaa tehtävää.
   Satunnainen dithering ei yksinkertaisesti pujota pitkiä oikeita sekvenssejä. (Vielä
   se on edelleen oletusarvo monissa koodissa - koska *helppoissa* tehtävissä se on
   tarpeeksi hyvä ja kuolleen yksinkertainen.)

2. **Todellinen tutkiminen tarkoittaa optimistista suhtautumista tuntemattomaan – yksi tapa
   tai jotain muuta.** Olitpa sitten optimismia *alkuarvoissa*
   (strategia 2), *toimintavalintaan* (strategia 3) tai a
   *itse tuotettu palkkio* (strategiat 4–5), yhteinen säiettä on: *make
   tutkimaton näyttää houkuttelevalta*, ja anna oppimisen viedä sinut sinne.

3. **Harvan palkkion ruudukossa kaikki neljä "todellista" strategiaa toimivat – ja
   ennuste-error bonus tulee sinne nopeimmin**, koska se tuottaa
   terävin "tämä on uusi" -signaali.

4. ***Syvällä* ketjulla, jossa optimismin on kuljettava pitkä matka
   yksinkertainen mestari on optimistinen alustus.** Se edistää optimismia
   arvofunktion kautta ilmaiseksi. UCB hajoaa ensin (sen bonus
   ei koskaan syötä arvofunktiota, joten se ei voi *suunnitella* syvälle). Palkkio
   bonukset toimivat paremmin – ne *edistävät* – mutta pelkkää taulukkomuotoista Q-oppimista
   on hidas työntämään tätä optimismia pitkälle ketjulle ennen
   bonus hajoaa.

5. **Tämä viimeinen kohta on juuri syy miksi syvän tutkimisen skaalaaminen pikseleihin
   tarvitaan ylimääräistä tulivoimaa** — bootstrapped DQN, RND todellisella hermoverkolla
   (joten optimismi *ylentää* samankaltaisten tilojen sijaan
   lisäämällä yksi solu kerrallaan), Go-Explore (kirjaimellisesti muistaa ja
   palaa lupaaviin valtioihin). Tässä olevat taulukkolelut näyttävät sinulle
   *periaatteet*; todelliset järjestelmät ovat nämä samat periaatteet plus verkko
   joka yleistää.

## Avainsanat muistaa {#key-words-to-remember}

| sana | Merkitys |
|------|---------|
| **Etsinnän ja hyödyntämisen välinen kompromissi** | Kokeile uusia asioita vs. käteistä, mitä tiedät – RL:n keskeinen jännitys |
| **Häiriö** | "Tutkiminen" lisäämällä satunnaista kohinaa toimiin (ε-ahne, Gaussin kohina) – heikko vaikeissa tehtävissä |
| **Optimismia epävarmuuden edessä** | Sateenvarjoperiaate: kohtele tuntematonta ikään kuin se olisi hienoa, kunnes olet tarkistanut |
| **Optimistinen alustus** | Toteuta tämä periaate aloittamalla kaikki arvot suurimmasta mahdollisesta tuotosta |
| **UCB** | Ylempi luottamusraja: valitse `argmax (value + bonus that shrinks with visit count)` |
| **Syvä tutkimus** | Tutkimus, joka vaatii pitkän *yhdenmukaisen* sarjan "epätavallisia" toimia, ei vain yhtä |
| **`beta` hehkutus** | Uteliaisuuden paino pienenee ajan myötä, jotta agentti lopulta lopettaa tutkimisen ja hyväksikäytön |

## Yhden lauseen yhteenveto {#one-sentence-summary}

> **ε-ahneus on vain melua; jokainen todellinen etsintästrategia toimii tekemällä
> tutkimaton näyttää houkuttelevalta – optimististen arvojen, toimintavalinnan kautta
> bonus tai itse tuotettu uutuuspalkkio – ja oikea valinta riippuu
> siitä, onko palkintosi vain *niukka* (kuten yhden piilotetun palkinnon löytäminen
> tasaisella pellolla) tai aidosti *syvällä* (kuten yhdistelmälukko, joka vaatii a
> pitkä, tarkka tiettyjen valintojen sarja.**
