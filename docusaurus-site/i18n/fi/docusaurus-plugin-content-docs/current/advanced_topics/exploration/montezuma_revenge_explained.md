# Treeni Montezuman kostosta 🏛️🔑

## Miksi tämä peli on kuuluisa (RL-piireissä)

Vuonna 2015 DeepMindin DQN oppi pelaamaan kymmeniä Atari-pelejä osoitteessa
yli-inhimillinen taso raakapikseleistä. Se pääsi otsikoihin. Mutta haudattiin
tulostaulukko oli yksi peli, jossa DQN teki **0** - sama kuin tekisi
ei mitään: **Montezuma's Revenge**.

Miksi? Katso, mitä peli pyytää sinulta aivan ensimmäisessä huoneessa:

1. Kiipeä *alas* tikkaat.
2. Kävele reunan yli.
3. Hyppää pyörivän kallon yli (mittaus → kuolet).
4. Kiipeä *ylös* toiset tikkaat.
5. Tartu avaimeen.

Se on noin **100 tarkkaa painikkeen painallusta**, ja peli antaa sinulle
**ei ainuttakaan pistettä** ennen kuin avain on kädessä. Palkintosignaali on a
tasainen, piirteetön **nolla** koko avausjakson ajan.

Normaali RL-agentti oppii mukautumalla todellisuudessa saamiinsa palkkioihin.
Jos palkkio on nolla kaikkialla, missä se voi saavuttaa, *ei ole mitään opittavaa
alkaen* — se on kuin yrittäisi löytää täysin tasaisen laakson pohjaa
tunne alamäkeen. Joten DQN vain nykisi ympäri
aloitusalusta ikuisesti. Montezumasta tuli *kovan *vertailu
tutkimus**: peli, jonka voit voittaa vain, jos tutkit *taitavasti*, et
satunnaisesti.

Läpimurto tapahtui vuonna 2018 **Random Network Distillation (RND)** avulla —
ja temppu oli juuri työkohdan 1 aihe: lisää **luonnollinen
uteliaisuusbonus**, joten agentti palkitsee *itsensä* uusien näyttöjen saavuttamisesta,
ja yhtäkkiä siinä on tiheä signaali, joka vetää sen syvemmälle tasolle. RND
sai yli-inhimillisen pisteen Montezumasta. (Myöhemmin: Go-Explore, Agent57,…)

## Tosielämän esimerkkejä "Montezuma-tyylisestä" harvasta palkinnosta

- **Yhdistelmälukko / aarteenetsintä salaperäisillä vihjeillä.** Ei osittaista
  luotto. Olet nollassa, kunnes olet yhtäkkiä palkinnon ääressä.
- **Paperin hyväksyminen tai kannattavuuden aloittaminen.** Kuukausi
  ei ulkoista palkkiota, sitten (ehkä) iso.
- **Videopelin speedrun-reitti.** Kymmenet kehyksen mukaiset tulot peräkkäin
  ilman palautetta, ennen kuin temppu joko toimii tai ei.
- **Pakohuoneet.** Huone ei kerro sinulle juuri mitään ennen kuin olet kahlittu
  useita löytöjä yhdessä.

Kaikissa näissä "kokeile vain satunnaisia juttuja" on toivotonta. Sinun täytyy
*järjestelmällisesti* tutkia – ja sisäinen "ooh, se on uutta, jatka"
signaali pitää sinut systemaattisena.

## Miksi emme itse asiassa harjoittele Pixel Montezumalla täällä

*oikean* asian tekeminen oikein tarkoittaa:

- konvoluutioverkko 210 × 160 RGB-näytön näkemiseksi,
- kehyksen pinoaminen (jotta agentti voi tietää, mihin suuntaan kallo liikkuu),
- RND-moduuli (kaksi muuta verkkoa: kiinteä satunnainen "kohde" ja koulutettu
  "ennustaja"),
- ja **kymmeniä miljoonia ympäristökehyksiä** - monta GPU-tuntia.

Se on tutkimusprojekti, ei opetuskäsikirjoitus. Joten `montezuma_revenge.py`
tekee sen sijaan kaksi rehellistä asiaa:

### 1. Se *koskettaa* oikeaa peliä (jos "ale-py" on asennettu)

Se latautuu `ALE/MontezumaRevenge-v5` Gymnasiumin kautta, pitää **univormua-
satunnainen agentti 2000 askelta** ja raportoi pelin kokonaispalkinnon. The
sen tulostama numero on lähes aina **0.0** - abstrakti lause "harva
palkkio" muuttui konkreettiseksi, tee se itse tosiasiaksi. Jos Atari
pakettia ei ole asennettu, se tulostaa yksirivisen `pip install` komento ja
siirtyy eteenpäin.

### 2. Se kouluttaa taulukkoagentin *mittakaavassa*: "MiniMontezumaEnv"

Tämä on pieni verkkomaailma, jolla on sama *luuranko* kuin Montezuman ensimmäisellä
huone:

```
###############
#S....#.......#
#.....#.......#
#.....#...T...#     S = start
#.....D.......#     K = key      D = door (only passable while carrying the key)
#..K..#.......#     T = treasure (the ONLY tile that gives a reward)
###############
```

Voittaaksesi sinun tulee: kävellä **avaimelle** (~6 liikettä), nostaa se; kävellä kohti
**ovi** (~4 liikettä) — joka nyt avautuu; kävellä läpi ja saavuttaa
**aarre** (~5 liikettä). Noin **15 täydellistä liikettä**, **nolla palautetta
aarteeseen asti**. The `has_key` lippu on osa agentin tilaa, joten
kun nappaat avaimen, sinulla on toinen huone *uusia* tiloja
löydä – aivan kuin uudet näytöt avautuvat todellisessa pelissä.

Koulutamme sitten tavallisen **taulukkomuotoisen Q-oppijan** kahdesti:

| Agentti | Tulos MiniMontezumasta |
|-------|--------------------------|
| **ei uteliaisuutta (epsilon-ahne)** | Paluu pysyy **0** kaikissa 1 500 jaksossa. Se ei koskaan edes saavuta avainta. (Kuulostaako tutulta? Se on oikean pelin DQN.) |
| **ennustus-virhe-uteliaisuusbonuksella** | Saavuttaa aarteen noin 20–25 jaksossa ja oppii sitten **optimaalisen 15-vaiheisen reitin**. (Se on RND-idea, kutistettu sopimaan Q-taulukkoon.) |

Kuvassa on kaksi oppimiskäyrää vierekkäin plus todellinen
reitti utelias agentti oppinut, piirretty ruudukko (käynnistys → avain → ovi →
aarre). Skripti myös tulostaa kyseisen reitin ASCII-kehyksinä.

## Oppitunti

> **"Sharse Reward" ei ole yhden oudon Atari-pelin omituisuus – se on
> oletuksena missä tahansa maailmassa, jossa menestys vaatii pitkän, tietyn sarjan
> toimia.** Vain palkkio -agentti (vanilja DQN) ei kirjaimellisesti voi saada
> aloitettu: gradienttia ei ole seurattava. Uteliaisuusbonus valmistaa
> yksi — tiheä, itse luoma "tämä on uutta, jatka" -signaali - ja
> tuo signaali kuljettaa agentin nollien aavikon poikki
> ensimmäinen todellinen palkinto. Kaikki sen jälkeen (RND, Go-Explore, Agent57) on a
> samasta liikkeestä suurennettu hermoverkkoversio.

## Avainsanat muistaa

| sana | Merkitys |
|------|---------|
| **Kova tutkimus** | Ongelmia, joissa onnistut vain tutkimalla taitavasti; satunnainen etsintä epäonnistuu |
| **Hiha palkinto** | Palkkio on nolla melkein kaikkialla; saat sen vasta pitkän oikean sarjan jälkeen |
| **Montezuman kosto** | Atari-peli, josta klassiset syvä-RL-agentit (DQN, A3C) saivat 0 – kanoninen kovan tutkimisen vertailukohta |
| **RND (satunnainen verkkotislaus)** | Vuoden 2018 menetelmä, joka voitti Montezuman käyttämällä ennustusvirhe-uteliaisuusbonusta |
| **Go-Explore** | "Muista lupaavat osavaltiot, palaa niihin ja tutki sitten sieltä" - toinen Montezuma-krakkeri |
| **Pienoismalli** | Pieni, halpa ympäristö, joka säilyttää vaikean ongelman *rakenteen*, jotta voit tutkia sitä nopeasti |

## Yhden lauseen yhteenveto

> **Montezuma's Revenge on peli, joka opetti, että RL "palkitsee sinua koskaan
> vastaanottaminen ei voi opettaa sinulle mitään" - ja korjaus silloin ja nyt on a
> uteliaisuusbonus, jonka avulla agentti palkitsee itsensä tutkimisesta siihen asti
> löytää oikean palkinnon.**
