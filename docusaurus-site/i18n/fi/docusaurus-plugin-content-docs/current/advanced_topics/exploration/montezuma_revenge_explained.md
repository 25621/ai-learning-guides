# Harjoittelu Montezuma's Revenge -pelissä 🏛️🔑

## Miksi tämä peli on kuuluisa (RL-piireissä)

Vuonna 2015 DeepMindin DQN oppi pelaamaan kymmeniä Atari-pelejä
yli-inhimillisellä tasolla raakapikseleistä. Se pääsi otsikoihin. Mutta tulostaulukon
syövereihin oli haudattu eräs peli, jossa DQN teki **0** - eli saman verran kuin tekisi
tekemättä mitään: **Montezuma's Revenge**.

Miksi? Katso, mitä peli pyytää sinulta aivan ensimmäisessä huoneessa:

1. Kiipeä *alas* tikkaat.
2. Kävele reunan yli.
3. Hyppää pyörivän kallon yli (jos kosket siihen → kuolet).
4. Kiipeä *ylös* toiset tikkaat.
5. Tartu avaimeen.

Se on noin **100 tarkkaa ohjainpainallusta**, eikä peli anna sinulle
**ainuttakaan pistettä** ennen kuin avain on kädessä. Palkintosignaali on
tasainen ja piirteetön **nolla** koko avausjakson ajan.

Normaali RL-agentti oppii mukautumalla ympäristöstä saamiinsa palkkioihin.
Jos palkkio on nolla kaikissa tiloissa, joita agentti voi saavuttaa, *ei ole mitään, mistä oppia* —
se on kuin yrittäisi löytää täysin tasaisen laakson pohjaa
ilman alamäen tuntua. Joten DQN vain harhailisi paikoillaan
aloitustasolla ikuisesti. Montezumasta tuli **haastavan etsinnän benchmark-tehtävä** (hard exploration benchmark):
peli, jonka voi voittaa vain, jos tutkii *systemaattisesti ja taitavasti*, ei
satunnaisesti.

Läpimurto tapahtui vuonna 2018 **Random Network Distillation (RND)** -menetelmällä —
ja temppu oli juuri edellisessä osiossa esitelty **luontainen
uteliaisuusbonus** (intrinsic reward). Sen avulla agentti palkitsee *itsensä* uusien tilojen saavuttamisesta,
mikä tarjoaa tiheän signaalin, joka vetää sitä syvemmälle peliin. RND
saavutti yli-inhimillisen pistemäärän Montezumassa. (Myöhempiä menetelmiä: Go-Explore, Agent57,…)

## Tosielämän esimerkkejä "Montezuma-tyylisestä" harvasta palkinnosta

- **Yhdistelmälukko / aarteenetsintä salaperäisillä vihjeillä.** Ei osittaista
  luotto. Olet nollassa, kunnes olet yhtäkkiä palkinnon ääressä.
- **Tieteellisen artikkelin hyväksyminen tai kannattavan yrityksen perustaminen.**
  Kuukausien ajan ei ulkoista palkkiota, ja sitten (ehkä) saadaan se suuri onnistuminen.
- **Videopelin speedrun-reitti.** Kymmenet kehyksen mukaiset tulot peräkkäin
  ilman palautetta, ennen kuin temppu joko toimii tai ei.
- **Pakohuoneet.** Huoneet eivät kerro juuri mitään ennen kuin olet yhdistänyt
  useita löytöjä toisiinsa.

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

Kyseessä on tutkimusprojekti, ei opetuskäyttöön tarkoitettu skripti. Joten `montezuma_revenge.py`
tekee sen sijaan kaksi asiaa:

### 1. Se kokeilee oikeaa peliä (jos "ale-py" on asennettu)

Se lataa `ALE/MontezumaRevenge-v5`-ympäristön Gymnasiumin kautta, suorittaa **tasaisen
satunnaista käytäntöä 2000 askeleen ajan** ja raportoi pelin kokonaispalkinnon.
Sen tulostama luku on lähes aina **0.0** — näin abstrakti käsite "harva
palkinto" muuttui konkreettiseksi ja todistetuksi tosiasiaksi. Jos Atari-
pakettia ei ole asennettu, se tulostaa yksirivisen `pip install` -komennon ja
siirtyy eteenpäin.

### 2. Se kouluttaa taulukkoagentin *mittakaavassa*: "MiniMontezumaEnv"

Tämä on pieni ruudukkomailma, jolla on sama perusrakenne kuin Montezuman ensimmäisessä
huoneessa:

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
**aarre** (~5 liikettä). Tarvitaan noin **15 täydellistä askelta**, ja palkinto on **nolla
aarteen saavuttamiseen asti**. `has_key`-muuttuja on osa agentin tilaa, joten
kun nappaat avaimen, toisesta huoneesta avautuu täysin *uusia* tiloja
löydettäväksi — aivan kuin uudet ruudut ja tasot avautuvat todellisessa pelissä.

Koulutamme sitten tavallisen **taulukkomuotoisen Q-oppijan** kahdesti:

| Agentti | Tulos MiniMontezumasta |
|-------|--------------------------|
| **ei uteliaisuutta (epsilon-ahne)** | Paluu pysyy **0** kaikissa 1 500 jaksossa. Se ei koskaan edes saavuta avainta. (Kuulostaako tutulta? Se on oikean pelin DQN.) |
| **ennustus-virhe-uteliaisuusbonuksella** | Saavuttaa aarteen noin 20–25 jaksossa ja oppii sitten **optimaalisen 15-vaiheisen reitin**. (Se on RND-idea, kutistettu sopimaan Q-taulukkoon.) |

Kuvassa on kaksi oppimiskäyrää vierekkäin plus todellinen
reitti, jonka utelias agentti oppi, piirrettynä ruudukkoon (alku → avain → ovi →
aarre). Skripti myös tulostaa kyseisen reitin ASCII-kehyksinä.

## Oppitunti

> **Harva palkinto ("sparse reward") ei ole vain yhden oudon Atari-pelin omituisuus – se on
> oletuksena missä tahansa maailmassa, jossa menestys vaatii pitkän ja tarkan
> toimintasarjan.** Vain ulkoisesta palkkiosta oppiva agentti (vanilja-DQN) ei kirjaimellisesti voi päästä
> edes alkuun: seurattavaa gradienttia ei ole olemassa. Uteliaisuusbonus luo
> sellaisen — tiheän, itse luodun "tämä on uutta, jatka" -signaalin — ja
> tuo signaali kuljettaa agentin nollapalkintojen aavikon poikki
> ensimmäiselle todelliselle palkinnolle. Kaikki sen jälkeen tulleet menetelmät (RND, Go-Explore, Agent57) ovat
> samoja ideoita hyödyntäviä, suuremman mittakaavan neuroverkkoversioita.

## Avainsanat muistaa

| sana | Merkitys |
|------|---------|
| **Kova tutkimus** | Ongelmia, joissa onnistut vain tutkimalla taitavasti; satunnainen etsintä epäonnistuu |
| **Harva palkinto** | Palkkio on nolla melkein kaikkialla; saat sen vasta pitkän oikean sarjan jälkeen |
| **Montezuman kosto** | Atari-peli, josta klassiset syvä-RL-agentit (DQN, A3C) saivat 0 – kanoninen kovan tutkimisen vertailukohta |
| **RND (satunnainen verkkotislaus)** | Vuoden 2018 menetelmä, joka voitti Montezuman käyttämällä ennustusvirhe-uteliaisuusbonusta |
| **Go-Explore** | "Muista lupaavat osavaltiot, palaa niihin ja tutki sitten sieltä" - toinen Montezuma-krakkeri |
| **Pienoismalli** | Pieni, halpa ympäristö, joka säilyttää vaikean ongelman *rakenteen*, jotta voit tutkia sitä nopeasti |

## Yhden lauseen yhteenveto

> **Montezuma's Revenge on peli, joka opetti, ettei vahvistusoppimisessa palkinto, jota et koskaan
> saavuta, voi opettaa sinulle mitään — ja korjauskeino niin silloin kuin nytkin on
> uteliaisuusbonus, jonka avulla agentti palkitsee itsensä tutkimisesta siihen asti, kunnes se
> löytää todellisen palkinnon.**
