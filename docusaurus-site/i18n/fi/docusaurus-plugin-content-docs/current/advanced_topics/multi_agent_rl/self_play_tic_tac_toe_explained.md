# Itseleikki: Agentin opettaminen antamalla sen leikkiä itseään ♟️

## Mitä on itsepelaaminen?

Kuvittele lapsi, joka haluaa päästä todella hyväksi shakissa, mutta jolla ei ole ketään, jota pelata.
Joten hän pelaa itseään. Vasen käsi vs oikea käsi. Jokaisessa pelissä *molemmat* osapuolet yrittävät
voittaa. Jokaisessa pelissä *molemmat* osapuolet oppivat, mikä toimi.

Tämä on **itsepeliä**: yksi agentti toimii molempina pelaajina ja joka liike
siitä tulee opetus sille, joka muuttaa seuraavaksi. Ei opettajaa, ei asiantuntevaa vastustajaa.
Vain oppija, joka on myös omat tikkaansa.

Itsepeli kuulostaa tempulta – tarvitset varmasti oikean vastustajan? - mutta on
viime vuosikymmenen kuuluisimpien RL-virstanpylväiden takana oleva moottori:
**AlphaGo Zero**, **AlphaZero**, **MuZero**, **OpenAI Five**. He kaikki käyttävät
itse leikkiä. Syy on yksinkertainen: kun parannat, vastustajasi paranee
sama määrä. Haaste vastaa aina taitojasi.

---

## Miksi se toimii

Kolme asiaa tekevät itsepelaamisesta erityisen:

1. **Loputtomia vastustajia.** Pelit eivät lopu koskaan. Vastustaja on aina
   läsnä ja ilmainen.
2. **Opetussuunnitelma, joka kasvaa kanssasi.** Aloittelija voi pelata vain muuta
   aloittelijat. Kun paranet, myös varjosi paranee - automaattisesti.
3. **Symmetria.** Nollasummapelissä (toisen pelaajan voitto on toisen tappio),
   yksi Q-arvojen joukko kuvaa molempia puolia; käännät vain kyltin, kun se
   on toisen pelaajan vuoro. Joten *yksi* Q-taulukko voi opettaa itsensä.

Tic-tac-toe on täydellinen testialus: tarpeeksi pieni mahtumaan sanakirjaan, mutta
niin monimutkainen, että satunnaiset liikkeet johtavat lähes aina tappioon strategista pelaajaa vastaan.

---

## Analogia tosielämästä

- **Tennistä seinää vasten.** Et voi hävitä seinälle, mutta sinä
  voi harjoitella syöttöjäsi. Itseleikki tekee tämän molemmissa päissä – sinä olet
  seinä *ja* soitin, ja vaihdat edestakaisin.
- **Keskustelukerho, joka väittelee molemmin puolin.** Parempia väittelijöitä syntyy
  puolustaa aina päinvastaista näkemystä kuin mitä he henkilökohtaisesti uskovat.
  Jokainen argumentti harjoittelee sekä hyökkäystä että puolustusta.
- **AlphaGo Zero.** Se oppi nolla-ihmispeleistä. Alkaen satunnaisesta
  liikkuu se pelasi miljoonia pelejä itseään vastaan; muutamassa päivässä se oli
  parempi kuin kaikki edellinen Go-ohjelma, mukaan lukien se, joka voitti Leen
  Sedol.

---

## Mitä koodimme tekee

Opimme yhden Q-pöydän *nykyisen pelaajan siirrettäväksi*:

```
Q[(board, player_to_move)][action] = expected return for that player
```

Harjoituskierros on:

1. Käynnistä tyhjä taulu. `player = X`.
2. Molemmat pelaajat toimivat **saman agentin** kanssa käyttämällä ε-greedyä.
3. Kävele jokaisen pelin jälkeen taaksepäin jokaisen (lauta, pelaaja, toiminta) läpi.
   kolminkertaistaa historian ja käytä Q-learning-päivitystä.
4. Palkinto kääntää merkin kierrosten välillä: jos X voittaa, jokainen X:n tekemä liike saa
   +1 (tai bootstrap-arvo tulevasta voittajatilasta); jokainen O:n tekemä liike saa -1.
5. Laskemme (vähennämme) tutkimusnopeuttamme (ε) hitaasti arvosta 0,2 → 0,02, joten agentti sitoutuu parhaaseen peliinsä
   myöhään harjoittelussa satunnaisten liikkeiden yrittämisen sijaan.

Joka 2 500 jaksoa arvioimme agentin **satunnaiseen vastustajaan**
(jäädytämme oppimisprosessin, joten Q-taulukkoon ei tehdä uusia päivityksiä arvioinnin aikana ja molemmat osapuolet pelaavat ahneesti). Agentin pitäisi voittaa tai tasapeli
~100 % näistä peleistä riittävän itsepelin jälkeen.

### Mitä sinun pitäisi nähdä

50 000 itsepelatun jakson jälkeen:

| Ottelu | Odotettu tulos |
|----------|-----------------|
| Koulutettu agentti vs satunnainen vastustaja (1000 peliä) | **~95-99 % voittoja tai tasapelejä**, käytännössä 0 % tappioita |
| Koulutettu agentti vs. Itse (200 ahnetta peliä) | **Kaikki 200 arvontaa**. Tic-tac-toe on peli, joka päättyy aina tasapeliin, jos molemmat pelaajat pelaavat täydellisesti. Se, että itsepelaaminen vetää jokaista peliä, on merkki lähentymisestä. |

Juoni `outputs/self_play_tic_tac_toe.png` näyttää agentin voiton/tasapelin/tappion
murto-osat vs. satunnainen vastustaja ajan myötä:
- Voittoprosentti alkaa ~60 % (kun molemmat pelaajat pelaavat satunnaisesti, ensimmäisellä pelaajalla on luontainen etu, koska he saavat lisää merkkejä pelilaudalle, mikä johtaa pelaajan X noin 60 % voittoprosenttiin).
- Nousee yli 90 %:iin.
- Tappioaste putoaa lähes 0 prosenttiin.

Käsikirjoitus tulostaa myös näytepelin liikkeeltä liikkeelle lopussa, jotta näet
agenttipeli.

---

## Varo näitä hienouksia

- **Sign flips väliä.** Yleinen bugi: unohtaa, että "vastustaja
  niiden arvon maksimoiminen" tarkoittaa *omamme minimoimista* bootstrap-kohteessa.
  Koodimme päivitys käyttää `target = reward - gamma * max(Q[next, opponent])`.
- **Symmetriaa ei hyödynnetä tässä.** Todellinen toteutus kanonisoituisi
  laudat (eli ne pyörittävät tai heijastavat minkä tahansa kortin tilan vakiomuotoon, ainutlaatuiseen "normaalimuotoon", jotta agentti tunnistaa identtiset korttitilanteet) jakamaan Q-arvot 8:lle.
  symmetriat. Ohitamme tämän – tilatila on tarpeeksi pieni raa'alla voimalla.
- **Q-pöytä kasvaa.** 50 000 itsepelatun pelin jälkeen näet muutaman
  tuhat state-player-näppäintä. Se on hyvä tässä; shakille tai Go:lle
  tarvitsevat sen sijaan hermoverkon, minkä vuoksi **AlphaZero korvaa
  pöytä, jossa on CNN + MCTS**.

---

## Missä itsepeli katkeaa

- **Ei-nollasummapelit.** "Molemmat osapuolet ovat tyytyväisiä" ei ole yhteensopiva
  symmetrinen leikki; et voi vain kääntää kylttiä.
- **Epäsymmetriset roolit.** Jos "hyökkääjällä" ja "puolustajalla" on eri toiminta
  tilaa, tarvitset kaksi erillistä verkkoa.
- **Strategiapyöräily.** Puhdas itsepeli voi jäädä jumiin
  kivi-paperi-sakset kaltaiset pyörät. AlphaStar korjasi tämän pitämällä suuren
  *pooli* (tai "liiga") agentin ja poiminnan aiemmista versioista
  vastustajia kyseisestä poolista satunnaisesti, joten agentti oppii voittamaan monia
  eri pelityylit nykyisen sijaan.
- **Palkkion hakkerointi.** Itsepelaaminen tekee molemmista osapuolista älykkäämpiä, mutta vain
  peli *sellaisena kuin määritit sen*. Jos palkitsemisjärjestelmässäsi on tahattomia porsaanreikiä (kuten pelaajan palkitseminen vain siitä, että hän selviytyy pidempään voiton sijaan), molemmat osapuolet käyttävät hyväkseen porsaanreikää, mikä johtaa omituiseen, hyödyttömään käyttäytymiseen sen sijaan, että hallittaisiin varsinaista peliä.

---

## Avainsanat muistaa

| sana | Merkitys |
|------|---------|
| **itseleikki**      | Sama agentti pelaa molemmin puolin peliä |
| **nollasumma**       | Yhden pelaajan voitto = toisen tappio |
| **Symmetria**       | Yksi Q-pöytä voi palvella molempia puolia, jos käännät kylttejä |
| **Väestöleikki**| Pelaa itse *monien* menneiden versioiden kanssa itsestäsi vastustajina (AlphaStar) |
| **Opetussuunnitelma**     | Luonnollinen vaikeusaste – itsepelaaminen saa sen ilmaiseksi |
| **MCTS**           | Monte-Carlo Tree Search — suunnittelualgoritmi AlphaZero yhdistää itsepelin |

---

## Yhden lauseen yhteenveto

> **Itsepelaaminen muuttaa parantumisen omiksi tikkaiksi: aina kun saat
> parempi, vastustajasi tekee myös automaattisesti.**

Tämä idea on laajennettu **hermoverkoilla** (aivojen inspiroima matemaattinen).
toiminnot, jotka oppivat kuvioita tiedosta) ja puuhaku, päihittävät parhaat ihmiset
Go, shakki, shogi, Dota 2 ja StarCraft.
