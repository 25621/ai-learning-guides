# Tavoite-ehdollinen käytäntö

## Suuri idea: yksi käytäntö hallitsee niitä kaikkia

Kuvittele, että olet jakelukuljettaja. Et tarvitse täysin erilaista osaamista jokaista eri osoitetta varten. Osaat ajaa, lukea karttaa ja navigoida liikenteessä – liität vain *tämän päivän määränpään* ja lähdet.

**Tavoite-ehdollinen käytäntö (goal-conditioned policy)** toimii samalla tavalla. Sen sijaan, että kouluttaisimme yhtä agenttia, joka voi saavuttaa vain yhden kiinteän tavoitteen, koulutamme yhden agentin, joka ottaa minkä tahansa tavoitteen syötteeksi ja selvittää, miten sinne päästään.

---

## Miten se eroaa tavallisesta RL:stä

Normaalissa RL-oppimisessa palkintofunktio on määritelty kiinteästi seuraavasti: "saavuta ruutu (7, 7), saat +1." Agentti oppii täsmälleen yhden asian: kuinka tavoittaa *tuo* tietty ruutu.

Tavoite-ehdollisessa vahvistusoppimisessa palkinto riippuu siitä, saavuttaako agentti *minkä tahansa sille kyseisellä kerralla annetun tavoitteen*. Käytäntö oppii:

> **"Mitä minun pitäisi tehdä, kun otetaan huomioon missä olen ja missä haluan olla?"**

Tavoite kulkee *agentin mukana*, aivan kuten navigointisovellukseen syötetty määränpää.

---

## Harvan palkkion ongelma

Tässä on koukku: harvoista palkinnoista (sparse rewards, eli vain +1 maalissa ja 0 muualla) oppiminen on äärimmäisen vaikeaa. Useimmat yritykset epäonnistuvat — agentti harhailee satunnaisesti, ei koskaan saavuta maalia, eikä verkko saa mitään hyödyllistä oppimateriaalia.

Kuvittele, että yrität oppia heittämään tikkaa silmät sidottuina. Heität tuhat kertaa ja heität aina ohi. Tuhansien epäonnistumisten jälkeen et vieläkään tiedä, miltä "hyvä heitto" tuntuu.

Tässä tulee apuun **Hindsight Experience Replay (HER)**.

---

## Hindsight Experience Replay (HER) -toistopuskuri: Epäonnistumisista oppiminen

HER-toistopuskurin idea on nerokkaan yksinkertainen. Epäonnistuneen jakson jälkeen HER kysyy:

> *"Vaikka et saavuttanutkaan tavoitettasi… mihin oikein päädyit?"*

Sitten se **toistaa saman jakson**, mutta teeskentelee agentin todellisen lopullisen sijainnin **olleen** tavoite koko ajan. Yhtäkkiä epäonnistuneesta jaksosta tulee onnistunut – tosin eri tavoitteen saavuttamiseksi.

Se on kuin epäonnistunut koripalloilija, joka heittää jatkuvasti korin ohi. HER sanoisi: "Okei, osuit vasempaan seinään joka kerta. Onnittelut – olet loistava lyömään vasenta seinää! Kirjataan ne heitot onnistuneiksi vasemman seinän lyöntiyrityksiksi." Ajan myötä pelaaja kehittää taitojaan lyödä *mitä tahansa* kohdetta ja pystyy siirtämään nämä taidot lopulta oikeaan koriin.

Tämä muuttaa tuhansia "epäonnistumisia" rikkaaksi kirjastoksi *onnistuneita* navigointeja moniin eri paikkoihin. Agentti oppii saavuttamaan ne kaikki, mikä yleistyy lopulta todelliseen kohteeseen.

---

## Analogia tosielämästä: taapero oppii pinoamaan lohkoja

Taaperolapsi, joka yrittää laittaa lohkon ämpäriin, missaa jatkuvasti. Mutta jokainen "miss" laskeutuu korttelin *johonkin*. Jos toistat jokaisen poissaolon sanomalla "yritit laittaa sen *oikealle* – ja teit sen!", taapero kehittää hienomotoriikkaa koko pöydässä. Pian he voivat asettaa lohkon mihin tahansa - myös ämpäriin.

---

## Mitä koodimme tekee

Skripti `goal_conditioned_policy.py` kulkee **7x7 sokkelossa** seinien kanssa. Jokaisen jakson alussa valitaan satunnainen maalisolu. Agentin on löydettävä se.

Käytännössä syötteitä on kaksi jokaisessa vaiheessa:
1. Missä agentti tällä hetkellä on
2. Minne se haluaa mennä

Jokaisen jakson (onnistuneen tai epäonnistuneen) jälkeen HER luo useita synteettisiä "onnistuksia" merkitsemällä todelliset vieraillut paikat uudelleen vaihtoehtoisiksi tavoitteiksi.

Koulutus kestää 3 000 jaksoa vähenevällä tutkintanopeudella (exploration rate) – agentti tutkii aluksi enemmän ja luottaa sitten yhä enemmän oppimaansa.

---

## Mitä kaaviot osoittavat

![Tavoite-ehdollisen käytännön tulokset](outputs/goal_conditioned_policy.png)

**Vasemmalla – onnistumisprosentti harjoittelun aikana:** Jokainen jakso on joko menestys (tavoitteen saavuttaminen) tai epäonnistuminen. Käyrä nousee tasaisesti agentin yleisen navigointitaidon kehittyessä. Lopulta agentti saavuttaa minkä tahansa tavoitteen melkein joka kerta.

**Oikealla — maalin onnistumisprosentin lämpökartta (heatmap):** Harjoittelun jälkeen testaamme agenttia kaikissa mahdollisissa maalisoluissa ja värjäämme jokaisen solun sen mukaan, kuinka usein agentti saavuttaa sen. Vihreä tarkoittaa, että agentti saavuttaa luotettavasti kyseisen kohdan; punainen tarkoittaa, että se kamppailee edelleen. Hyvin koulutettu agentti näyttää enimmäkseen vihreää koko sokkelossa.

---

## Missä tämä näkyy todellisessa maailmassa

| Sovellus | "tavoite" |
|-------------|------------|
| Robotin käsi ulottuu | Kohdista 3D-asento |
| Itse ajava auto | GPS-koordinaatti |
| Kielimallipohjainen avustaja | Käyttäjän ohje |
| Videopeli, ei-pelaajahahmo | Mikä tahansa reittipiste kartalla |

Tavoite-ehdolliset käytännöt ovat yksi HIRO-arkkitehtuurin (Hierarchical RL with subgoals) rakennuspalikoista – korkean tason johtaja valitsee osatavoitteen, ja matalan tason työntekijä on juuri tällainen tavoitteellinen käytäntö.

---

## Yhden lauseen yhteenveto

> **Tavoite-ehdollinen käytäntö on agentti, joka voi navigoida mihin tahansa kohteeseen – ja HER mahdollistaa epäonnistumisesta oppimisen teeskentelemällä, että jokainen ohi mennyt yritys oli alun perinkin suunnattu sinne, minne se lopulta päätyi.**
