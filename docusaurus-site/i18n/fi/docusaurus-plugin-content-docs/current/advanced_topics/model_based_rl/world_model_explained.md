# Maailmanmallin kouluttaminen: Agentin opettaminen unelmoimaan 🌍

## Mikä on "maailmanmalli"?

**Maailmamalli** on agentin *sisäinen kopio maailmankaikkeudesta*. Anna se a
tila ja toiminta, ja se ennustaa mitä tapahtuu seuraavaksi:

```
(state, action)  ──►  Neural Network  ──►  (next_state, reward)
```

Se ei ole todellinen maailma - se on **simulaattori, jonka agentti on rakentanut itselleen**
katsoa todellisuutta ja oppia matkimaan sitä.

Kun malli on koulutettu, agentti antaa agentin kysyä "mitä jos" -kysymyksiä ottamatta
mikä tahansa todellinen toiminta:

> *"Jos painan nyt vasemmalle ja sitten kahdesti oikealle, mihin päädyn? Will napa
> pudota?"*

Agentti voi pohtia sata suunnitelmaa mallissaan siinä ajassa, joka kestää
tehdäksesi yhden oikean liikkeen. Se on koko pointti.

---

## Analogia tosielämästä

Mieti, kuinka *sinä* ratkaiset pulman. Et liikuta fyysisesti jokaista kappaletta
jokaiseen aukkoon. **Kuvittele** mitä tapahtuu, jos pala A menee tänne. Jos se
henkinen simulaatio näyttää väärältä, hylkäät sen ennen kuin nostat sormea.

Aivoillasi on oppinut maailmanmalli, joka on rakennettu vuosien näkemästä esineitä
käyttäytyä – sen avulla voit simuloida tuloksia ennen sitoutumista.

Muita esimerkkejä:

- **Sahkin pelaaja** kuvittelee liikkeet useita kierroksia eteenpäin.
- **Kuljettaja** ajattelee: "Jos jarrutan nyt, takana olevalla autolla on tarpeeksi tilaa."
- **Lapsi** pinoaa palikkoja: "Jos laitan ison päälle, torni tekee
  heilua." (He oppivat tämän mallin kaatamalla torneja aiemmin.)

Joka tapauksessa **henkinen malli + mielikuvitus = parempia päätöksiä pienemmällä riskillä**.

---

## Kuinka agentti rakentaa mallinsa?

Se vain **kelloa**. Erityisesti:

1. **Kerää dataa.** Anna minkä tahansa käytännön (jopa satunnaisen) olla vuorovaikutuksessa todellisen kanssa
   ympäristöä hetkeksi. Tallenna jokainen siirtymä:
   ```
   (state, action, reward, next_state)
   ```
2. **Opettele neuroverkko** ennustamaan `next_state` ja `reward` alkaen
   `(state, action)`. Tämä on ohjattua oppimista: jokainen tallennettu siirtymä on a
   merkitty esimerkki, jossa syöte on "mitä agentti näki ja teki" ja
   otsikko on "mitä todella tapahtui seuraavaksi".
3. **Validoi.** Pidä 10 % tiedoista ja tarkista mallin ennusteet
   todellisia vastaan. Pieni virhe tarkoittaa, että malli on tallentanut
   ympäristön **dynamiikka**: miten tilat muuttuvat toimien jälkeen.

Käyttämämme temppu: ennustamisen sijaan `next_state` suoraan, ennustaa
**delta** `next_state − state`. Suurin osa fysiikasta on inkrementaalista ("kärry liikkui a
tiny bit"), ja pienet kohteet ovat hermoverkkoja ystävällisiä.

---

## Meidän kokoonpanomme

| Valinta | Arvo | Miksi? |
|--------|-------|-----|
| Ympäristö | `CartPole-v1` | 4-D-tila, 2 toimintoa – helppo mallintaa |
| Data | 20 000 siirtymää satunnaisesta käytännöstä | Laaja kattavuus valtiotilasta |
| Verkko | MLP, 2 × 128 ReLU piilotettu | MLP = Multi-Layer Perceptron (tavallinen "vanilja" neuroverkko). Kaksi piilotettua kerrosta 128 neuronista ReLU-aktivaatioiden avulla. Riittävä kapasiteetti, nopea harjoitella. |
| Tappio | MSE päällä `(delta_state, reward)` | MSE = Mean Squared Error (neliöityjen ennustusvirheiden keskiarvo). Normaali regressiohäviö. |
| Optimoija | Adam, lr = 1e-3, 30 aikakausia | Adam = mukautuva optimoija (säätää oppimisnopeudet parametria kohti automaattisesti). Valmis hyllystä tarkoittaa, että erityistä viritystä ei tarvita. |

Koko harjoitus päättyy muutamassa sekunnissa CPU:lla.

---

## Miltä "hyvä" näyttää?

Kaksi diagnostista merkitystä:

### 1. Yksivaiheinen tarkkuus (validointi MSE)

Tämä on "kuinka hyvin malli ennustaa YKSI askeleen tulevaisuuteen?" 30 jälkeen
aikakausina sinun pitäisi nähdä validointi MSE välillä **1e-4 - 1e-3**. Se on
pieni – napojen kulmat ja kärryjen sijainnit ovat tarkkoja muutaman desimaalin tarkkuudella.

### 2. **Komppausvirhe** k-vaiheen käyttöönotoissa

Tämä on *todellinen* testi. Ota tila, syötä se mallin läpi ja ota sitten
sen ennuste ja syötä se takaisin mallin kautta - varten `k` askeleet peräkkäin.
Virhe kasvaa, koska jokainen askel lisää hieman kohinaa edellisen päälle
ennustus.

```
Step  1:  L2 error ≈ 0.01   (almost perfect)
Step  5:  L2 error ≈ 0.05
Step 10:  L2 error ≈ 0.15
Step 20:  L2 error ≈ 0.40   (visibly drifting)
```

*(L2-virhe = Euklidinen etäisyys ennustetun seuraavan tilan ja todellisen välillä —
ajattele sitä seuraavasti: "Kuinka kaukana mallin arvaus on 4-D-tilaavaruudessa?")*

**Miksi tällä on väliä.** Jos suunnittelemme mallin kanssa 15 askelta eteenpäin, *tarkka*
vaiheessa 15 oleva tila on väärä - mutta jos "hyvien suunnitelmien" suhteellinen sijoitus
vs. huonot suunnitelmat" säilyy, suunnittelu toimii edelleen. (Tämä on mitä
`model_based_planning.py` hyväksikäytöt.)

Juoni sisään `outputs/world_model.png` näyttää molemmat diagnoosit rinnakkain:
harjoittelu-tappiokäyrä menee hienosti alas lokimittakaavassa ja rollout-virhe
käyrä nousee mukavasti.

---

## Miksi ennustaa *Delta*?

Vertaa kahta tapaa ilmaista sama ongelma verkkoon:

| Kohde | Tyypillinen suuruusluokka | Helppoa vai vaikeaa? |
|--------|------------------:|--------------|
| `next_state`        | 0–2,4 (kärryn sijainti) | Verkon on toistettava sijainti **ja** pieni muutos |
| `next_state - state`| ~0.02            | Verkko vain oppii pienen muutoksen |

Deltan ennustaminen tarkoittaa myös: jos verkko tulostaa nollia (harjoittamattomana, aloittelijana
verkko usein tekee), ennuste on yksinkertaisesti "mitään ei liikuta" – järkevä ja turvallinen oletus yhdelle
aikaaskel. Sitä vastoin absoluuttisen ennustaminen `next_state` suoraan antaisi aluksi täysin satunnaisia roskaarvoja, mikä aiheuttaa varhaisen harjoittelun olevan erittäin epävakaa.

---

## Mitä tämä ostaa meille

Koulutettu maailmanmalli on perusta:

- **Suunnittelu** — hae kuviteltuja toimintajaksoja (katso
  `model_based_planning.py`).
- **Dyna-tyylinen lisäys** — kouluta Q-verkko kuviteltujen tietojen pohjalta
  moninkertaistaa näytteen tehokkuuden.
- **Uteliaisuus / tutkiminen** — käynti kertoo, että malli ei pysty ennustamaan hyvin.
- **Dreamer / World-Models -paperit** — harjoita *politiikka* kokonaan sisällä
  malli, jossa ei ole todellista vuorovaikutusta alkuperäisen tiedonkeruun lisäksi.

---

## Rajoitukset ja varoitukset

- **Out-of-distribution drift.** Malli tuntee vain osan maailmasta
  on nähnyt. Suunnittele liian aggressiivisesti ja päädyt alueille, joita mallissa ei ole koskaan ollut
  vieraillut - ennusteet ovat puhdasta fantasiaa.
- **Korjausvirhe.** Suunnittelu pitkillä **horisonteilla** (monia askeleita tulevaisuuteen) on epäluotettavaa kertyvien virheiden vuoksi, kuten kaavio osoittaa.
  Nykyaikaiset järjestelmät korjaavat tämän käyttämällä **todennäköisyyspohjaisia ryhmiä** (kouluttaa useita malleja ja tarkistaa, ovatko ne samaa mieltä, kuten PETS tai Dreamer), joten suunnittelija
  tietää tarkalleen *kuinka epävarma* malli on joka vaiheessa ja voi välttää riskialttiit, tuntemattomat polut.
- **Stokastiset ympäristöt.** Normaali deterministinen regressori ennustaa vain *keskiarvon* keskiarvon
  lopputulos ja jättää täysin huomiotta mahdollisten tulosten leviämisen. Monimutkaiset, todelliset ympäristöt vaativat todennäköisyyslaskentaa
  mallit (kuten ne, joissa on Gaussin lähdöt tai **latentit stokastiset mallit** – verkot, jotka
  koodaa maailmantila todennäköisyysjakaumana pakatussa avaruudessa,
  antaa heidän vangita aidon satunnaisuuden sen sijaan, että laskeisi sen keskiarvon pois), jotta se edustaa tarkasti epävarmuutta ja satunnaisuutta.

---

## Avainsanat

| Termi | Pelkkää englantia |
|------|---------------|
| **Maailman malli** | Hermoverkko, joka jäljittelee ympäristöä |
| **Dynamiikka** | Toiminto `(s, a) → s'` |
| **Palkintomalli** | Toiminto `(s, a) → r` (usein mukana) |
| **Yksivaiheinen ennuste** | Mitä malli tuottaa todellisesta tilasta |
| **Käyttöönotto** | Toistuvat yksivaiheiset ennusteet, syöttämällä ulostulot takaisin |
| **Korjausvirhe** | Pienet virheet, jotka kasvavat käyttöönoton aikana |

---

## Yhden lauseen yhteenveto

> **Maailmamalli on pieni hermokopio universumista, jonka agentti voi
> neuvottele – ja haaveile sisälläsi – ennen kuin riskeeraat tositoimia.**

Seuraavaksi: `model_based_planning.py` panee tämän mallin toimimaan varsinaisessa päätöksenteossa.
