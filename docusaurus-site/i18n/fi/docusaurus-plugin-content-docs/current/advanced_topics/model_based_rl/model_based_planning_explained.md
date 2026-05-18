# Opitun maailmanmallin käyttäminen suunnittelussa (MPC) 🔮

## Suuri idea

Sinulla on **maailmanmalli** (neuroverkko, joka ennustaa tulevaisuutta).
Mitä nyt?

Suorinta käyttöä on **suunnittelu**: kysy joka hetki mallilta: "Mitä tapahtuisi, jos kokeilisin *tätä* suunnitelmaa? Tai *tuota* suunnitelmaa? Tai *tätä kolmatta* suunnitelmaa?"
Valitse sitten parhaalta vaikuttava suunnitelma – mutta suorita **vain sen ensimmäinen askel**.
Koska maailmanmalli ei ole täydellinen, suoritamme vain yhden toiminnon, tarkkailemme todellista uutta tilaa todellisesta ympäristöstä ja suunnittelemme sitten kaiken uudelleen alusta alkaen.

Tällä tempulla on nimi: **Model Predictive Control** (MPC).

---

## Analogia tosielämästä

Olet ravintolassa katsomassa ruokalistaa. Et tilaa viiden ruokalajin illallista heti alkuun
– tilaat ensin alkupalan ja katsot sitten, kuinka kylläiseksi tunnet itsesi
ennen kuin päätät jälkiruoasta.

Tai: ajat mutkaisella tiellä. Et lukitse ohjausliikkeitä
seuraavaksi 30 sekunniksi – katsot jatkuvasti eteenpäin, suunnittelet muutaman sekunnin etukäteen, suoritat
seuraavan ohjaustoiminnon ja suunnittelet sitten uudelleen.

Tuo **plan-far / act-near / re-plan** -silmukka (suunnittele pitkälle, toimi lähellä, suunnittele uudelleen) on MPC.

---

## Kuinka "Satunnainen ammunta" toimii

On olemassa kehittyneempiä suunnittelijoita - esimerkiksi:
- **CEM** (Cross-Entropy Method): tarkenna iteratiivisesti suunnitelmien jakautumista pitämällä vain top-k kullakin kierroksella.
- **MCTS** (Monte Carlo Tree Search): rakenna hakupuu simulaatiotilastojen ohjaamana, jota käyttävät AlphaGo ja MuZero.
- **Gradienttipohjaiset suunnittelijat**: erottele mallin ennusteet toimien suhteen ja seuraa gradienttia suoraan.

Käytämme yksinkertaisinta toimivaa: **satunnaiskuvausta**.

```
Given the current state s:
    1. Sample N=200 random action sequences of length H=15.
    2. For each sequence, simulate it through the world model from s, summing
       a shaped reward at each step.   (200 dreams in parallel — fast!)
    3. Find the sequence with the highest predicted total reward.
    4. Execute that sequence's FIRST action in the real environment.
    5. Observe the real next state.  Discard the rest of the plan.
    6. Go to step 1 — re-plan from scratch.
```

200 suunnitelmaa × 15 askelta = 3 000 kuviteltua siirtymää todellista askelta kohti. Maailma
malli ajaa ne kaikki yhdessä erässä hermoverkon eteenpäinsiirrossa - tyypillisesti
muutama millisekunti.

---

## Miksi suunnitella jokainen askel uudelleen?

Koska malli on epätäydellinen. Virheet yhdistetään käyttöönoton aikana (kuten näkyy kaaviossa, jonka on luonut `world_model.py`, tallennettu `outputs/world_model.png`). Vaiheen 0 suunnitelma on luotettava vain muutaman ensimmäisen liikkeen aikana; vaiheessa 15 malli alkaa hallusinoida. Joten luotamme vain
**ensimmäiseen siirtoon** ja päivitämme sitten suunnitelman uusimman todellisen tilan mukaan.

Tämä on sama syy, miksi ihmiset eivät kirjoita 100 siirron shakkisuunnitelmaa ja pidä siitä
kiinni – olosuhteet muuttuvat, ja mitä kauemmas tulevaisuuteen yrität arvioida, sitä vähemmän se vastaa
todellisuutta.

---

## Ryppy: Palkinnon on kerrottava suunnittelijalle jotain

CartPolessa todellinen palkinto on `+1` joka askeleella, kunnes tanko putoaa. Malli
ennustaa uskollisesti `+1, +1, +1, ...` melkein jokaiselle suunnitelmalle, koska
satunnaiset suunnitelmat päättyvät harvoin nopeasti mallin sisällä – ja siten jokaisen suunnitelman
pisteet ovat samat. Suunnittelijalla ei ole mahdollisuutta valita parasta.

Korjaus: korvataan todellinen palkkio **pehmeällä välityspalkkiolla (proxy reward)** suunnittelun aikana:

```python
reward_proxy(state) = 1
                    - |pole_angle| / 0.21          # onko tanko pystyssä? (1=kyllä)
                    - 0.1 * |cart_position| / 2.4  # onko kärry keskellä? (1=kyllä)
```

Nyt suunnitelmat, jotka *päättyisivät* tangon kaatumiseen, saavat näkyvästi huonompia arvoja kuin
suunnitelmat, jotka pysyvät pystyssä. Suunnittelija voi pisteyttää ja järjestää ne paremmuusjärjestykseen.

> **Oikean elämän oppitunti.** Tasainen palkintosignaali – "selvisit toisen sekunnin" -
> on hyödytön lyhyen aikavälin suunnittelussa. Tiheät, muotoillut signaalit auttavat.

---

## Mitä koodimme tekee

`model_based_planning.py`:

1. **Lataa** maailmanmallin painot, jotka tallennettiin skriptissä `world_model.py`. (Jos ne
   puuttuvat, se kouluttaa mallin uudelleen lennossa.)
2. **Käyttää 20 jaksoa** MPC:stä oikeassa CartPole-v1-ympäristössä.
3. **Käyttää myös 20 jaksoa** tasaisen satunnaista käytäntöä vertailukohtana.
4. **Piirtää** molemmat vierekkäin kuvaajaan ja tulostaa keskiarvot.

### Mitä sinun pitäisi nähdä, kun käytät sitä

| Käytäntö | Keskimääräinen palkinto (selviydytyt askeleet) |
|--------|-------------------------------:|
| Satunnainen        | ~22 (tyypillistä CartPolelle – tanko kaatuu nopeasti) |
| MPC (meidän)    | ~150–500 (vaihtelee satunnaissiementen mukaan; monet jaksot lähellä 500) |
| Maksimi | 500 |

Tämä **5–25-kertainen parannus** saavutetaan täysin ilman erillistä käytäntöverkkoa tai arvofunktiota,
eikä se vaadi lisäkoulutusta. Vain opittu maailmanmalli + 200 uneksittua suunnitelmaa askelta kohti.

Kuvaaja `outputs/model_based_planning.png` näyttää kaksi värillistä palkkia jaksoa kohden
— MPC on lähes aina huomattavasti pidempi kuin Random, ja monet jaksot yltävät
500 askeleen maksimikattoon.

---

## Mallipohjaisen suunnittelun vahvuudet

- **Näytetehokas (sample efficient).** Kaikki oppiminen tapahtui yhdestä satunnaisten
  siirtymien erästä. Hyödyllisen käytännön muodostamiseksi ei tarvittu lainkaan uutta vuorovaikutusta ympäristön kanssa.
- **Helppo kohdistaa uudelleen.** Haluatko ohjata agenttia toisella tavalla? Vaihda vain
  palkkion välityspalkkiota – skriptiä ei tarvitse kouluttaa uudelleen. (Kokeile vaikkapa maksimoida kärryn nopeus!)
- **Tulkittavissa.** Voit tarkastaa agentin harkitsemat suunnitelmat,
  ennustetut reitit ja niiden pisteet.

## Heikkoudet (ja mitä ihmiset tekevät niistä)

- **Satunnainen ammunta on typerää.** Se näyttelee suunnitelmia sokeasti. Korkeammalle
  mitat, valitse **CEM** (Cross-Entropy Method – katso edellä) tai
  **iLQR** (Iteratiivinen Linear-Quadratic Regulator, klassinen optimaalinen ohjaus
  menetelmä, joka approksimoi mallin paikallisesti lineaarisena ja ratkaisee sen
  analyyttisesti) tai täysin **gradienttipohjainen**-suunnittelija parantaa toimintoja seuraamalla gradientteja
  erottuva malli.
- **Yhdistetty mallivirhe.** Pitkät horisontit ajautuvat. Ihmiset käyttävät **todennäköisyyslaskentaa
  kokoonpanot** (useita malleja, jotka on koulutettu samoilla tiedoilla, kuten PETS, Chua et
  al. 2018), jotta suunnittelija voi havaita erimielisyydet ja rangaista suunnitelmista
  malli on epävarma.
- **Todellinen palkkio on se, mitä haluamme.** Palkkion muotoilu (reward shaping) auttaa, mutta
  monimutkaisempia tehtäviä varten ihmiset oppivat **arvofunktion**, joka on koulutettu *maailmanmallin sisällä*
  – eli opitun arvioijan (critic), joka arvioi pitkän aikavälin tuottoa mistä tahansa
  tilasta ilman käsintehtyä välityspalkkiota. Sekä **Dreamer** (joka kouluttaa
  toimija-arvioijan (actor-critic) täysin latentissa mielikuvituksessa) että **MuZero** (joka yhdistää
  MCTS-puuhaun opittuun arvoverkkoon) käyttävät tätä ideaa.

---

## Kuinka tämä liittyy nykyaikaisiin järjestelmiin

Juuri suorittamasi tarkka resepti – **oppinut dynamiikkaa + suunnittelu** – on se
nykyaikaisen tekoälytutkimuksen vahvimpien RL-järjestelmien selkäranka:

- **MuZero** (DeepMind): yhdistää opitun maailmanmallin Monte Carlo Tree Searchiin. Se hallitsee Go, shakki, shogi ja Atari ilman sääntöjä etukäteen.
- **Dreamer / DreamerV3** (Hafner et al.): kouluttaa käytäntöä *oppineen* sisällä
  **latentti-avaruus**-maailmanmalli (eli malli pakkaa raakakuvat tai tilat kompaktiksi, abstraktiksi esitykseksi ennen tulevaisuuden ennustamista). Se saavuttaa huippuluokan (huipputason) suorituskyvyn yli 100:ssa vertailussa.
- **PETS / PlaNet / TD-MPC**: nämä ovat algoritmiperheitä (tutkimuslinjoja), jotka skaalaavat juuri tämän idean monimutkaisiin jatkuviin ohjaustehtäviin, kuten robotiikkaan.

Olet rakentanut - muutamassa sadassa rivissä - perheen pienimmän jäsenen.

---

## Avainsanat

| Termi | Selkosuomeksi |
|------|---------------|
| **MPC**           | Malliennakoiva ohjaus (Model Predictive Control) – suunnittele eteenpäin, toimi kerran, suunnittele uudelleen |
| **Satunnaisammunta** | Tehdään paljon satunnaisia suunnitelmia ja valitaan niistä paras |
| **Horisontti (H)**   | Kuinka monta askelta suunnitelma katsoo eteenpäin |
| **N näytettä**     | Kuinka monta ehdokassuunnitelmaa harkitaan askelta kohti |
| **Vähenevä horisontti** | Suunnitellaan uudelleen joka askeleella sen sijaan, että sitouduttaisiin yhteen pitkään suunnitelmaan |
| **Palkintojen välityspalkkio / muotoilu** | Jatkuva ja tiheä korvikepalkkio, joka antaa suunnittelijalle hyödyllisen signaalin optimointia varten |

---

## Yhden lauseen yhteenveto

> **Kun sinulla on maailmanmalli, suunnittelu on vain "unelma sadasta tulevaisuudesta,
> valitse paras ensimmäinen askel, toista."**

Se on mallipohjaisen RL:n koko salaisuus.
