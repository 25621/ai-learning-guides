# Oppitun suunnittelumallin (MPC) käyttäminen 🔮

## Suuri Idea {#the-big-idea}

Sinulla on **maailmamalli** (hermoverkko, joka ennustaa tulevaisuutta).
Mitä nyt?

Suorinta käyttöä on **suunnittelu**: kysy joka hetki mallilta "mitä
tapahtuisi, jos kokeilisin *tätä* suunnitelmaa? *se* suunnitelma? *se toinen* suunnitelma?"
Tee sitten se suunnitelma, joka näyttää parhaalta – mutta **vain sen ensimmäinen askel**.
Koska malli ei ole täydellinen, suoritamme vain yhden toiminnon, tarkkailemme todellista uutta tilaa todellisesta ympäristöstä ja suunnittelemme sitten uudelleen alusta.

Tällä temppulla on nimi: **Model Predictive Control** (MPC).

---

## Analogia tosielämästä {#a-real-life-analogy}

Olet ravintolassa katsomassa valikkoa. Et sitoudu viiden kurssiin
tilaa paikan päällä – tilaat ensimmäisen ruokalajin ja katsot sitten, kuinka täyteläiseksi tunnet olosi
päättää jälkiruoasta uudelleen.

Tai: ajat mutkaisella tiellä. Et lukitse ohjaustuloja
seuraavat 30 sekuntia – katsot jatkuvasti eteenpäin, suunnittelet muutaman sekunnin, otat
seuraava ohjaustoiminto ja suunnittele uudelleen.

Tuo **plan-far / act-near / re-plan** -silmukka on MPC.

---

## Kuinka "Satunnainen ammunta" toimii {#how-random-shooting-works}

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

## Miksi suunnitella jokainen askel uudelleen? {#why-re-plan-every-step}

Koska malli on epätäydellinen. Virheet yhdistetään käyttöönoton aikana (kuten näkyy kaaviossa, jonka on luonut `world_model.py`, tallennettu `outputs/world_model.png`). Vaiheen 0 suunnitelma on luotettava vain muutaman ensimmäisen liikkeen aikana; vaiheessa 15 malli on hallusinaatio. Joten luotamme vain
**ensimmäinen siirto** ja päivitä sitten suunnitelma uusimpaan todelliseen tilaan.

Tämä on sama syy, miksi ihmiset eivät kirjoita 100 liikkeen shakkisuunnitelmaa ja pidä siitä kiinni
se – olosuhteet muuttuvat, ja mitä kauempana arvelet, sitä vähemmän se vastaa
todellisuutta.

---

## Ryppy: Palkinnon on kerrottava suunnittelijalle jotain {#a-wrinkle-the-reward-has-to-tell-the-planner-something}

CartPolessa todellinen palkinto on `+1` joka askeleella, kunnes napa putoaa. Malli
ennustaa uskollisesti `+1, +1, +1, ...` melkein jokaiseen suunnitelmaan, koska
satunnaiset suunnitelmat harvoin päättyvät nopeasti mallin sisällä - ja niin jokainen suunnitelma
pisteet samat. Suunnittelijalla ei ole valinnanvaraa.

Korjaus: korvaa todellinen palkkio **pehmeällä välityspalvelimella** suunnittelun aikana:

```python
reward_proxy(state) = 1
                    - |pole_angle| / 0.21          # pole upright?  (1=yes)
                    - 0.1 * |cart_position| / 2.4  # cart centred?  (1=yes)
```

Nyt suunnitelmat, jotka *päättyisivät* kaatuneeseen tangon, saavat näkyvästi huonompia arvoja kuin
suunnitelmat, jotka pysyvät pystyssä. Suunnittelija voi luokitella ne.

> **Oikean elämän oppitunti.** Tasainen palkintosignaali – "selvisit toisen sekunnin" -
> on hyödytön lyhyen aikavälin suunnittelussa. Tiheät, muotoillut signaalit auttavat.

---

## Mitä koodimme tekee {#what-our-code-does}

`model_based_planning.py`:

1. **Lataa** maailman mallin painot, jotka säästävät `world_model.py`. (Jos he ovat
   puuttuu, se kouluttaa yhden uudelleen lennossa.)
2. **Käyttää 20 jaksoa** MPC:stä oikealla CartPole-v1:llä.
3. **Käyttää myös 20 jaksoa** tasaisen satunnaisen käytännön pohjalta.
4. **Piirtoi** sekä vierekkäin että tulostaa keskiarvot.

### Mitä sinun pitäisi nähdä, kun käytät sitä {#what-you-should-see-when-you-run-it}

| politiikka | Keskimääräinen palkinto (vaiheet selviytyneet) |
|--------|-------------------------------:|
| Satunnainen        | ~22 (tyypillistä CartPolelle – sauva putoaa nopeasti) |
| MPC (meidän)    | ~150–500 (vaihtelee siemenittain; monet jaksot lähellä 500) |
| Max mahdollista  | 500 |

Tämä **5–25-kertainen parannus** saavutetaan ilman politiikkaverkostoa tai arvoa
toimi, eikä lisäkoulutusta. Vain maailmanmalli + 200 unta askelta kohti.

Juoni `outputs/model_based_planning.png` näyttää kaksi värillistä palkkia jaksoa kohden
— MPC lähes aina pitempi kuin Random, ja monet jaksot osuvat
500 portaan katto.

---

## Mallipohjaisen suunnittelun vahvuudet {#strengths-of-model-based-planning}

- **Sample tehokas.** Kaikki oppiminen tehtiin yhdestä satunnaisesta erästä
  siirtymät. Hyödyllisen politiikan muodostamiseksi ei tarvinnut enempää env-vuorovaikutusta.
- **Helppo kohdistaa uudelleen.** Haluatko hallita agenttia eri tavalla? Vaihda
  palkkion välityspalvelin – ei uudelleenkoulutusta. (Kokeile maksimoida kärryn nopeus huvin vuoksi.)
- **Tulkitavissa.** Voit tarkastaa suunnitelmat, joita agentti harkitsi
  ennustetut lentoradat ja pisteet.

## Heikkoudet (ja mitä ihmiset tekevät niistä) {#weaknesses-and-what-people-do-about-them}

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
- **Todellinen palkkio on se, mitä haluamme.** Palkinnon muotoilu auttaa, mutta varten
  monimutkaisempia tehtäviä ihmiset oppivat **arvofunktion** koulutettu *sisällä*
  maailmanmalli - oppinut kriitikko, joka arvioi pitkän aikavälin tuoton mistä tahansa
  tila ilman käsintehtyä välityspalvelinta. Molemmat **Dreamer** (joka harjoittelee
  näyttelijä-kriitikko täysin piilevässä mielikuvituksessa) ja **MuZero** (joka yhdistää
  MCTS oppitun arvoverkoston kanssa) käyttävät tätä ideaa.

---

## Kuinka tämä liittyy nykyaikaisiin järjestelmiin {#how-this-connects-to-modern-systems}

Juuri suorittamasi tarkka resepti – **oppinut dynamiikkaa + suunnittelu** – on se
nykyaikaisen tekoälytutkimuksen vahvimpien RL-järjestelmien selkäranka:

- **MuZero** (DeepMind): yhdistää opitun maailmanmallin Monte Carlo Tree Searchiin. Se hallitsee Go, shakki, shogi ja Atari ilman sääntöjä etukäteen.
- **Dreamer / DreamerV3** (Hafner et al.): kouluttaa politiikkaa *oppineen* sisällä
  **latentti-avaruus**-maailmanmalli (eli malli pakkaa raakakuvat tai tilat kompaktiksi, abstraktiksi esitykseksi ennen tulevaisuuden ennustamista). Se saavuttaa huippuluokan (huipputason) suorituskyvyn yli 100:ssa vertailussa.
- **PETS / PlaNet / TD-MPC**: nämä ovat algoritmiperheitä (tutkimuslinjoja), jotka skaalaavat juuri tämän idean monimutkaisiin jatkuviin ohjaustehtäviin, kuten robotiikkaan.

Olet rakentanut - muutamassa sadassa rivissä - perheen pienimmän jäsenen.

---

## Avainsanat {#key-words}

| Termi | Pelkkää englantia |
|------|---------------|
| **MPC**           | Mallin ennakoiva ohjaus – suunnittele etukäteen, toimi kerran, suunnittele uudelleen |
| **Satunnainen ammunta** | Tee paljon satunnaisia suunnitelmia, valitse paras |
| **Horisontti (H)**   | Kuinka monta askelta suunnitelma näyttää eteenpäin |
| **N näytettä**     | Kuinka monta ehdokassuunnitelmaa harkitsemme askelta kohti |
| **Vähentyvä horisontti** | Suunnittele uudelleen joka vaiheessa sen sijaan, että sitoutuisi yhteen suunnitelmaan |
| **Palkintojen välityspalvelin / muotoilu** | Sujuva korvikepalkkio, joka antaa suunnittelijalle hyödyllisen signaalin optimointia varten |

---

## Yhden lauseen yhteenveto {#one-sentence-summary}

> **Kun sinulla on maailmanmalli, suunnittelu on vain "unelma sadasta tulevaisuudesta,
> valitse paras ensimmäinen askel, toista."**

Se on mallipohjaisen RL:n koko salaisuus.
