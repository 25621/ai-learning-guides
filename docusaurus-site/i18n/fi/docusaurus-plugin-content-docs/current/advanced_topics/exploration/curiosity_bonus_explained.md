# Uteliaisuusbonus (sisäinen motivaatio) 🧭

## Mikä se on? {#what-is-it}

Kuvittele, että lapsi putoaa uuteen huoneeseen. Kukaan ei maksa heille, kukaan ei taputa -
Silti he tekevät linjan kaapille, jota he eivät ole avanneet, napin
he eivät ole painaneet, meluisa lelu nurkassa. He juoksevat an
**sisäinen palkinto**: *"Tämä näyttää uudelta. Käy katsomassa."*

**Uteliaisuusbonus** antaa vahvistus-oppimisagentille saman
sisäinen asema. Ympäristön todellinen palkinto ("ulkoinen" palkkio)
pisteet, rahat, pelin voitto) jätetään täsmälleen sellaiseksi kuin se on. Lisäämme vain a
toinen, itse tuotettu palkkio vierailusta asioissa, jotka agentti löytää *uudelta*
tai *yllättävää*, ja harjoittele summalla:

```
reward the agent learns from  =  real reward  +  beta * curiosity bonus
```

`beta` on nuppi, joka alkaa suurelta (ole utelias!) ja kutistuu ajan myötä
(Lopeta hölmöily, käy käteisellä, mitä opit).

## Miksi vaivautua? Harva-palkitsemisongelma {#why-bother-the-sparse-reward-problem}

Normaalit RL-agentit oppivat tosiasiallisesti saamistaan palkkioista. Se toimii
hienoa, kun palkintoja on kaikkialla ("+1 jokaisesta askeleesta pysyt pystyssä" sisään
CartPole). Se hajoaa, kun palkinto on **harva** - nolla, nolla,
nolla, ... , nolla ja sitten lopuksi +1 pitkän, erittäin tarkan jälkeen
oikeiden toimien sarja.

Oikeita esimerkkejä niukoista palkkioista:

- **Montezuma's Revenge** (Atari-peli): ensimmäinen pisteesi saapuu vain
  ~100 tarkan liikkeen jälkeen - kiivetä tikkaita alas, väistää kalloa, kiivetä ylös,
  nappaa avain. Siihen asti tulos on tasainen nolla.
- **Yhdistelmälukko.** 9 999 väärää koodia ei anna sinulle mitään; yksi antaa
  sinä palkinto.
- **Lääkelöydöt / tieteelliset kokeet.** Tuhansia epäonnistuneita kokeita,
  sitten toimivan.
- **Pitkän todisteen tai ohjelman kirjoittaminen.** Ei osittaista hyvitystä, ennen kuin koko
  asia selviää.

Pelkästään palkitseva agentti näissä tilanteissa on kuin opiskelija, joka kieltäytyy
opiskele, ellei heille makseta oikeasta vastauksesta lopullisessa - he eivät koskaan saa
alkoi. Uteliaisuus on bonus, joka sanoo *"tutkiminen on oma palkintonsa"*
niin agentti jatkaa tökkimistä, kunnes se kompastuu todelliseen palkintoon.

## Two Flavors of Curiosity (molemmat toteutettu muodossa curiosity_bonus.py) {#two-flavours-of-curiosity-both-implemented-in-curiosity_bonuspy}

### 1. Laskuriin perustuva uutuus: "Olen tuskin ollut täällä" {#1-count-based-novelty-ive-barely-been-here}

Yksinkertaisin mahdollinen uutuussignaali. Pidä laskua `N(s, a)` kuinka monesta
kertaa olet ryhtynyt toimiin `a` tilassa `s`, ja anna itsellesi se bonus
kutistuu, kun summa kasvaa:

```
curiosity bonus  =  1 / sqrt( N(s, a) + 1 )
```

Ensimmäinen kerta, kun yrität jotain: bonus = 1.0. 100 yrityksen jälkeen: bonus = 0,1.
10 000 yrityksen jälkeen: 0,01. Agentti palkitaan siitä, että hän on mennyt sinne, missä se ei ole käynyt
ollut, ja viehe haalistuu luonnollisesti hyvin tallatulta maalta.

**Analogia tosielämästä:** turisti, jolla on luettelo "paikoista, joissa en ole käynyt".
Upouusi naapurusto? Ensisijainen. Kahvila, jossa olet käynyt viisikymmentä
kertaa? Ei jännitä enää.

Tämä on kirjan vanhin idea (MBIE-EB, UCB). Sen heikkous: a
valtava tai jatkuva maailma, et koskaan käy *täsmälleen* samassa tilassa kahdesti, joten
raakaluku on aina 1 – siksi seuraava maku on olemassa.

### 2. Ennustus-virheuutuus: "En nähnyt *sen* tulevan" {#2-prediction-error-novelty-i-didnt-see-that-coming}

Tämä on kuuluisan **ICM:n** (Intrinsic Curiosity Module,
Pathak et ai. 2017) ja sen serkku **RND** (Random Network Distillation,
Burda et ai. 2018). Laskemisen sijaan agentti pitää vähän
**malli, joka yrittää ennustaa mitä tapahtuu seuraavaksi** — "jos olen täällä ja teen
tämä, mihin päädyn?" - ja palkitsee itsensä ** kuinka väärässä malli on
oli**:

```
curiosity bonus  =  surprise  =  -log P( the state I actually reached | where I was, what I did )
```

- Tilanne, jota malli ei ole koskaan nähnyt → se ennustaa huonosti → suuri yllätys
  → iso bonus → "mennä tutkimaan siellä!"
- Tilanne, jonka malli on nähnyt sata kertaa → se ennustaa täydellisesti →
  nolla yllätys → nolla bonus → "ollu paikalla, ymmärretty, siirry eteenpäin."

**Analogia tosielämästä:** lapsi oppii, miten maailma toimii pelaamalla.
Lasin työntäminen pöydältä *ensimmäistä kertaa* on kiehtovaa (se
särkynyt!). Sadalla kerralla tiesit jo, että se särkyy – ei
mielenkiintoista. Uteliaisuus = ero sen välillä, mitä odotit ja mitä
tapahtui.

Taulukkokoodissamme "malli" on vain siirtymämäärien taulukko ja
"Kuinka väärin se oli" on yllätys `-log P`. Todellinen ICM/RND käyttää hermostoa
verkkoja, joten sama idea toimii raakapikseleissä – mutta periaate on
identtinen.

> **Miksi kaksi versiota?** Laskentapohjainen on yksinkertaista ja loistava lähtökohta.
> Ennustusvirhe skaalautuu suuriin, koskaan toistuviin maailmoihin ja antaa a
> *terävämpi* signaali: deterministisessä ympäristössä, kun olet nähnyt a
> siirtyä, kun yllätys putoaa välittömästi arvoon ~0, kun taas luku
> bonus häviää vain hitaasti `1/sqrt(N)`. Kokeissamme
> ennustusvirheagentti ratkaisee MiniMontezuman parissakymmenessä jaksossa;
> myös laskenta-agentti pääsee perille, vain hitaammin ja vähemmän luotettavasti.

## Mitä koodimme tekee {#what-our-code-does}

`curiosity_bonus.py` kouluttaa tavallista **taulukkomuotoista Q-oppijaa**
`MiniMontezumaEnv` - pieni kahden huoneen verkkomaailma, johon sinun täytyy kävellä a
**avain**, nosta se (nyt **ovi** aukeaa), kävele läpi ja saavuta
**aarre**. Palkinto (+1) näkyy *vain* aarteen kohdalla, ~15 jälkeen
täydelliset liikkeet. Se johtaa kolmea agenttia ja suunnittelee niitä:

| Agentti | Mitä se tekee MiniMontezumassa |
|-------|-------------------------------|
| **epsilon-ahne (ei uteliaisuutta)** | Vaeltaa lähellä lähtöä, *ei koskaan* saavuta avainta, tulos pysyy 0 ikuisesti. |
| **lukuperusteinen bonus** | Löytää avaimen luotettavasti; ketjuttaa koko ketjun aarteeseen ehkä ~40 % jaksoista. Toimii - vain vähän meluisa. |
| **ennustus-virhe -bonus** | Ensin saavuttaa avaimen *ja* aarteen ~20–25 jaksossa; kuten `beta` hajoaa se sulautuu sen ratkaisemiseen jokaisessa jaksossa. |

Kuvassa näkyy:
- oppimiskäyrä: *P(tavoita aarre)* harjoittelun yli,
- toinen käyrä välitavoitteelle *P (poista avain)*,
- ja **tilan vierailun lämpökartat** ruudukosta - ei-uteliaisuusagentti
  on tiukka möykky lähellä alkua; uteliaita agentteja tulvii *molemmat* huoneet.

## Mekanismi yhdessä kuvassa {#the-mechanism-in-one-picture}

```
            no curiosity                       with curiosity bonus
   reward:  0 0 0 0 0 0 0 0 ... 0  (+1?)        0 0 0 0 0 0 0 0 ... 0  (+1!)
            └──── nothing to learn from ──┘     └ + 0.4 0.3 0.9 0.2 ... ┘  (self-made,
                                                  dense, points "toward the new stuff")
   result:  random walk, never finds +1         systematically sweeps the world,
                                                 trips over +1, then the bonus fades
```

Uteliaisuusbonus muuttaa *"En ole nähnyt tätä"* palkinnoksi, joten
agentti **työntää tietoisesti tutkimattomalle alueelle** sen sijaan, että
heiluttaa satunnaisesti. Ja koska bonus kutistuu asioiden muuttuessa
tuttu (ja `beta` hajoaa), kun agentti on löytänyt sen todellisen palkkion
lopettaa luonnollisesti ryöstelyn ja alkaa hyödyntää.

## Muutama rehellinen varoitus {#a-few-honest-caveats}

- ** "Noisy-TV-ongelma".** Ennustevirheiden agentti voi hypnotisoitua
  puhtaan satunnaisuuden lähteellä (televisio, joka näyttää staattista, noppia heittäen) - se
  *ei koskaan* voi ennustaa sitä, joten yllätys ei koskaan katoa. ICM:n todellinen temppu on
  ennustaa *oppitussa ominaisuustilassa*, joka jättää agentin huomioimatta
  ei voi hallita; RND sivuuttaa sen eri tavalla. Meidän deterministimme
  gridworldillä ei ole meluisaa televisiota, joten emme osu tähän.
- **Uteliaisuus on keino, ei päämäärä.** Siksi `beta` rappeutuu. Agentti
  joka pysyy maksimaalisena uteliaana ikuisesti, ei koskaan asettu todellisuuteen
  *voitto*.
- **Syvän tutkimisen skaalaaminen on edelleen vaikeaa.** Palkinnon bonus auttaa
  mutta pelkkä taulukkomainen Q-oppiminen on hidasta levittämään tuloksena olevaa optimismia
  pitkä ketju alas (katso `compare_exploration.py`). Halkeileva pikseli
  Montezuma tarvitsi ylimääräistä tulivoimaa - RND hermoverkolla, bootstrapped
  DQN, Go-Explore.

## Avainsanat muistaa {#key-words-to-remember}

| sana | Merkitys |
|------|---------|
| **Sisäinen palkinto** | Palkkio, jonka agentti tuottaa itselleen, ympäristön palkkiosta erillään |
| **Ulkoinen palkinto** | Ympäristön todellinen palkinto (pisteet, voitto/tappio) |
| **Hiha palkinto** | Palkkio on nolla melkein kaikkialla; saat sen vasta pitkän oikean sarjan jälkeen |
| **Uutuus/yllätys** | Kuinka uusi tai odottamaton tila (tai siirtymä) on - asia, jonka uteliaisuus palkitsee |
| **Lukuperusteinen bonus** | Uutuus ≈ `1/sqrt(visit count)` - klassinen tutkimusbonus |
| **ICM** | Intrinsic Curiosity Module: uutuus = eteenpäin suuntautuvan mallin ennustevirhe (opetetussa ominaisuustilassa) |
| **`beta`** | Uteliaisuusbonuksen paino; yleensä hehkutetaan kohti nollaa, jotta agentti lopulta hyödyntää |

## Yhden lauseen yhteenveto {#one-sentence-summary}

> **Uteliaisuusbonus on itsensä antama palkinto uutuudesta – se valmistaa
> tiheä "mene-tutki-tulle" -signaali, joka vetää agentin läpi
> harvoin palkitsevia maailmoja, joita se ei muuten koskaan ratkaisisi, sitten kohteliaasti haalistuu
> pois, kun kaikki on tuttua.**
