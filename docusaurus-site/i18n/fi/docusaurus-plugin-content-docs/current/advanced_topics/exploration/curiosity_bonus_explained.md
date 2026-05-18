# Uteliaisuusbonus (sisäinen motivaatio) 🧭

## Mikä se on?

Kuvittele, että lapsi päätyy uuteen huoneeseen. Kukaan ei maksa lapselle eikä taputa –
silti hän suuntaa suoraan kaapille, jota ei ole vielä avannut, tai nappia kohti, jota ei ole
painanut, tai nurkassa olevan meluisan lelun luo. Häntä ajaa eteenpäin **sisäinen palkinto** (intrinsic reward):
*"Tämä näyttää uudelta. Käy katsomassa."*

**Uteliaisuusbonus** antaa vahvistusoppimisagentille samanlaisen
sisäisen motivaation (intrinsic drive). Ympäristön todellinen palkinto ("ulkoinen" palkkio, kuten
pisteet, rahat, pelin voitto) jätetään täsmälleen sellaiseksi kuin se on. Lisäämme vain
toisen, itse tuotetun palkkion sellaisissa tiloissa vierailemisesta, jotka agentti kokee *uutena*
tai *yllättävänä*, ja harjoittelemme näiden kahden summalla:

```
reward the agent learns from  =  real reward  +  beta * curiosity bonus
```

`beta` on säätönuppi, joka on aluksi suuri (ole utelias!) ja pienenee ajan myötä
(Lopeta hölmöily ja hyödynnä oppimaasi).

## Miksi vaivautua? Harvan palkinnon ongelma

Normaalit RL-agentit oppivat tosiasiallisesti saamistaan palkkioista. Se toimii
hienoa, kun palkintoja on kaikkialla ("+1 jokaisesta askeleesta, jonka pysyt pystyssä"
CartPolessa). Se epäonnistuu, kun palkinto on **harva** – nolla, nolla,
nolla, ... , nolla ja sitten lopuksi +1 vasta pitkän, erittäin tarkan ja
oikean toimintasarjan jälkeen.

Oikeita esimerkkejä niukoista palkkioista:

- **Montezuma's Revenge** (Atari-peli): ensimmäinen pisteesi saadaan vasta
  ~100 tarkan liikkeen jälkeen (kiivetään tikkaita alas, väistetään kalloa, kiivetään ylös ja
  siepataan avain). Siihen asti palkinto on puhdas nolla.
- **Yhdistelmälukko.** 9 999 väärää koodia ei anna mitään; vasta se yksi oikea antaa
  sinulle palkinnon.
- **Lääkelöydöt / tieteelliset kokeet.** Tuhansia epäonnistuneita kokeita,
  ja sitten vihdoin yksi toimiva.
- **Pitkän todisteen tai ohjelman kirjoittaminen.** Ei osittaista hyvitystä, ennen kuin koko
  koodi tai todistus on täysin valmis.

Agentti, joka oppii vain ulkoisista palkkioista, on näissä tilanteissa kuin opiskelija, joka kieltäytyy
opiskelemasta, ellei hänelle makseta jokaisesta oikeasta vastauksesta loppukokeessa – hän ei koskaan pääse
edes aloittamaan. Uteliaisuus on bonus, joka sanoo *"tutkiminen on oma palkintonsa"*,
minkä ansiosta agentti jatkaa etsimistä ja kokeilemista, kunnes se lopulta törmää todelliseen palkintoon.

## Kaksi uteliaisuuden muotoa (molemmat toteutettu tiedostossa `curiosity_bonus.py`)

### 1. Laskuriin perustuva uutuus: "Olen tuskin vieraillut täällä"

Yksinkertaisin mahdollinen uutuussignaali. Pidetään kirjaa tilassa `s` tehdyn toiminnon `a` käyntikerroista `N(s, a)`, ja annetaan itselle bonus,
joka pienenee käyntikertojen kasvaessa:

```
curiosity bonus  =  1 / sqrt( N(s, a) + 1 )
```

Ensimmäinen kerta, kun yrität jotain: bonus = 1.0. 100 yrityksen jälkeen: bonus = 0,1.
10 000 yrityksen jälkeen: 0,01. Agentti palkitaan siitä, että se menee sinne, missä se ei ole aiemmin
ollut, ja houkutus haalistuu luonnollisesti tutuilla alueilla.

**Analogia tosielämästä:** turisti, jolla on luettelo "paikoista, joissa en ole käynyt".
Upouusi naapurusto? Ensisijainen. Kahvila, jossa olet käynyt viisikymmentä
kertaa? Ei jännitä enää.

Tämä on kirjan vanhin idea (MBIE-EB, UCB). Sen heikkous:
valtavassa tai jatkuvassa maailmassa et koskaan päädy *täsmälleen* samaan tilaan kahdesti, joten
laskurin arvo on aina 0 tai 1 – siksi toinen uteliaisuustyyppi on olemassa.

### 2. Ennustus-virheuutuus: "En nähnyt *sen* tulevan"

Tämä on kuuluisan **ICM:n** (Intrinsic Curiosity Module,
Pathak et ai. 2017) ja sen serkku **RND** (Random Network Distillation,
Burda et al. 2018). Laskemisen sijaan agentti ylläpitää pientä
**mallia, joka yrittää ennustaa mitä tapahtuu seuraavaksi** — "jos olen tässä tilassa ja teen
tämän toiminnon, mihin tilaan päädyn?" — ja palkitsee itsensä sen perusteella, **kuinka väärässä malli
oli**:

```
curiosity bonus  =  surprise  =  -log P( saavutettu tila | nykyinen tila, valittu toiminto )
```

- Tilanne, jota malli ei ole koskaan nähnyt → se ennustaa huonosti → suuri yllätys
  → iso bonus → "mennä tutkimaan siellä!"
- Tilanne, jonka malli on nähnyt sata kertaa → se ennustaa täydellisesti →
  nolla yllätys → nolla bonus → "nähty ja koettu, siirrytään eteenpäin."

**Analogia tosielämästä:** lapsi oppii, miten maailma toimii pelaamalla.
Lasin työntäminen pöydältä *ensimmäistä kertaa* on kiehtovaa (se
särkynyt!). Sadalla kerralla tiesit jo, että se särkyy – ei
mielenkiintoista. Uteliaisuus = ero sen välillä, mitä odotit ja mitä
tapahtui.

Taulukkokoodissamme "malli" on vain siirtymämäärien taulukko ja
"Kuinka väärin se oli" on yllätys `-log P`. Todellinen ICM/RND käyttää hermostoa
verkkoja, joten sama idea toimii raakapikseleissä – mutta periaate on
identtinen.

> **Miksi kaksi versiota?** Laskentaan perustuva on yksinkertainen ja loistava lähtökohta.
> Ennustusvirhe skaalautuu suuriin, koskaan toistuviin maailmoihin ja antaa
> *terävämmän* signaalin: deterministisessä ympäristössä, kun siirtymä on kerran nähty,
> yllätys putoaa välittömästi lähelle nollaa (~0), kun taas laskuripohjainen
> bonus häviää vain hitaasti kaavalla `1/sqrt(N)`. Kokeissamme
> ennustusvirheeseen perustuva agentti ratkaisee MiniMontezuman parissakymmenessä jaksossa;
> myös laskuriin perustuva agentti pääsee perille, mutta vain hitaammin ja epävarmemmin.

## Mitä koodimme tekee

`curiosity_bonus.py` kouluttaa tavallista **taulukkomuotoista Q-oppijaa**
`MiniMontezumaEnv`-ympäristössä (pieni kahden huoneen ruudukkomailma), jossa agentin täytyy kävellä
**avaimen** luo, poimia se (jolloin **ovi** aukeaa), kävellä läpi ja saavuttaa
**aarre**. Palkinto (+1) saadaan *vain* aarteesta noin 15
täydellisen askeleen jälkeen. Skripti ajaa kolmea agenttia ja piirtää niiden tulokset kuvaajaan:

| Agentti | Mitä se tekee MiniMontezumassa |
|-------|-------------------------------|
| **epsilon-ahne (ei uteliaisuutta)** | Vaeltaa lähellä aloitusta, *ei koskaan* löydä avainta, ja tulos pysyy nollassa ikuisesti. |
| **laskuripohjainen bonus** | Löytää avaimen luotettavasti ja ratkaisee koko tehtävän aarteen saavuttamiseksi noin 40 %:ssa jaksoista. Toimii, mutta on hieman epävakaa (noisy). |
| **ennustusvirhebonus** | Saavuttaa avaimen *ja* aarteen noin 20–25 jaksossa; kun `beta` pienenee, se oppii ratkaisemaan tehtävän luotettavasti jokaisessa jaksossa. |

Kuvassa näkyy:
- oppimiskäyrä: *P(tavoita aarre)* harjoittelun yli,
- toinen käyrä välitavoitteelle *P(avaimen nouto)*,
- ja **tiloissa vierailujen lämpökartat** ruudukosta — ei-uteliaalla agentilla
  se on tiivis möykky lähellä aloitusta, kun taas uteliaat agentit tutkivat molempia huoneita laajasti.

## Mekanismi yhdessä kuvassa

```
            no curiosity                       with curiosity bonus
   reward:  0 0 0 0 0 0 0 0 ... 0  (+1?)        0 0 0 0 0 0 0 0 ... 0  (+1!)
            └─ ei mitään mistä oppia ───┘       └ + 0.4 0.3 0.9 0.2 ... ┘  (itse luotu,
                                                  tiheä, osoittaa kohti uutta)
   result:  satunnaisvaellus, ei löydä +1       haravoi maailmaa järjestelmällisesti,
                                                 törmää +1:teen, minkä jälkeen bonus vaimenee
```

Uteliaisuusbonus muuttaa *"en ole nähnyt tätä"* -kokemuksen palkinnoksi, joten
agentti **hakeutuu tietoisesti tutkimattomille alueille** sen sijaan, että
vaeltelisi satunnaisesti. Koska bonus kutistuu asioiden muuttuessa
tutuiksi (ja `beta` pienenee), kun agentti on löytänyt todellisen palkkion,
se lopettaa luonnollisesti harhailun ja alkaa hyödyntää oppimaansa (exploitation).

## Muutama rehellinen varoitus

- ** "Noisy-TV-ongelma".** Ennustevirheiden agentti voi hypnotisoitua
  puhtaan satunnaisuuden lähteellä (televisio, joka näyttää staattista, noppia heittäen) - se
  *ei koskaan* voi ennustaa sitä, joten yllätys ei koskaan katoa. ICM:n todellinen temppu on
  ennustaa *oppitussa ominaisuustilassa*, joka jättää agentin huomioimatta
  ei voi hallita; RND sivuuttaa sen eri tavalla. Meidän deterministimme
  gridworldillä ei ole meluisaa televisiota, joten emme osu tähän.
- **Uteliaisuus on vain keino, ei päämäärä.** Siksi `beta` vaimenee. Agentti,
  joka pysyy maksimaalisen uteliaana ikuisesti, ei koskaan keskity keräämään todellisia
  palkintoja ja voittoja.
- **Syvän tutkimisen skaalaaminen on edelleen vaikeaa.** Palkinnon bonus auttaa
  mutta pelkkä taulukkomainen Q-oppiminen on hidasta levittämään tuloksena olevaa optimismia
  pitkinä ketjuina (katso `compare_exploration.py`). Pikselipohjaisen Montezuman
  ratkaiseminen vaati paljon enemmän tulivoimaa — RND:n neuroverkoilla, bootstrapped-
  DQN, Go-Explore.

## Avainsanat muistaa

| sana | Merkitys |
|------|---------|
| **Sisäinen palkinto** | Palkkio, jonka agentti tuottaa itselleen, ympäristön palkkiosta erillään |
| **Ulkoinen palkinto** | Ympäristön todellinen palkinto (pisteet, voitto/tappio) |
| **Harva palkinto** | Palkkio on nolla melkein kaikkialla; saat sen vasta pitkän oikean sarjan jälkeen |
| **Uutuus/yllätys** | Kuinka uusi tai odottamaton tila (tai siirtymä) on - asia, jonka uteliaisuus palkitsee |
| **Lukuperusteinen bonus** | Uutuus ≈ `1/sqrt(visit count)` - klassinen tutkimusbonus |
| **ICM** | Intrinsic Curiosity Module: uutuus = eteenpäin suuntautuvan mallin ennustevirhe (opetetussa ominaisuustilassa) |
| **`beta`** | Uteliaisuusbonuksen paino; yleensä hehkutetaan kohti nollaa, jotta agentti lopulta hyödyntää |

## Yhden lauseen yhteenveto

> **Uteliaisuusbonus on agentin itselleen antama palkinto uutuudesta – se luo
> tiheän tutkimissignaalin, joka vetää agentin läpi harvan palkkion maailmojen,
> joita se ei muuten koskaan ratkaisisi, ja vaimenee sitten hienovaraisesti
> pois, kun kaikki on jo tuttua.**
