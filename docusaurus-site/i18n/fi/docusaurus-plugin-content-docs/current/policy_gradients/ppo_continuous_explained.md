# PPO jatkuvaan ohjaukseen: BipedalWalker Walkin tekeminen

## Diskreetti vs. jatkuvat toiminnot

Toistaiseksi jokaisessa ratkaisemassamme ympäristössä on **erillisiä** toimintoja:
- CartPole: paina VASEN tai paina OIKEA (2 vaihtoehtoa)
- LunarLander: ei ammu mitään / vasen / pää / oikea (4 vaihtoehtoa)

Mutta tosimaailman robotit tarvitsevat **jatkuvia** toimia:
- Humanoidirobotti: "kuinka kovaa niveltä työnnetään" (mikä tahansa arvo välillä -1 - +1)
- Auto: "tarkasti kuinka paljon ohjauspyörää pitää kääntää" (mikä tahansa kulma -30° - +30°)
- Käsivarsi: "kohdista täsmälleen 2,3 Newtonia tähän suuntaan"

**Tosielämän esimerkki:** Kirjoittaminen näppäimistöllä = diskreetti (paina A, B, C...).
Kirjoittaminen kynällä = jatkuva (siirrä kättä 2,3 cm oikealle, paina 40g voimaa...).

---

## Gaussin jatkuvan toiminnan politiikka

Jatkuville toimille kategoriallisen jakauman sijaan (valitse N luokasta),
käytämme **normaalia (Gaussin) jakaumaa**:

```
Action ~ Normal(μ, σ)
```

Missä:
- **μ (mu, keskiarvo)**: Jakauman keskus – toiminta-arvo, johon verkosto pyrkii
- **σ (sigma, keskihajonta)**: Hajautus – kuinka paljon satunnaisuutta/tutkimusta lisätään

```
        Probability
             │
        0.4 ─┤      ██████
             │    ████████████
        0.2 ─┤  ██████████████████
             │████████████████████████
             └──────────────────────── Action value
           -1  -0.5   0   0.5   1
                      ↑
                   mean μ
```

**Tosielämän esimerkki:** Taitava jousiampuja tähtää kohteen keskelle (μ).
Kaikki nuolet eivät laske täsmälleen samaan paikkaan – niissä on jonkin verran leviämistä (σ).
Kun he harjoittelevat, ne tarkentuvat (σ pienenee) pysyen samalla keskittyneinä napakymppiin.

---

## Gaussin näyttelijä-kriitikkoverkostomme

```
State (24 numbers) → [256 neurons] → [256 neurons] →
    ├── Actor: 4 mean values  (μ₁, μ₂, μ₃, μ₄)
    │          + 4 log_std params (shared across all states!)
    └── Critic: 1 value (V(s))
```

The `log_std` (**keskihajonnan** logaritmi – leviämisen tai epävarmuuden mitta)
on **opetettava parametri** — ei tilariippuvainen.
Tämä tekee sen yksinkertaiseksi ja antaa samalla tutkimisen muuttua harjoittelun aikana.

**Miksi log_std std:n sijaan?** Keskihajonnan on oltava positiivinen. Käyttämällä `log_std` sallii
verkko tulostaa minkä tahansa reaaliluvun (positiivisen tai negatiivisen), niin käytämme
`exp(log_std)` — eksponentiaalinen funktio, joka on logaritmin käänteisfunktio — to
palauta taattu positiivinen std. Tämä estää std:n muuttumasta negatiiviseksi tai nollaksi.

---

## Jatkuvien toimintojen lokin todennäköisyyden laskeminen

Erillisissä toimissa: `log_prob = log(P(action=LEFT))`

Jatkuville toimille **normaalijakauma** kuvaa tasaista kellonmuotoista käyrää
keskiarvon ympärillä. Yhden tarkan arvon todennäköisyys on nolla jatkuvassa matematiikassa, joten käytämme sitä
käyrän korkeus kyseisessä arvossa, jota kutsutaan **pdf:ksi** (todennäköisyystiheysfunktio):
```
log_prob = Σᵢ log[Normal(μᵢ, σᵢ).pdf(aᵢ)]
```

`log` tarkoittaa luonnollista logaritmia. Se muuttaa pienistä tiheysarvoista vakaita lukuja, jotka ovat
hermoverkkojen on helpompi optimoida. Summaamme kaikki toiminnan ulottuvuudet (4 for
BipedalWalker), koska koko toiminta on yksi 4-numeroinen vektori.

**Tosielämän esimerkki:** Mikä on todennäköisyys saada huomenna tarkalleen 5,732...°C?
Jatkuvassa säässä katsot normaalijakaumakäyrää ja katsot kuinka korkea se on
juuri tuossa vaiheessa. Todennäköisemmillä lämpötiloilla (lähellä keskiarvoa) on suurempi todennäköisyys.

---

## BipedalWalker: Kävelyhaaste

BipedalWalker-v3 on 2D-robotti, jonka on opittava kävelemään putoamatta:

```
          O (head)
         /│\
        / │ \
       /  │  \
      L   │   R   ← two legs, each with a knee joint
     / \  │  / \
    ●   ● │ ●   ●  ← 4 motors (hip/knee for each leg)
```

**Tila-avaruus (24 numeroa):**
- Runko: kulma, kulmanopeus, vaakanopeus, pystynopeus (4 numeroa)
- Nivelet: 4 moottoria (2 lonkkaa, 2 polvea), joista kukin tarjoaa kulman ja nopeuden, plus 2 maakosketusanturia (yksi kumpaankin jalkaan) (10 numeroa)
- 10 LIDAR-anturia (etäisyyslukemat, jotka näkevät maan edessä) (10 numeroa)

**Toimintotila (4 jatkuvaa arvoa, kukin [-1, 1]):**
Toiminta-arvot ohjaavat **vääntömomenttia** (moottoreiden kohdistamaa pyörimisvoimaa) täsmälleen 4 nivelelle (toimia ei kohdisteta suoraan runkoon):
- Jalan 1 lonkan vääntömomentti, jalan 1 polven vääntömomentti, jalan 2 lonkan vääntömomentti, jalan 2 polven vääntömomentti

**Palkinnot:**
- +300 tavoitteen saavuttamisesta (oikea puoli)
- -100 kaatumisesta (kosketus maahan keholla)
- Pieni palkkio jokaisesta edistymisen askeleesta
- Pieni sakko jokaisesta moottorin käytöstä (palkitsee tehokkuus)

**Ratkaistu milloin:** Keskimääräinen palkinto > 300 yli 100 jaksosta

---

## Keskeinen ero diskreetistä PPO:sta

Kaikki on samaa PAITSI:

| | Diskreetti PPO | Jatkuva PPO |
|---|---|---|
| **käytäntö** | Kategorinen (logit) | Normaali(μ, σ) |
| **Näyte** | toiminta = esimerkki kohteesta 0,1,...,N} | toiminta = μ + σ × kohina |
| **log_prob** | log P(toiminta=k) | Σ log Normaali(μᵢ, σᵢ).pdf(aᵢ) |
| **Kiinnitin** | Ei tarvita | Kiinnitä toiminnot kohtaan [-1, 1] |

**Logitit** ovat raakoja, normalisoimattomia pisteitä erillisille toimille. Kategorinen politiikka kääntyy
ne todennäköisyyksiksi **softmax**-funktiolla, joka ottaa minkä tahansa joukon lukuja ja
puristaa ne kelvolliseen todennäköisyysjakaumaan (kaikki arvot positiivisia, summa 1).
Esimerkiksi logiteista [2.0, 1.0, 0.5] tulee todennäköisyyksiä [0.59, 0.24, 0.17]. Jatkuva PPO **ei** käytä softmaxia itse toimintoon,
koska toimintoa ei valita kiinteästä valikosta. Sen sijaan politiikka tuottaa keskiarvon
ja normaalijakauman keskihajonnan, sitten ottaa siitä näytteen reaaliarvoiset vääntömomentit.

**Clamp** tarkoittaa arvon pakottamista kelvolliselle alueelle. Koodi käyttää `action.clamp(-1, 1)` siis
ympäristö ei koskaan saa moottorikäskyä sallittujen rajojen ulkopuolella.

**Clip** PPO:ssa tarkoittaa jotain muuta: PPO leikkaa todennäköisyyssuhteen tappion sisällä,
kuten on selitetty [PPO-leikkausosio](./ppo_scratch_explained.md#the-clipping-trick).
Toimintakiinnitys suojaa ympäristöliittymää; PPO-leikkaus suojaa käytäntöpäivitystä.

---

## Kävely tyhjästä: mitä agentti oppii

**Varhainen harjoittelu (negatiiviset palkinnot):** Robotti pomppii satunnaisesti, putoaa välittömästi.
Jokainen jakso päättyy kaatumiseen sekunneissa.

**Puoliharjoittelu:** Robotti huomaa, että jalkojen liikkuminen vuorotellen kehittää eteenpäin.
Se alkaa tehdä pieniä, kiusallisia askelia – palkkiosta tulee vähemmän negatiivinen.

**Myöhäinen harjoittelu:** Tasainen, tehokas kävely **askel** syntyy. Kävely on toistuva liike
kuvio, kuten vuorotellen vasen ja oikea askel. Robotti mukautuu epätasaiseen maastoon dynaamisesti käyttämällä LIDAR-antureita mukauttamaan askeleitaan reaaliajassa.

**Tosielämän esimerkki:** Vauva oppii kävelemään:
1. Putoaa välittömästi (negatiivinen palkinto)
2. Ottaa yhden askeleen, kaatuu (hieman vähemmän negatiivinen)
3. Ottaa muutaman askeleen (pieni positiivinen palkinto)
4. Kävelee huoneen poikki (suuri positiivinen palkinto!)

---

## Miksi BipedalWalker tarvitsee PPO:n (ei REINFORCE)

- **BipedalWalker-jaksot** voivat olla jopa 1600 askelta (paljon pidempiä kuin CartPole!)
- **Palkinnot ovat harvat** — eteenpäin etenemispalkkiot ovat pieniä askelta kohti
- **REINFORCE tarvitsisi** tuhansia täydellisiä jaksoja saadakseen hyödyllisen signaalin

PPO:n n-vaiheiset päivitykset [GAE (yleinen etuarvio)](./ppo_scratch_explained.md#gae-smarter-advantage-estimates) anna robotin oppia keskeneräisistä jaksoista:
> "Vaikka kaaduin 50 askeleen jälkeen, ne askeleet osoittivat JOTKIN edistystä.
> Anna minun käyttää 50-vaiheista tuottoarviota sen sijaan, että odotan jakson päättymistä."

---

## Tulokset

500 päivityksen jälkeen (≈ 1 miljoona ympäristövaihetta):
- Robotti etenee näkyvästi satunnaisesta heiluttamisesta kohti jotakin eteenpäin suuntautuvaa liikettä
- Oppimiskäyrän jatkuva parantaminen
- Täysi konvergenssi palkitsemiseen > 300 vaatii enemmän harjoittelua (5-10M askelta)

Oppimiskäyrä näyttää jatkuvan ohjauksen ominaiskäyrän "S-käyrän":
1. Hidas alkukehitys (oppimisen vakaus)
2. Nopea parannus (kävelyhavainto)
3. Asteittainen tarkentaminen (kävelyoptimointi)

---

## Avaimet takeawayt

| käsite | Pelkkää englantia |
|---------|---------------|
| **Gaussin politiikka** | Sen sijaan, että valitsisit valikosta, heitä tikkaa arvoalueelle |
| **μ (keskiarvo)** | Mihin politiikka "tarkoittaa" |
| **σ (std)** | Kuinka paljon satunnaisuutta / tutkimista käytäntö käyttää |
| **log_std opittava parametri** | Gradienttipohjaisella optimoinnilla päivitetty maailmanlaajuinen etsintänopeus (gradientti *nousu* palkkiolla tai vastaava gradientti *lasku* PPO-häviössä) – aivan kuten mikä tahansa muu verkon paino. |
| **Jatkuva ohjaus** | Reaaliarvoisten lähtöjen (vääntömomentit, voimat, kulmat) hallinta |

---

## Mitä seuraavaksi?

PPO:ssa on monia **hyperparametrejä** — asetukset, jotka valitset ennen harjoittelun aloittamista (toisin kuin
*parametrit*, kuten verkon painot, jotka opitaan automaattisesti). Esimerkkejä ovat mm
`clip_eps`, oppimisnopeus, aikakausien lukumäärä ja eräkoko.

Kuinka herkkä PPO on näille valinnoille? `ppo_hyperparams.py` suorittaa kokeita
vaihtelemalla järjestelmällisesti kutakin hyperparametria ja näyttää vaikutuksen oppimisnopeuteen ja -vakauteen.
