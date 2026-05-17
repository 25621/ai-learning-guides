# Kohdeverkosto: Napakymppien vakauttaminen 🎯

## Liikkuva maalitolppa -ongelma

Kuvittele, että yrität lyödä napakymppiä jousella ja nuolella. Ammu, katso missä
nuolesi laskeutui ja säädä tavoite seuraavaa kertaa varten. Yksinkertaista, eikö?

Kuvittele nyt napakymppi MOVES joka kerta kun ammut! Jokaista nuolta ammut hieman
muuttaa sitä, missä häränsilmä on seuraavaa laukausta varten. Et koskaan paranisi - olisit
jahtaamaan kohdetta, joka aina pakenee.

Juuri tämä on DQN:n ongelma ilman kohdeverkkoa!

---

## Miksi Q-Targets liikkuu jatkuvasti

DQN:ssä kunkin päivityksen tavoite on:
> tavoite = palkkio + γ × max(Q(seuraava_tila))

Tässä **γ (gamma)** on **alennustekijä** – luku välillä 0 ja 1 (yleensä 0,99)
joka hallitsee sitä, kuinka paljon agentti välittää *tulevaisuuden* palkkioista verrattuna *välittömiin* palkkioihin.

**Tosielämän esimerkki:** Kuvittele, että joku tarjoaa sinulle evästeen nyt tai kaksi evästettä huomenna.
Jos todella haluat keksejä nyt, γ-arvosi on alhainen (alennat tulevaisuutta voimakkaasti). Jos olet
kärsivällinen ja iloinen odottamaan, γ-arvosi on korkea (tulevaisuuden palkkioilla on melkein yhtä paljon merkitystä kuin nyt).
RL:ssä γ = 0,99 tarkoittaa, että seuraava askel on 99 % palkkiosta tällä hetkellä.

Oikean puolen Q-arvot tulevat... samasta verkostosta, jota harjoittelemme!

Joten joka kerta kun päivitämme verkkoa (parantaaksemme Q-arvoja), muutamme myös
tavoitteita. Se on palautesilmukka:

1. Päivitä verkko → Q-arvot muuttuvat
2. Q-arvot muuttuvat → tavoitteet muuttuvat
3. Tavoitteet muuttuvat → päivitä verkko eri tavalla
4. Toista ikuisesti – epävakaa!

**Tosielämän esimerkki:** Yrität punnita itsesi vaa'alla, joka muuttaa lukemiaan
aina kun astut sen päälle. Et koskaan tiedä todellista painoasi!

---

## Ratkaisu: Jäädytä Bullseye! ❄️

**Kohdeverkko** on kopio Q-pääverkosta, joka jäätyy paikoilleen.

- **Verkkoverkko** (`qnet`): Päivitetty jokainen harjoitusvaihe – oppii nopeasti
- **Kohdeverkko** (`target_net`): Jäädytetty kopio — päivitetään vain 100 askeleen välein

Käytämme FROZEN-tavoitetta tavoitteiden laskemiseen:
> tavoite = palkkio + γ × max(Q_TARGET(seuraava_tila))

Tavoite pysyy paikallaan 100 askelta! Tämä antaa verkkoverkostolle vakaan tavoitteen
tavoitella. Sitten kopioimme online-painot kohteeseen, jäähdytämme uudelleen ja toistamme.

**Tosielämän esimerkki:** Ajattele oppilasta ja opettajaa. Opettaja antaa läksyt
(kohde). Opiskelija oppii ja kehittyy. 100 oppitunnin jälkeen opettaja PÄIVITYS
kotitehtävästä tulee vaikeampaa. Opettaja ei muutu joka minuutti – se
olisi liian kaoottista!

---

## Täydellinen DQN-resepti 🍕

Täydellinen DQN-algoritmi (kokemuksen toisto + kohdeverkko) on:

```
1. Initialize online network Q and target network Q_target (same weights)
2. Create replay buffer (memory box)

Every environment step:
  a. Pick action using ε-greedy with Q
  b. Store (state, action, reward, next_state) in buffer

Every 4 steps:
  c. Sample random mini-batch from buffer
  d. Compute targets using Q_TARGET (frozen!)
  e. Update Q to minimize loss

Every 100 steps:
  f. Copy Q weights → Q_TARGET (sync target)
```

Tämä on tarkka algoritmi DeepMind DQN -paperista (2015)!

---

## Mitä vertailu osoittaa

Kun juokset `dqn_target_network.py`, näet:

**Ilman kohdeverkkoa (vain DQN + toisto):**
- Harjoittelu saattaa olla "okei", mutta säännöllisin romahduksin
- Q-arvot voivat poiketa (räjähtää tai värähtää)
– Oppiminen on vähemmän ennakoitavissa

**Täysi DQN (toisto + kohdeverkko):**
- Johdonmukaisempaa ylöspäin suuntautuvaa oppimista
- Q-arvot pysyvät kohtuullisella alueella
- Nopeampi konvergenssi ratkaistuun kynnykseen (195+ CartPolessa)

---

## "Tappava kolmikko" ☠️

Vahvistusoppimisessa kolmen asian yhdistäminen luo epävakautta:

1. **Funktion approksimaatio** (hermoverkko taulukon sijaan) ← käytämme tätä
2. **Bootstrapping** (käyttäen Q-arvoja Q-arvojen arvioimiseen) ← käytämme tätä
3. **Käytännön ulkopuolinen oppiminen** (Q-learning käyttää maksimimäärää, ei varsinaista käytäntöä) ← käytämme tätä

Kaikki kolme yhdessä = "tappava kolmikko". DQN kesyttää tämän seuraavasti:
- Kokemustoisto → katkaisee korrelaatiot
- Kohdeverkko → katkaisee palautesilmukan

Se ei ratkaise ongelmaa täysin, mutta tekee siitä hallittavan!

---

## Avainsanasto

| sana | Merkitys |
|------|---------|
| **Kohdeverkosto** | Jäädytetty kopio Q-verkosta, jota käytetään vain kohteiden laskemiseen |
| **Verkkoverkko** | Q-verkostoa koulutetaan aktiivisesti |
| **Sync** | Online-verkon painojen kopioiminen kohdeverkkoon |
| **Palautesilmukka** | Kun järjestelmän lähtö syöttää takaisin tulon muuttamiseksi (voi aiheuttaa epävakautta) |
| **Tappava kolmikko** | Funktion approksimaatio + bootstrapping + off-policy yhdistelmä, joka aiheuttaa epävakautta |

---

## Tosimaailman vaikutus

Vuonna 2015 DeepMind julkaisi DQN-paperinsa, jossa esitettiin tekoäly, joka voisi pelata 49 Ataria.
pelejä yli-inhimillisellä tasolla – käyttämällä VAIN näitä kahta temppua (toisto + kohdeverkko).

Ennen tätä ihmiset ajattelivat, että et voi kouluttaa neuroverkkoja RL:n kanssa, koska
epävakautta. DeepMind osoitti heidän olevan väärässä, ja se aloitti syvän RL-vallankumouksen!

Seuraavaksi käytämme tätä täydellistä DQN-reseptiä Atari Pongissa – todellisessa videopelissä raakapikseleillä
syötteenä!
