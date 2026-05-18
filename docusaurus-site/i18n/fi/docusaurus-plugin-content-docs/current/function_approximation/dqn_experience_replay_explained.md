# Kokemustoisto: Opetetaan robotti muistamaan 🎒

## Ongelma: unohtaminen (ja hämmennys)

Muistatko kuinka naiivi DQN oli epävakaa? Suurin syy on **korreloitu oppiminen**.

Kun robotti pelaa peliä, se kokee asiat järjestyksessä:
> Vaihe 1 → Vaihe 2 → Vaihe 3 → Vaihe 4 → ...

Nämä vaiheet liittyvät toisiinsa! Jos robotti nojaa vasemmalle vaiheessa 10, myös vaihe 11 nojaa
vasemmalle. He eivät ole itsenäisiä - he ovat riippuvaisia ​​toisistaan.

Kun päivitämme verkon näiden korreloitujen vaiheiden avulla, se on kuin yrittäisi oppia
historiaa lukemalla samaa lukua uudestaan ​​ja uudestaan. Osaisit todella hyvin yhdessä luvussa
ja unohda kaikki muu!

**Esimerkki tosielämästä:** Kuvittele, että opiskelet koetta varten harjoittelemalla vain eilisen
kotitehtävät. Tulet hämmästyttäväksi juuri noissa ongelmissa, mutta testissä on erilaisia ​​kysymyksiä!
Sinun täytyy harjoitella yhdistelmää erilaisia ​​ongelmia.

---

## Ratkaisu: Muistilaatikko 📦

**Experience Replay** lisää robottiin suuren muistilaatikon (**toistopuskurin**).

Uusimmasta kokemuksesta oppimisen sijaan robotti:
1. **Tallentaa** jokaisen kokemuksen muistilaatikkoon: (tila, toiminta, palkinto, seuraava tila)
2. **Poimii satunnaisesti** kourallisen muistoja laatikosta
3. **Oppii tuosta satunnaisesta sekoituksesta** vain viimeisimmän vaiheen sijaan

```
Game Step 1 → [store in box]
Game Step 2 → [store in box]
Game Step 3 → [store in box]
...
Game Step 50 → [store in box] → pick 64 random memories → update network
Game Step 51 → [store in box] → pick 64 random memories → update network
```

**Esimerkki tosielämästä:** Ajattele valokuva-albumia. Et opi elämästäsi pelkästään
katsomassa tämän päivän kuvia. Voit selata myös VANHAJA valokuvia – sekoitus hyviä muistoja ja
hankalia hetkiä. Tämä auttaa sinua ymmärtämään malleja koko elämäsi aikana, ei vain tänään.

---

## Miksi satunnaisotos auttaa

Kun valitsemme muistoja satunnaisesti, rikkomme korrelaatiot. Robotti saattaa oppia:
- Muisto, jossa sauva oli täydellinen (500 askelta sitten)
- Muisto, jossa sauva oli kaatumassa (20 askelta sitten)
- Muisto, jossa kävi tuuri (vaiheesta 3)

Tämä satunnainen sekoitus tarkoittaa:
✅ Robotti oppii erilaisista tilanteista
✅ Jokainen muisti voidaan "toistaa" monta kertaa (tehokas kokemuksen käyttö)
✅ Verkko ei sovi liikaa viimeaikaisiin tapahtumiin

---

## Mini-eräoppiminen

Sen sijaan, että päivittäisimme YHDEN kokemuksen kerrallaan, päivitämme **64 kokemusta kerralla**
("mini-erä"). Tämä on tällainen:
- Vanha tapa: Lue yksi oppikortti, kysy itse
- Uusi tapa: Lue 64 erilaista muistikorttia ja testaa sitten itseäsi

Mini-erät tekevät oppimissignaalista paljon luotettavamman ja vähemmän meluisan.

---

## Lämmittelyjakso

Emme aloita oppimista heti! Toistopuskuri tarvitsee ensin muistoja.
Odotamme, kunnes laatikossa on vähintään **500 kokemusta** ennen harjoittelun alkamista.

**Tosielämän esimerkki:** Et yrittäisi valmistaa ateriaa ennen kuin olet kerännyt omasi
ainesosia. Lämmittelyaika on kuin ruokaostoksia ennen ruoanlaittoa!

---

## Mitä vertailu osoittaa

Kun juokset `dqn_experience_replay.py`, näet kaksi oppimiskäyrää:

| Naiivi DQN | DQN + toisto |
|-----------|-------------|
| Erittäin kuoppainen | Tasaisempi |
| Toistuvat kaatumiset (unohtaa kaiken) | Tasapainoisempi parannus |
| Korkea varianssi | Pienempi varianssi |

Toistoversio yleensä:
- Saavuttaa hyvät pisteet luotettavammin
- Ei putoa 500:sta 30:een yhtä usein
- Näyttää vakaamman oppimisen edistymisen

---

## Toistopuskuri koodissa

```
ReplayBuffer:
  - capacity: 10,000 memories (oldest are forgotten when full)
  - push(state, action, reward, next_state, done)
  - sample(batch_size=64) → random batch
```

Ajattele sitä kuin muistikirjaa, jossa on 10 000 riviä. Kun se on täynnä, pyyhit vanhimman
rivi ja kirjoita uusin. Opiskelet aina satunnaiselta sivulta!

---

## Avainsanasto

| sana | Merkitys |
|------|---------|
| **Koe toisto** | Tallenna ja käytä satunnaisesti aiempia kokemuksia koulutukseen |
| **Toistopuskuri** | Muistilaatikko, joka tallentaa menneet (tila, toiminta, palkinto, seuraava_tila) monikot |
| **Vastaavia päivityksiä** | Kun harjoitustiedot riippuvat itsestään (huonosta oppimiselle!) |
| **Mini-erä** | Pieni satunnainen näyte muisteista, joita käytetään yhdessä päivitysvaiheessa |
| **Sisustus** | Peräkkäisten kokemusten välisten yhteyksien katkaiseminen |

---

## Mitä vielä puuttuu?

Jopa toistopuskurilla on toinen ongelma: **liikkuva kohde**.

Joka kerta kun päivitämme verkkoa, Q-arvot muuttuvat. Mutta nuo päivitetyt Q-arvot ovat
Käytetään MYÖS SEURAAVAN päivityksen tavoitteen laskemiseen. Se on hämmennyksen ympyrä!

Tämän ratkaisee **Target Network** - jäädytetty kopio verkosta, joka vain
päivittyy 100 askeleen välein. Tämä saa "bullseye" pysymään paikallaan jonkin aikaa, joten
robotti voi tähdätä siihen luotettavasti!
