# Monikätinen rosvo -ongelma 🎰

## Mikä se on?

Kuvittele, että olet syntymäpäiväjuhlissa ja siellä on **10 erilaista karkkipurkkia**.
Jokaisessa purkissa on karkkia sisällä, mutta joissakin purkeissa on *namia* karkkia ja joissakin purkeissa
*ei-niin-namia* karkkia.Et tiedä mikä purkki on paras – sinun täytyy kokeilla niitä!

Joka kerta kun kurkottelet purkkiin, saat karkkia.Työsi on:

> **Hanki mahdollisimman paljon herkullisia karkkia!**

Se on Multi-Armed Banditin ongelma!Karkkipurkkien sijaan tiedemiehet kutsuvat
ne "käsivarret" (kuten kädet kolikkopelissä).Jokainen käsi antaa sinulle palkinnon, mutta
palkinnot ovat joka kerta erilaisia.

---

## Suuri kysymys: Pitäisikö minun kokeilla uusia purkkeja vai pysyä suosikkini kanssa?

Tämä on vaikein osa!Oletetaan, että kokeilit purkkia nro 3 ja se oli melko hyvä.
Nyt sinulla on valinnanvaraa:

- **Hyödynnä**: Jatka purkin 3 valitsemista, koska tiedät jo sen olevan hyvä.
- **Tutki**: Kokeile uutta purkkia – ehkä Purkki #7 on vielä *parempi*!

Jos valitset vain ensimmäisen purkin, josta pidit, saatat jäädä paitsi supernamiin
purkki.Mutta jos *aina* kokeilet uusia purkkeja, et koskaan käytä jo oppimaasi!

**Tosielämän esimerkki:** Ajattele suosikkiravintolaasi.Tilaat aina
kananuggetit (hyödynnä!), mutta ehkä pizza on vielä parempi.Jos et koskaan
kokeile jotain uutta, et koskaan tiedä!

---

## Epsilon-ahne strategia {#the-epsilon-greedy-strategy}

Älykäs tapa ratkaista tämä on nimeltään **epsilon-ahne** (epsilon on vain
Kreikkalainen kirjain ε, sanottu kuten "ep-sih-lon"):

1. **Useimmiten (sanotaan 90 %)**: Valitse * mielestäsi* paras purkki.
2. **Joskus (sanotaan 10 %)**: Valitse *satunnainen* purkki tutkittavaksi!

10 % tutkimusmatkat auttavat sinua löytämään parempia purkkeja.90% hyväksikäytöstä
matkojen avulla voit käyttää jo oppimaasi.

---

## Mitä koodimme löysi

Testasimme 10 käsivartta (karamellipurkkeja) 200 eri lapsen kanssa, jokaisessa 1000 poimua:

| strategia | % ajasta parhaan purkin valitsemiseen |
|----------|----------------------------------|
| **Älä koskaan tutki (ε=0)** | 14,5% — jumissa aikaisin, ei koskaan löytänyt parasta! |
| **Tutki 1 % ajasta (ε=0,01)** | 37,6% - hiljalleen löytyi paras purkki |
| **Tutki 10 % ajasta (ε=0,10)** | **74,2 %** — oppi nopeasti, valittiin suurimman osan ajasta! |

**Oppitunti**: Pienellä tutkimisella pääsee pitkälle!

---

## Esimerkkejä tosielämästä

- **Netflixin suositukset**: Jos Netflix näyttää sinulle elokuvan, näytät todennäköisesti
  tykkää (hyödynnä) tai ehdota jotain uutta (tutkii)?
- **Lääkäri valitsee hoidon**: Käytä hoitoa, joka yleensä toimii (hyödynnä)
  tai kokeilla uutta, joka saattaa olla vielä parempi (tutkita)?
- **Mehiläinen etsimässä kukkia**: Pitäisikö sen vierailla kukkien luona, joilla se tietää olevan
  nektaria tai lentää uudelle pellolle?

---

## Avainsanat muistaa

- **Arm**: Yksi vaihtoehdoista (kuten karkkipurkki)
- **Palkinto**: Mitä saat, kun valitset käsivarren (kuten karkkia)
- **Hyödynnä**: Käytä sitä, minkä tiedät jo olevan hyvää
- **Tutki**: Kokeile jotain uutta saadaksesi lisätietoja
- **Epsilon (ε)**: Mahdollisuus tutkia hyväksikäytön sijaan

Suuri idea: **Sinun on tasapainotettava uusien asioiden kokeileminen ja sen, mitä tiedät!**
