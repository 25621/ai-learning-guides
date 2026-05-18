# SARSA Cliff Walkingiin 🏔️

## Mikä se on?

Kuvittele **erittäin pitkä käytävä**, jonka toisella reunalla on **kauhea kallio**. Jos putoat pois
cliff, sinun on palattava aina takaisin alkuun! Tavoitteesi on kävellä yhdestä päästä päähän
muut mahdollisimman nopeasti putoamatta.

**SARSA** on robotti, joka oppii kävelemään tällä käytävällä harjoittelemalla. Se oppii ottamaan a
*turvallinen polku*, joka välttää kallion – vaikka se olisikin vähän pidempi – koska se tietää, että se saattaa
liukastu vahingossa lähelle reunaa tutkiessasi!

---

## Suuri idea: oppia siitä, mitä todella teet

SARSA tarkoittaa: **S**tila → **A**toiminta → **R**palkinto → **S**seuraava tila → **A**toiminta

Nämä ovat viisi tietoa, joita SARSA käyttää oppiessaan:

1. **S** — Missä olen juuri nyt? (nykyinen tila)
2. **A** — Mitä tein?
3. **R** — Minkä palkinnon sain?
4. **S** — Mihin päädyin?
5. **A** — Mihin toimiin *itse asiassa aion ryhtyä seuraavaksi*?

Viimeinen "A" tekee SARSAsta erityisen! Se päivittyy käyttämällä toimintoja, jotka se todella suorittaa
seuraava* (vaikka se olisi satunnainen tutkiva liike), ei ole täydellinen ihanteellinen toiminta.

**Tosielämän esimerkki:** Ajattele pyörällä ajamisen oppimista. Jos tiedät, horjut joskus
satunnaisesti (etsintä), pysyt hieman kauempana pysäköityistä autoista - koska tiedät omasi
horjuva itse saattaa kääntyä! SARSA tekee tämän: se oppii turvallisen polun, koska se ottaa huomioon
omia satunnaisia ​​virheitään.

---

## Cliff-kävelykartta

```
[ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]
[ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]
[ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]
[S][C][C][C][C][C][C][C][C][C][C][G]
   ← ← ← ← CLIFF ← ← ← ← ←
```

- **S** = aloitus (alhaalla vasemmalla)
- **G** = tavoite (alhaalla oikealla)
- **C** = Cliff — astu tänne = -100 palkinto, aloita uudelleen!
- Joka toinen askel = -1 palkinto

---

## Mitä koodimme löysi

Kun olet harjoitellut SARSAa 500 jaksoa:

| Tulos | Arvo |
|--------|-------|
| Viimeinen 50 jakson keskimääräinen palkinto | **-21.6** |
| Optimaalinen (riskillinen) polkupalkkio | -13 |

SARSA:n opittu käytäntö kulkee **ruudukon huipulla** – turvallinen kiertotie! Se maksaa muutaman
ylimääräisiä askelia (-21 sijaan -13), mutta tuskin koskaan putoa kalliolta harjoituksen aikana.

---

## Esimerkkejä tosielämästä

- **Sairaanhoitaja antaa lääkkeitä**: noudattaa todistettua turvallista protokollaa (turvallista polkua), vaikka
  hieman nopeampi menetelmä on olemassa, koska pienet virheet (tutkiminen) voivat olla vaarallisia.
- **Lentolentäjät**: Noudata tiukkoja tarkistuslistoja (turvallisia polkuja), vaikka pikakuvakkeet saattavat tuntua
  nopeammin, mikä selittää inhimillisen erehdyksen.
- **Kokkaamaan oppiminen**: Aloita hyvin testatuilla resepteillä (turvallisia), ei vaarallisilla pikanäppäimillä.

---

## Avainsanat muistaa

- **On-policy**: oppii käytännöstä, jota se todella käyttää (mukaan lukien sen satunnaiset virheet)
- **SARSA-päivitys**: Käyttää *todellista* seuraavaa toimintoa, ei teoreettisesti parasta
- **Turvallinen polku**: Pidempi polku, joka välttää vaaran ja ottaa huomioon etsintävirheet
- **TD (Temporal Difference) -ohjaus**: Päivitetään arvot jokaisen vaiheen jälkeen (ei odota koko jaksoa)

Suuri idea: **SARSA on rehellinen – se oppii siitä, mitä se todella tekee, ei siitä, mitä se haluaa
se tekisi. Tämä tekee siitä varovaisen ja turvallisen lähellä vaaraa!**
