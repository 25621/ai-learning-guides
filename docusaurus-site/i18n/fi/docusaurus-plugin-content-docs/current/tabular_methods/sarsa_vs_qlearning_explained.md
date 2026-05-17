# SARSA vs Q-Learning: Turvalliset vs. optimaaliset polut 🐢 vs 🐇

## Mikä se on?

Kahden robotin on molempien käveltävä **kallioreunaa** pitkin päästäkseen maaliin. Molemmat robotit
ovat edelleen *oppimassa* ja tekevät joskus satunnaisia liikkeitä (hups!).

- 🐢 **SARSA-robotti**: "Tiedän, että heilun joskus... joten kävelen kauas kalliolta
  olla turvassa, vaikka se vie kauemmin."
- 🐇 **Q-Learning robot**: "Lyhin polku halaa kalliota - mennään! (Putoo
  joskus oppiessaan, mutta lopulta oppii parhaan reitin.)"

Molemmat robotit ovat älykkäitä, mutta tekevät **erilaisen kompromissin**: turvallinen, mutta hitaampi vs.
optimaalinen-mutta-riskialtista-oppimisen aikana.

---

## Keskeinen ero: mitä "seuraavaa toimintoa" käytät?

Kun päivität pisteitä jokaisen vaiheen jälkeen, molemmat algoritmit kysyvät:
> "Mikä on *seuraavan tilan* arvo?"

| Algoritmi | Käyttää seuraavaa toimintoa... | Käytännössä? |
|-----------|------------------------|------------|
| **SARSA** | ...jotka otan *itse asiassa* (ehkä satunnaisesti!) | Kyllä |
| **Q-Learning** | ...se on *teoreettisesti paras* (aina ahne) | Ei |

**Tosielämän esimerkki:** Kaksi lasta opettelee pyöräilemään.

- **SARSA-lapsi**: Pysyy lähellä ruohoa, koska *he tietävät* heiluvansa joskus satunnaisesti.
  He suunnittelevat todellista horjuvaa itseään.
- **Q-Learning Kid**: Harjoittelee keskellä polkua, koska he kuvittelevat täydellisen
  tuleva itse, joka ei koskaan horju. He kaatuvat joskus nyt, mutta oppivat parhaan tien nopeammin.

Molemmat lapset oppivat lopulta – mutta harjoituksen aikana SARSA-lapsi kaatuu vähemmän!

---

## Mitä koodimme löysi

Molemmat algoritmit suoritettiin **500 jaksoa** Cliff Walkingissa ε = 0,1 (ε = epsilon, "EP-sih-lon"; tässä se tarkoittaa 10 %:n todennäköisyyttä tehdä satunnainen liike):

| Metrinen | SARSA | Q-oppiminen |
|--------|-------|------------|
| Keskimääräinen palkinto harjoituksen aikana (viimeiset 50 ep) | **-19.7** | **-51.0** |
| Ahne arviointi (ei tutkimista) | -17 | **-13** |

- **Harjoittelun aikana**: SARSA saa **paljon parempia palkintoja**, koska se välttää kallion
  (laskee omat satunnaiset liikkeensä)
- **Harjoittelun jälkeen** (puhdas ahne): Q-Learning löytää **lyhyemmän optimaalisen polun** (-13)!

Kun ε kutistuu kohti nollaa, molemmat algoritmit konvergoivat samaan optimaaliseen käytäntöön.

---

## Visuaalinen yhteenveto

```
SARSA path (during training):     Q-Learning path (greedy, after training):
[ ][→][→][→][→][→][→][→][→][→][→][↓]   [ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][↓]
[↑][→][→][→][→][→][→][→][→][→][→][↓]   [ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][↓]
[ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][↓]   [ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][↓]
[S][C][C][C][C][C][C][C][C][C][C][G]   [S][→][→][→][→][→][→][→][→][→][→][G]
     (safe detour, top rows)                 (optimal, hugs cliff)
```

---

## Esimerkkejä tosielämästä

- **Uusi kirurgi vs. kokenut kirurgi**: Uusi kirurgi (SARSA) pysyy kaukana riskialtis
  tekniikoita oppimisen aikana. Kokenut kirurgi (Q-Learning ahne) käyttää tehokkainta
  tekniikkaa sen hallitsemisen jälkeen.
- **Kaupunkiajossa vs. moottoritiereitti**: SARSA-tyyppinen suunnittelu vie turvallisempia asuinkatuja;
  Q-Learning löytää optimaalisen mutta kapean tien.
- **Opiskelija opiskelee**: SARSA-opiskelija pitää kiinni hyvin ymmärretyistä aiheista harjoituksen aikana.
  Q-Learning-opiskelija pyrkii vaikeimpiin ongelmiin (epäonnistuu enemmän), mutta oppii optimaalisen strategian.

---

## Avainsanat muistaa

- **On-policy** (SARSA): oppii mitä *todellisuudessa teet*, mukaan lukien satunnainen tutkiminen
- **Käytännöstä riippumaton (off-policy)** (Q-Learning): oppii *parhasta mahdollisesta* käyttäytymisestä erikseen
  mitä todella teet
- **Turvallinen polku**: Pidempi reitti, joka välttää vaaran, käytetään, kun etsintä aiheuttaa onnettomuuksia
- **Optimaalinen polku**: Lyhin/korkein palkkio reitti, joka löytyy, kun etsintä ei tapahdu
- **Etsinnän ja hyödyntämisen kompromissi**: Tasapaino uusien asioiden kokeilemisen ja käytön välillä
  mitä tiedät

Suuri idea: **SARSA on turvallisempi koulutuksen aikana (on-policy), Q-Learning löytää optimaalisen
polku nopeammin (käytännöstä riippumaton). Kumpi on parempi, riippuu siitä, onko kalliolta putoamisella merkitystä!**
