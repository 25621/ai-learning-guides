# Q-Learning Agent for Frozen Lake 🧊

## Mikä se on?

Kuvittele jäätynyt lampi, jossa on liukasta jäätä. Siellä on **aloitusruutu** ja **maaliruutu**
jossa on joitain **reikiä** keskellä. Jos putoat kuoppaan, aloitat alusta!

Jää on liukasta, joten vaikka yrittäisit kävellä oikein, saatat sen sijaan liukua ylös tai alas.
**Q-Learning-agentti** on robotti, joka oppii – yrittämällä yhä uudelleen – miten päästä
aloitusruudusta maaliin putoamatta!

---

## Mitä Q-Learningin "Q" tarkoittaa?

**"Q"** tarkoittaa **"laatua"** - erityisesti *laatua* tietyn ottamisen
toimia tietyssä tilanteessa.

Ajattele sitä kuin ravintolan arvosanaa: "Kuinka hyvää (laatua) on tilata pizza TÄSTÄ
ravintola?" Q(s, a) kysyy: "Kuinka hyvä on toimia **a**, kun olen tilassa **s**?"

Korkea Q-arvo tarkoittaa: "Loistava valinta! Tämä toiminta johtaa paljon palkintoja."
Matala Q-arvo tarkoittaa: "Huono idea! Tämä toiminta johtaa yleensä ongelmiin."

**Tosielämän esimerkki:** Kuvittele, että olet lapsi, joka päättää syödä karkkia ennen illallista.
Q-arvo "syö karkkia nyt" saattaa olla korkea juuri nyt (se maistuu hyvältä!), mutta yleisesti ottaen matala
(äiti suuttuu, tunnet olosi kipeäksi myöhemmin). Q-learning oppii ottamaan huomioon tulevaisuuden
seuraukset - ei vain välitön tunne!

---

## Suuri idea: Maaginen pistetaulukko

Q-Learning rakentaa suuren taulukon nimeltä **Q-table**. Jokainen rivi on neliö jäällä,
ja jokainen sarake on toiminto (vasen, oikea, ylös, alas). Sisällä olevat numerot ovat **pisteitä**:
"Kuinka hyvä on tehdä tämä toimenpide tältä aukiolta?"

Aina kun robotti yrittää liikettä:
1. Se saa palautetta (putosiko se? saavuttiko tavoitteen?)
2. Se päivittää taulukon pistemäärän käyttämällä tätä kaavaa:

> **Uusi pistemäärä = vanha pistemäärä + oppimisaste × (mitä todella tapahtui − mitä odotin)**

Robotti pohjimmiltaan kysyy: "Oliko tämä liike parempi vai huonompi kuin luulin?"

**Tosielämän esimerkki:** Ajattele vauvaa, joka oppii kävelemään. Joka kerta kun he yrittävät astua ja kaatua,
he oppivat "se askel oli huono". Joka kerta kun he onnistuvat, he muistavat "se toimi!" jälkeen
monet yrittävät, he keksivät kuinka kävellä. Q-learning tekee saman asian, mutta pöydällä!

---

## Mikä tekee Q-Learningistä erityisen: se on politiikan ulkopuolella!

Tässä on jotain fiksua: kun Q-Learning päivittää taulukkoaan, se *olettaa aina, että se tekee
täydellinen liike ensi kerralla*, vaikka harjoituksen aikana se joskus tutkii satunnaisia liikkeitä.

Tämä tekee Q-Learningistä **poikkeuksen**: strategian, jonka se *oppii* (valitse aina tunnetuin
toiminta) on erillinen strategiasta, jota se *seuraa* harjoituksen aikana (joskus valitse satunnainen
tutkittava toimenpide). Konkreettisesti Q-taulukkopäivitys käyttää seuraavan *maksimi* Q-arvoa
tila - teoreettinen paras - vaikka robotin todellinen seuraava liike olisi satunnainen.

Yksinkertaisesti sanottuna: robotti saattaa satunnaisesti vaeltaa vasemmalle tutkimaan, mutta se oppii silti
laskee ikään kuin se tekisi *parhaan* toimenpiteen seuraavaksi. Tämä erottelu mahdollistaa Q-Learningin
lähentyä optimaaliseen strategiaan riippumatta siitä, kuinka paljon se tutkii.

---

## Mitä koodimme löysi

Harjoittelimme **50 000 jaksoa** liukkaalla 4 × 4 Frozen Lakella:

| Metrinen | Tulos |
|--------|--------|
| Ahne arvioinnin onnistumisprosentti | **73.1%** |
| Välitavoite (>70 %) | ✓ **LÄPISTETTY** |

Jää on erittäin liukasta, joten paraskaan politiikka ei voi voittaa 100% ajasta!

Opittu Q-taulukko näyttää agentin selvittäneen: mene alas ja oikealle välttäen reikiä.

---

## Esimerkkejä tosielämästä

- **Itse ajava auto**: Oppii, mitkä kaistat valita risteyksissä kokeiluajojen avulla.
- **Suositusjärjestelmät**: Oppiminen, mitä elokuvia suositellaan sen perusteella, pitivätkö käyttäjät
  aikaisemmat ehdotukset.
- **Videopelin tekoäly**: Hahmo, joka oppii navigoimaan sokkelossa kokeilemalla monia polkuja.

---

## Avainsanat muistaa

- **Q-taulukko**: Taulukko siitä, "kuinka hyvä kukin toiminto on kussakin tilassa"
- **Q(s, a)**: Toiminnon a pisteet tilassa s
- **Palkinto**: Mitä agentti saa toimenpiteen jälkeen (+1 tavoitteen saavuttamisesta, 0 muuten)
- **Poiskäytäntö**: Oppii optimaalisen strategian vaikka tutkii satunnaisesti
- **ε-ahne** (ε = epsilon): Suurimman osan ajasta tekevät tunnetuimman toiminnan; joskus tutkia satunnaisesti
- **Alennuskerroin γ** (γ = gamma): kuinka paljon tulevat palkkiot ovat arvokkaita (kuten rahan suosiminen nyt tai myöhemmin)

Suuri idea: **Q-Learning rakentaa "huijausarkin" jokaiseen tilanteeseen ja paranee jatkuvasti
sitä kunnes se tietää parhaan liikkeen kaikkialla.**
