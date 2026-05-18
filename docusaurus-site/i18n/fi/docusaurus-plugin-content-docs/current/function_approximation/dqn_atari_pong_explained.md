# DQN Atari Pongissa 🏓

## Mikä on Atari Pong?

Pong on yksi vanhimmista koskaan tehdyistä videopeleistä – se on kuin digitaalinen pöytätennis! Kaksi
melat pomppaavat palloa edestakaisin. Voitat pisteen, jos vastustaja missaa pallon.
Peli päättyy, kun joku saavuttaa 21 pistettä.

Meidän versiossamme tekoäly ohjaa yhtä melaa. Vastustaja (tietokone) hallitsee toista.
Peli alkaa aina tuloksesta −21 (pahin mahdollinen). Hyvä agentti saavuttaa 0 tai +21.

---

## Miksi Pong on vaikea tekoälylle?

CartPolessa robotti näki numerot suoraan (tangon kulma, kärryn nopeus...).
Pongissa se näkee vain **raakoja pikseleitä** – tuhansia pieniä värillisiä pisteitä näytöllä!

```
CartPole input: [0.02, −0.14, 0.01, −0.23]   ← 4 numbers, easy!
Pong input:     [pixel grid: 210×160×3]        ← 100,800 numbers, MUCH harder!
```

Robotin on selvitettävä pikseleistä:
- Missä mela on?
- Missä pallo on?
- Liikkuuko pallo vasemmalle vai oikealle?
- Kuinka nopeasti?

Ihmiset tekevät tämän automaattisesti (meillä on hämmästyttävä visio!). Tekoälylle tämä on valtava haaste.

---

## Seeing Motion: Kehyksen pinoaminen 🎬

Yksi kehys (kuvakaappaus) ei kerro, liikkuuko pallo vasemmalle vai oikealle. Tarvitset
nähdäksesi USEITA kehyksiä liikkeen ymmärtämiseksi – aivan kuten käännettävä kirja toimii vain silloin, kun
selaat monia sivuja.

**Kehysten pinoaminen:** Syötä viimeiset 4 kehystä verkkoon samanaikaisesti.

```
Frame 1: ball at position 40
Frame 2: ball at position 43    → Stack these 4 frames → Network sees MOTION!
Frame 3: ball at position 46
Frame 4: ball at position 49
```

Verkko voi nyt päätellä: "pallo liikkuu oikealla nopeudella 3"

**Esimerkki tosielämästä:** Elokuvan katsominen vs. yhden kuvan katsominen. Auton yksi runko
rotu on vain sumea kuva. Katso 4 kehystä ja voit kertoa kumpi auto on nopeampi!

---

## Näkeminen CNN-verkon avulla 🔍

Pikselituloissa käytämme erityistä hermoverkkoa, jota kutsutaan **konvoluutioneuroverkoksi
(CNN)**. Sen sijaan, että CNN katsoisi kaikkia pikseleitä kerralla, se käyttää liukuikkunoita havaitsemiseen
kuvioita – kuten silmät skannaavat kuvaa.

```
Raw pixels (84×84×4 frames)
       ↓
Conv Layer 1 (8×8 filter, stride 4) → finds edges and shapes
       ↓
Conv Layer 2 (4×4 filter, stride 2) → finds objects (paddles, ball)
       ↓
Conv Layer 3 (3×3 filter, stride 1) → finds relationships
       ↓
Flatten → 512 neurons → Q-values (one per action)
```

**Esimerkki tosielämästä:** Kun etsit ystävääsi joukosta, aivosi huomaavat sen ensimmäisenä
muodot (henkilö), sitten piirteet (hiusväri), sitten yksityiskohdat (heidän kasvonsa). CNN:t toimivat
samalla tavalla – yksinkertaisista kuvioista monimutkaisiin!

---

## Esikäsittely: Maailman kutistuminen

Pong-kehykset ovat värillisiä 210×160 pikseliä. Se on liian iso! Esikäsittelemme jokaisen kehyksen:

1. **Harmaasävy** — värillä ei ole Pongille väliä (pallo on joka tapauksessa aina valkoinen)
2. **Muuta kokoa 84 × 84** — pienempi = nopeampi harjoittelu, mutta silti tarpeeksi selkeä nähdäksesi
3. **Normaloi arvoon [0,1]** — jaa pikseliarvot 255:llä, jotta ne ovat pieniä lukuja

**Tosielämän esimerkki:** Kuten 50 %:n valokopion tekeminen. Tärkeät yksityiskohdat
(pallo, melat) ovat edelleen näkyvissä, vain pienempiä. Kopiokone ei välitä
myös värejä!

---

## Palkinnon leikkaaminen: Kaikkien pelien tasapuolinen kohtelu ✂️

Pongissa saat +1 maalinteosta ja −1 maalinteosta.
Joissakin muissa Atari-peleissä pisteet voivat olla tuhansia!

**Leikkaamme palkinnot** arvoon [−1, +1], joten verkosto ei välitä palkintojen suuruudesta.
Tämä sama koodi voi harjoitella MILLOIN Atari-pelissä ilman palkintoasteikkojen viritystä.

---

## Kuinka kauan koulutus kestää?

| Koulutuksen kesto | Mitä agentti oppii |
|---|---|
| 100 000 askelta | Enimmäkseen satunnaista, tuskin reagoi |
| 1M askelta | Alkaa välillä liikkua kohti palloa |
| 5M askelta | Palauttaa muutaman laukauksen |
| 10M askelta | Kilpailukykyinen peli, saattaa voittaa jonkin verran |
| 20M+ askelta | Voittaa usein tietokonevastustajan |

Demomme kestää **300 000 askelta** – tarpeeksi nähdäksesi koulutusarkkitehtuurin toimivan ja
tarkkaile varhaista oppimista, mutta ei tarpeeksi hallitsemaan peliä.

**Tosielämän esimerkki:** Pianonsoiton oppiminen kestää kuukausia. 10 minuutin harjoitus
osoittaa, että teet sen oikein, mutta älä odota vielä konsertteja!

---

## Mitä koodimme löysi

300 000 askeleen jälkeen Pongissa:
- Agentti aloittaa pisteillä noin −20 (hädin tuskin palauttaa pallon)
- Loppuun mennessä se tyypillisesti paranee noin -15:een -10:een
- Oppimiskäyrä osoittaa asteittaista paranemista satunnaisesta pelaamisesta

Nähdäksesi todellisen kilpailukykyisen Pong-suorituskyvyn, sinun on suoritettava yli 10 miljoonaa askelta GPU:lla.
Toteutus on täydellinen ja oikea - se tarvitsee vain lisää aikaa!

---

## Avainsanasto

| sana | Merkitys |
|------|---------|
| **CNN** | Konvoluutioneuroverkko — erikoistunut kuvasisääntuloihin |
| **Kehysten pinoaminen** | Useiden peräkkäisten kuvien syöttäminen liikkeen tallentamiseksi |
| **Esikäsittely** | Raakakehysten muuntaminen (harmaasävy, koon muuttaminen, normalisointi) ennen syöttämistä verkkoon |
| **Palkintojen leikkaus** | Palkintojen rajoittaminen [−1, +1]:iin, jotta se voi toimia eri peleissä |
| **ALE** | Arcade Learning Environment — kirjasto, joka pyörittää Atari-pelejä |

---

## Historiallinen saavutus

Kun DeepMind julkaisi DQN:n vuonna 2015, maailma hämmästyi. YKSI algoritmi,
SAMALLA arkkitehtuurilla oppinut pelaamaan 49 erilaista Atari-peliä – monia
yli-inhimillinen taso – vain raakapikseleistä ja pisteistä!

Ennen DQN:ää ihmiset ajattelivat, että sinun oli koodattava käsin jokaisen pelin strategia.
DQN osoitti, että yleisoppija pystyi keksimään kaiken itse.
Se oli historiallinen hetki tekoälyssä!
