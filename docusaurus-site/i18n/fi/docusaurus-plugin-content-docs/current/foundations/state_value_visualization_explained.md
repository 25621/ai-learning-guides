# Tila-arvofunktiot 🗺️

## Mikä on "tila"?

Ajattele lautapelin pelaamista. Seisot milloin tahansa jollakin
laudan ruudulla. Tuo ruutu on sinun **tilasi** – se on missä olet
juuri nyt.

4 × 4 -ruudukkopelissämme on 16 ruutua (tilaa). Jokainen ruutu on paikka, jossa
agentti voi seisoa.

---

## Mikä on "arvo"?

Tässä on maaginen kysymys: **"Jos seison tässä ruudussa juuri nyt,
kuinka paljon palkintoja voin odottaa kerääväni ennen pelin päättymistä?"**

Tuo vastaus on tuon tilan **arvo**!

Ruutu, jonka arvo on **korkea**, tarkoittaa: "Tämä on loistava paikka – saavutan
täältä luultavasti tavoitteen!"

Ruutu, jonka arvo on **pieni**, tarkoittaa: "Ai, tästä eteenpäin asiat yleensä menevät
huonosti."

**Esimerkki tosielämästä:** Kuvittele, että pelaat piilosta. Jos olet piilossa
suuren puun takana (hyvä paikka), voittomahdollisuutesi on korkea – se on
arvokas tila! Jos piiloudut tyhjän huoneen keskelle, tulet
luultavasti löydetyksi – se on vähäarvoinen tila.

---

## Ruudukkomaailmamme (Grid-world)

Tässä on käyttämämme pelilauta:

```
S  .  .  .      S = Aloitus (Start)
.  H  .  H      H = Reikä (Hole, palkinto -1, peli päättyy)
.  .  .  H      G = Maali (Goal, palkinto +1, peli päättyy)
H  .  .  G      . = Tyhjä turvallinen ruutu
```

- Jos saavutat **G** (tavoite): saat **+1 pisteen** 🎉
- Jos astut **H** (reikä) päälle: saat **-1 pisteen** 😢
- Muut vaiheet: **0 pistettä**

Käytimme arvoa $\gamma$ (gamma) = 0,99, mikä tarkoittaa, että tulevilla palkkioilla on melkein yhtä paljon
merkitystä kuin välittömillä palkkioilla. (Huominen karkki on melkein yhtä hyvä kuin tänään!)

---

## Kaksi erilaista suunnitelmaa (käytännöt)

Testasimme kahta käytäntöä ja laskimme jokaisen ruudun arvon kummallekin:

### Käytäntö 1: Tasaisen satunnainen
Liiku satunnaisesti ylös, alas, vasemmalle tai oikealle yhtä suurella todennäköisyydellä.

```
Values (Uniform Random Policy):
-0.912  -0.932  -0.912  -0.942
-0.929   (H)   -0.898   (H)
-0.901  -0.801  -0.696   (H)
 (H)   -0.630  -0.104   (G)
```

Melkein kaikkialla arvot ovat **negatiivisia** – satunnainen käytäntö putoaa reikiin niin
usein, että missä tahansa oleminen on aika huono asia!

---

### Käytäntö 2: Maalia kohti painotettu
Liikkuu mieluummin oikealle ja alas (kohti maalia), mutta valitsee silti joskus muita
suuntia.

```
Values (Biased-Toward-Goal Policy):
-0.838  -0.895  -0.814  -0.961
-0.798   (H)   -0.665   (H)
-0.595  -0.143  -0.213   (H)
 (H)    0.254   0.673   (G)
```

Nyt **tavoitteen** lähellä olevilla ruuduilla on **positiiviset arvot** (0,254 ja 0,673)!
Älykäs käytäntö tekee näistä ruuduista hyviä paikkoja.

---

## Mitä värit kuvassamme tarkoittavat

Visualisoinnissamme:
- **Vihreät ruudut** = arvokas (hyviä paikkoja olla)
- **Punaiset ruudut** = pieni arvo (vältä näitä!)
- **Keltaiset ruudut** = jotain siltä väliltä

Voit nähdä **gradientin** — arvot muuttuvat vihreämmiksi, kun siirryt kohti tavoitetta
ja punaisemmiksi reikien lähellä.

---

## Miksi välitämme arvoista?

Arvot ovat vahvistusoppimisen *perusta*! Kun tietää arvon
jokaisessa tilassa, voi tehdä järkeviä päätöksiä:

> "Olen ruudussa A. Voin siirtyä ruutuun B (arvo = 0,5) tai ruutuun C (arvo = -0,3).
> Menen B:hen – sillä on suurempi arvo!"

Juuri näin monet RL-algoritmit (kuten Q-learning) oppivat parantamaan
päätöksiä kertomatta sääntöjä.

**Tosielämän esimerkki:** Kuvittele, että valitset jonon
ruokakaupassa. Jokainen jono on "tila". Sen tilan arvo on se, kuinka nopeasti
pääset kassalta. Katsot jonoja (tarkkailet tiloja) ja valitset
sen, jolla on suurin arvo (lyhyin odotus + vähiten ostoksia).

---

## Kuinka laskemme arvot

Käytimme **iteratiivista käytännön arviointia**, joka toimii seuraavasti:

1. Aloita: arvaa, että kaikki arvot ovat 0.
2. Päivitä: laske jokaiselle ruudulle, mihin arvon *pitäisi* perustua sen mukaan,
   minne käytäntö vie sinut seuraavaksi.
3. Toista, kunnes arvot lakkaavat muuttumasta (konvergoivat).

Matemaattisesti: **V(s) = Σ_a π(a|s) × [R(s,a) + γ × V(seuraava_tila)]**

Yksinkertaisesti sanottuna: "Tämän ruudun arvo = keskimääräinen palkinto, jonka saan
juuri nyt + osa sen ruudun arvosta, jonne päädyn."

---

## Avainsanat muistaa

- **Tila**: Missä olet juuri nyt (yksi ruutu taululla)
- **Arvo V(s)**: Odotettu kokonaispalkkio tilasta s alkaen
- **Käytäntö**: Suunnitelmasi siitä, mitä tehdä kussakin tilassa
- **Alennuskerroin γ**: Kuinka paljon välität tulevista palkkioista (0,99 = paljon!)
- **Käytännön arviointi**: Arvojen laskeminen jokaiselle tilalle tietyn käytännön mukaisesti

Suuri idea: **Jotkut paikat ovat parempia kuin toiset – ja arvofunktio
kertoo tarkalleen, kuinka hyvä kukin paikka on!**
