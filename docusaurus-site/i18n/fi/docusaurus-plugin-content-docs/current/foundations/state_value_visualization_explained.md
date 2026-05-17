# Tila-arvofunktiot 🗺️

## Mikä on "valtio"?

Ajattele lautapelin pelaamista. Minä seisot milloin tahansa *yllä*
laudan neliö. Tuo neliö on sinun **tilasi** – se on missä olet
juuri nyt.

4 × 4 -ruudukkopelissämme on 16 ruutua (tilaa). Jokainen neliö on paikka
agentti voi seisoa.

---

## Mikä on "arvo"?

Tässä on maaginen kysymys: **"Jos seison tällä torilla juuri nyt,
kuinka paljon aarretta voin odottaa kerääväni ennen pelin päättymistä?"**

Tuo vastaus on tuon tilan **arvo**!

Neliö, jonka arvo on **korkea**, tarkoittaa: "Tämä on loistava paikka - luultavasti
kerää täältä paljon aarteita!"

Neliö, jonka arvo on **pieni**, tarkoittaa: "Ai, tästä eteenpäin asiat yleensä menevät
huonosti."

**Esimerkki tosielämästä:** Kuvittele, että pelaat piilosta. Jos olet piilossa
suuren puun takana (hyvä paikka) voittomahdollisuutesi on korkea – se on a
arvokas valtio! Jos piiloudut tyhjän huoneen keskelle, tulet siihen
luultavasti löydetään – se on vähäarvoinen tila.

---

## Grid-maailmamme

Tässä on käyttämämme pelilauta:

```
S  .  .  .      S = Start
.  H  .  H      H = Hole (reward -1, game ends)
.  .  .  H      G = Goal (reward +1, game ends)
H  .  .  G      . = Empty safe square
```

- Jos saavutat **G** (tavoite): saat **+1 pisteen** 🎉
- Jos astut **H** (Hole) päälle: saat **-1 pisteen** 😢
- Muut vaiheet: **0 pistettä**

Käytimme arvoa γ (gamma) = 0,99, mikä tarkoittaa, että tulevilla palkkioilla on melkein yhtä paljon
välittöminä palkintoina. (Huominen karkki on melkein yhtä hyvä kuin tänään!)

---

## Kaksi erilaista suunnitelmaa (käytännöt)

Testasimme kahta käytäntöä ja laskimme jokaisen neliön arvon kullekin:

### Käytäntö 1: Uniform Random
Nosta satunnaisesti ylös, alas, vasemmalle tai oikealle yhtä suurella mahdollisuudella.

```
Values (Uniform Random Policy):
-0.912  -0.932  -0.912  -0.942
-0.929   (H)   -0.898   (H)
-0.901  -0.801  -0.696   (H)
 (H)   -0.630  -0.104   (G)
```

Melkein kaikkialla on **negatiivista** - satunnainen politiikka putoaa reikiin niin
usein missä tahansa oleminen on aika huonoa!

---

### Käytäntö 2: Puolueellinen kohti tavoitetta
Liikkuu mieluummin oikealle ja alas (kohti maalia), mutta silti joskus toisin
ohjeita.

```
Values (Biased-Toward-Goal Policy):
-0.838  -0.895  -0.814  -0.961
-0.798   (H)   -0.665   (H)
-0.595  -0.143  -0.213   (H)
 (H)    0.254   0.673   (G)
```

Nyt **tavoitteen** lähellä olevilla neliöillä on **positiiviset arvot** (0,254 ja 0,673)!
Älykäs politiikka tekee näistä aukioista hyviä paikkoja.

---

## Mitä värit kuvassamme tarkoittavat

Visualisoinnissamme:
- **Vihreät neliöt** = arvokas (hyviä paikkoja olla)
- **Punaiset neliöt** = pieni arvo (vältä näitä!)
- **Keltaiset neliöt** = jonnekin siltä väliltä

Voit nähdä **gradientin** — arvot muuttuvat vihreämmiksi, kun siirryt kohti tavoitetta
ja punaisempi Holesin lähellä.

---

## Miksi välitämme arvoista?

Arvot ovat vahvistavan oppimisen *perusta*! Kun tietää arvon
jokaisessa osavaltiossa voit tehdä järkeviä päätöksiä:

> "Olen ruudussa A. Voin siirtyä neliöön B (arvo = 0,5) tai neliöön C (arvo = -0,3).
> Menen B:hen – sillä on suurempi arvo!"

Juuri näin monet RL-algoritmit (kuten Q-learning) oppivat parantamaan
päätöksiä kertomatta sääntöjä.

**Tosielämän esimerkki:** Kuvittele, että valitset jonon seisomaan
ruokakauppa. Jokainen rivi on "tila". Sen tilan arvo on kuinka nopeasti
selviät kassalla. Katsot viivoja (tarkkailet tiloja) ja valitset
se, jolla on suurin arvo (lyhyin odotus + vähiten kohteita).

---

## Kuinka laskemme arvot

Käytimme **Iteratiivista politiikan arviointia**, joka toimii seuraavasti:

1. Aloita: arvaa, että kaikki arvot ovat 0.
2. Päivitys: laske jokaiselle ruudulle, mihin arvon *pitäisi* perustua
   minne politiikka vie sinut seuraavaksi.
3. Toista, kunnes arvot lakkaavat muuttumasta (konvergoivat).

Matemaattisesti: **V(s) = Σ_a π(a|s) × [R(s,a) + γ × V(seuraava_tila)]**

Selkeästi englanniksi: "Tämän neliön arvo = keskimääräinen palkinto, jonka saan
juuri nyt + vähän sen arvosta, minne päädyn."

---

## Avainsanat muistaa

- **Tila**: Missä olet juuri nyt (yksi ruutu taululla)
- **Arvo V(s)**: Odotettu kokonaispalkkio tilasta s alkaen
- **Käytäntö**: Suunnitelmasi, mitä tehdä kussakin osavaltiossa
- **Alennuskerroin γ**: kuinka paljon välität tulevista palkkioista (0,99 = paljon!)
- **Käytännön arviointi**: Arvojen laskeminen jokaiselle osavaltiolle tietyn käytännön mukaisesti

Suuri idea: **Jotkut paikat ovat parempia kuin toiset – ja arvofunktio
kertoo tarkalleen kuinka hyvä kukin paikka on!**
