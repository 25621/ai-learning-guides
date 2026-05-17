# Jäätynyt järvi satunnaisella käytännöllä 🧊

## Mikä on Frozen Lake?

Kuvittele, että pelaat **jäätyneellä lammikolla** ystäviesi kanssa.

Jää on pääosin turvallista, mutta joissakin paikoissa on **reikiä** - jos astut reikään,
putoat sisään ja peli on ohi! Lammen toisessa päässä on **lahja** 🎁.
Sinun tehtäväsi on liukua **aloitusruudusta** **lahjalle** putoamatta sisään.

Tältä jäätynyt järvi näyttää (4 ruutua × 4 ruutua):

```
S  F  F  F
F  H  F  H
F  F  F  H
H  F  F  G
```

- **S** = Aloitusruutu (mistä aloitat)
- **F** = Jääruutu (turvallista!)
- **H** = Reikä (pudota sisään, peli ohi 😨)
- **G** = Maali — lahja! 🎁

---

## Hankala osa: Liukas jää!

Oikealla jäälammikolla, kun yrität kävellä *oikealle*, joskus jää saa sinut
liukumaan sen sijaan *ylös* tai *alas*! Se tekee tästä vaikeaa.

Vaikka *haluaisit* mennä oikealle, peli saattaa viedä sinut jonnekin muualle.
Tätä kutsutaan **stokastisuudeksi** - se on hieno sana sille, että "asiat eivät aina mene
niin kuin suunnittelit."

---

## Mikä on satunnainen käytäntö?

**Käytäntö** (policy) on vain suunnitelma: "Tässä tilanteessa valitsen TÄMÄN toiminnon."

**Satunnainen käytäntö** tarkoittaa: "Minulla ei ole suunnitelmaa ollenkaan! Valitsen vain satunnaisen
suuntaan joka kerta - ylös, alas, vasemmalle tai oikealle - kuin pyörität kehrää!"

Se on kuin vauva, joka kävelee jäällä ilman aavistustakaan siitä, missä lahja on.

---

## Mitä koodimme löysi

Kokeilimme satunnaista käytäntöä **1 000 pelissä**:

| Tulos | Arvo |
|--------|-------|
| **Lahjalle pääsyt** | 11/1000 (1,1 %) |
| **Keskimääräiset askeleet per peli** | 7,5 askelta |
| **Nopein peli** | 2 askelta |
| **Pisin peli** | 33 askelta |

Suurimman osan ajasta satunnainen kävelijä putosi reikään nopeasti.
Vain yksi 100 pelistä päättyi lahjan löytämiseen!

---

## Miksi tämä on hyödyllistä?

Vaikka satunnainen käytäntö on kauheaa, se antaa meille **perustason** —
vertailukohtana.

Kun rakennamme myöhemmin *älykkään* käytännön (käyttäen Q-learningiä tai muita algoritmeja),
voimme sanoa: "Älykäs agenttimme menestyy 75% ajasta - paljon paremmin kuin
satunnaisen kävelijän 1%!"

**Tosielämän esimerkki:** Kuvittele, että yrität löytää luokkahuoneesi uudesta koulusta
kääntämällä satunnaisesti vasemmalle tai oikealle jokaisessa käytävässä. Saatat päästä sinne
lopulta, mutta se kestää kauan! Älykäs käytäntö on kuin kartta.

---

## Mitä lämpökartta näyttää

Kuvassamme **lämpökartta** näyttää millä ruuduilla satunnainen kävelijä vieraili
useimmiten:

- **Aloitusruudussa** käydään paljon (jokainen peli alkaa sieltä).
- **Reikien** lähellä olevissa ruuduissa käydään vähemmän (kävelijä putoaa usein ennen
  kuin saavuttaa ne).
- **Maaliruudulla** vieraillaan erittäin harvoin, koska satunnainen kävelijä ei juuri koskaan
  pääse sinne.

Tämä kertoo meille jotain tärkeää: satunnainen käytäntö juuttuu lähelle
alkua eikä koskaan oikeastaan tutki koko järveä.

---

## Avainsanat muistaa

- **Käytäntö**: Suunnitelmasi, mitä tehdä kussakin tilanteessa
- **Satunnainen käytäntö**: Ei suunnitelmaa – valitse vain satunnainen toiminta!
- **Perustaso**: Huono tulos, jota käytämme vertailussa (kuinka paljon paremmin voimme tehdä?)
- **Stokastinen**: Asiat eivät aina mene niin kuin suunnittelet (kuten liukas jää!)
- **Onnistumisprosentti**: Kuinka usein voitimme? (Tässä: 1,1 % - erittäin alhainen!)

Suuri idea: **Satunnainen käytäntö on vain lähtökohta. Todellinen oppiminen tarkoittaa
paremman suunnitelman rakentamista!**
