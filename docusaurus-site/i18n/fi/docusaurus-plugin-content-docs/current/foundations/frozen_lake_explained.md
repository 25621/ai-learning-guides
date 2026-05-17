# Jäätynyt järvi satunnaisella käytännöllä 🧊

## Mikä on Frozen Lake?

Kuvittele, että pelaat **jäätyneellä lammikolla** ystäviesi kanssa.

Jää on pääosin turvallista, mutta joissakin paikoissa on **reikiä** - jos astut reikään,
putoat sisään ja peli on ohi! Lammen toisessa päässä on **lahja** 🎁.
Sinun tehtäväsi on liukua **alusta** **nykyhetkeen** putoamatta sisään.

Tältä jäätynyt järvi näyttää (4 neliötä × 4 ruutua):

```
S  F  F  F
F  H  F  H
F  F  F  H
H  F  F  G
```

- **S** = Aloita (mistä aloitat)
- **F** = pakastejää (turvallista!)
- **H** = reikä (pudota sisään, peli ohi 😨)
- **G** = tavoite - nykyhetki! 🎁

---

## Hankala osa: Liukas jää!

Oikealla jäälammikolla, kun yrität kävellä *oikealle*, joskus jää tekee
liu'utat sen sijaan *ylös* tai *alas*! Se tekee tästä vaikean.

Vaikka *haluaisit* mennä oikealle, peli saattaa viedä sinut jonnekin muualle.
Tätä kutsutaan **stokastisuudeksi** - hieno sana "asiat eivät aina mene
niin kuin suunnittelit."

---

## Mikä on satunnainen politiikka?

**politiikka** on vain suunnitelma: "Tässä tilanteessa teen TÄMÄN toimenpiteen."

**Satunnainen käytäntö** tarkoittaa: "Minulla ei ole suunnitelmaa ollenkaan! Valitsen vain satunnaisen
suuntaan joka kerta - ylös, alas, vasemmalle tai oikealle - kuin pyörität kehrää!"

Se on kuin vauva, joka kävelee jäällä ilman aavistustakaan, missä nykyisyys on.

---

## Mitä koodimme löysi

Kokeilimme satunnaista käytäntöä **1 000 pelille**:

| Tulos | Arvo |
|--------|-------|
| **Ajat tulivat nykypäivään** | 11/1000 (1,1 %) |
| **Keskimääräiset askeleet per peli** | 7,5 askelta |
| **Nopein peli** | 2 askelta |
| **Pisin peli** | 33 askelta |

Suurimman osan ajasta satunnainen kävelijä putosi kuoppaan nopeasti.
Vain yksi 100 pelistä päättyi lahjan löytämiseen!

---

## Miksi tämä on hyödyllistä?

Vaikka satunnainen politiikka on kauheaa, se antaa meille **perustason** —
vertailukohtana.

Kun rakennamme myöhemmin *älykkään* käytännön (käyttäen Q-learningiä tai muita algoritmeja),
voimme sanoa: "Älykäs agenttimme menestyy 75% ajasta - paljon paremmin kuin
satunnaisen kävelijän 1%!"

**Tosielämän esimerkki:** Kuvittele, että yrität löytää luokkahuoneesi uudesta koulusta
kääntämällä satunnaisesti vasemmalle tai oikealle jokaisessa käytävässä. Saatat päästä sinne
lopulta, mutta se kestää kauan! Älykäs politiikka on kuin kartta.

---

## Mitä lämpökartta näyttää

Kuvassamme **lämpökartta** näyttää millä ruuduilla satunnainen kävelijä vieraili
useimmiten:

- **Aloitus**-ruudulla käydään paljon (jokainen peli alkaa sieltä).
- **reikien** lähellä olevilla aukioilla käydään vähemmän (kävelijä putoaa usein ennen
  saavuttaa heidät).
- **Goal** vierailee hyvin harvoin, koska satunnainen kävelijä ei juuri koskaan
  pääsee sinne.

Tämä kertoo meille jotain tärkeää: satunnainen politiikka juuttuu lähelle
alussa eikä koskaan oikeastaan tutki koko järveä.

---

## Avainsanat muistaa

- **Käytäntö**: Suunnitelmasi, mitä tehdä kussakin tilanteessa
- **Satunnainen käytäntö**: Ei suunnitelmaa – valitse vain satunnainen toiminta!
- **Perustaso**: Huono tulos, jota käytämme vertailussa (kuinka paljon paremmin voimme tehdä?)
- **Stokastinen**: Asiat eivät aina mene niin kuin suunnittelet (kuten liukas jää!)
- **Onnistumisprosentti**: Kuinka usein voitimme? (Tässä: 1,1 % - erittäin alhainen!)

Suuri idea: **Satunnainen politiikka on lähtökohta. Todellinen oppiminen tarkoittaa
rakentaa parempi suunnitelma!**
