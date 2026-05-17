# VAHVISTA: Robotin opettaminen tekemään parempia valintoja

## Mitä yritämme tehdä?

Kuvittele, että sinulla on robotti, joka pelaa videopeliä. Joka sekunti robotin on valittava:
**"Pitäisikö minun painaa nappia vai ei?"**

Sen sijaan, että muistaisimme jokaisen tilanteen taulukossa (kuten Q-learning), haluamme robotin oppivan
**resepti** — sääntöjoukko, joka sanoo suoraan: "Tee tässä tilanteessa tämä."

Tätä reseptiä kutsutaan **käytännöksi** (π, pi). Vahvistuksessa
π tarkoittaa "toimintojen valinnan sääntöä".

---

## Vanha tapa vs. Uusi tapa

**Vanha tapa (Q-learning / DQN):** Opi kuinka HYVÄ kukin toiminta on (Q-arvot), ja valitse sitten paras.
> "Vasemmalle työntäminen saa pisteet 7, OIKEALLE työntäminen 5 → paina VASEN!"

**Uusi tapa (Policy Gradient):** Opi suoraan, mikä toiminto VALITSE.
> "Kun sauva kallistuu oikealle, paina OIKEALLE 80 %:n todennäköisyydellä, paina VASEN 20 %:n todennäköisyydellä."
*(Sana **Gradient** viittaa matemaattiseen "askeleen", jonka otamme säätääksemme näitä todennäköisyyksiä hitaasti oikeaan suuntaan.)*

**Tosielämän esimerkki:** Opitaan ajamaan pyörällä.
- Vanha tapa: laske *tarkka pistemäärä* arvoille "kala vasemmalle 5 astetta" vs. "kalle vasemmalle 7 astetta".
- Uusi tapa: harjoittele vain, kunnes **kehosi** oppii – työnnä oikealta tuntuvaa jalkaa!

---

## Miten REINFORCE toimii?

REINFORCE katsoo, että robotti pelaa koko pelin alusta loppuun (yksi **jakso**).
kysyy: "Mitkä teot johtivat hyviin pisteisiin? Tehdään niitä lisää!"

### Askel askeleelta

**1. Toista jakso**

Robotti tekee valintoja ja kerää kokemusta:
```
Step 1: State = [pole tilting right] → Action = push RIGHT → Reward = +1
Step 2: State = [pole nearly balanced] → Action = push RIGHT → Reward = +1
Step 3: State = [pole tilting left] → Action = push LEFT  → Reward = +1
...
Step 47: State = [pole fell!] → Episode over
```

**2. Laske palautukset**

Laske jokaiselle vaiheelle G_t — **kokonaispalkkio tästä pisteestä eteenpäin**:
```
G_at_step_47 = 1
G_at_step_46 = 1 + 0.99 × 1 = 1.99
G_at_step_45 = 1 + 0.99 × 1.99 = 2.97
...
G_at_step_1  = 47 (roughly — higher return because it was from the start)
```

γ = 0,99 **alennuskerroin** tarkoittaa, että palkinnot tulevaisuudessa laskevat hieman vähemmän.

**Esimerkki tosielämästä:** Kultatähden saaminen ensimmäisenä koulupäivänä tuntuu jännittävämmältä kuin tietäminen
*saatat* saada sellaisen 100. päivänä. Tulevat palkinnot "alennetaan" hieman.

**3. Päivitä käytäntö**

Jokaisen suoritetun toimenpiteen osalta:
> Jos G_t oli KORKEA (toimi johti loistavaan tulokseen): **tee sitä enemmän!**
> Jos G_t oli ALHAINEN (toimi johti huonoon tulokseen): **tee sitä vähemmän!**

Matematiikka: `loss = -log_prob(action) × G_t`

Gradientin ottaminen ja käytännön päivittäminen on kuin kertoisi robotille:
*"Se toimenpide, jonka teit vaiheessa 20? Sinun pitäisi tehdä se 5 % useammin ensi kerralla!"*

---

## Mikä on politiikkaverkosto?

Taulukon sijasta käytämme **hermoverkkoa** edustamaan käytäntöä.

```
Observation      Policy Network       Action Probabilities
[cart pos]  →   [128 neurons]  →  →  [push LEFT: 30%]
[cart speed] →  [128 neurons]        [push RIGHT: 70%]
[pole angle] →
[pole speed] →
```

Verkko tulostaa **todennäköisyydet** jokaiselle toiminnolle. Otamme sitten näytteen:
> Heitä noppaa → 1-30: työnnä VASEN, 31-100: työnnä OIKEALLE

**Tosielämän esimerkki:** Sääsovellus sanoo "70 %:n mahdollisuus sateeseen". Et tiedä, että sataa – sinä
päättää todennäköisyyden perusteella. Robotti tekee saman asian!

---

## Normalisointi: miksi vähennämme keskiarvon

Ennen kuin käytät G_t:tä päivittämiseen, normalisoimme:
```
G_normalized = (G - mean(G)) / std(G)
```

**Miksi?** Kuvittele, että kaikki palkinnot ovat positiivisia (jotka ne ovat CartPolessa – aina +1 askelta kohti).
Ilman normalisointia KAIKKI toiminta näyttää "hyvältä" ja päivityssignaali on hämmentävä.

Normalisoinnin jälkeen jotkut tuotot ovat positiivisia (keskiarvon yläpuolella → työnnä enemmän), ja jotkut ovat
negatiivinen (keskiarvon alapuolella → paina vähemmän). Signaalista tulee paljon puhtaampi!

**Tosielämän esimerkki:** Opettajasi arvostelee käyrällä. Jos keskimääräinen pistemäärä on 70 ja saat
85, se on hienoa! Mutta jos keskiarvo on 90 ja sinulla on 85, se on alle keskiarvon. Raaka pistemäärä
ei yksin kerro koko tarinaa.

---

## Ongelma: Suuri varianssi

REINFORCElla on suuri heikkous: **varianssi**. Palautukset G_t ovat erittäin meluisia.

**Tosielämän esimerkki:** Kuvittele, että arvostelet kokin maistamalla vain YHDEN ateria kustakin ravintolasta.
Joskus kokilla oli huono päivä, joskus ainekset olivat poissa. Yksi ateria ei riitä
tietää luotettavasti, onko ravintola hyvä!

REINFORCE odottaa KOKO jaksoa ennen päivittämistä. Yksi jakso voi olla erittäin onnekas, toinen
erittäin epäonninen. Kaltevuus hyppää kaikkialle.

Tästä syystä oppimiskäyrä (kaaviossa) näyttää rosoiselta:
- Jotkut juoksut saavat 500 (mahtavaa!)
- Jotkut juoksut putoavat 50:een (kauheaa!)

Melusta huolimatta REINFORCE oppii lopulta – mutta se vaatii kärsivällisyyttä.

---

## Tulokset

```
Episode  100 | Avg reward (last 100):  43.1
Episode  200 | Avg reward (last 100): 193.9
Episode  500 | Avg reward (last 100): 408.4
Episode 1000 | Avg reward (last 100): 500.0  ← Solved!
```

Robotti oppii tasapainottamaan sauvaa enintään 500 askelta - RATKAISTU!

Varianssiongelmistaan huolimatta REINFORCE on CartPole on tehokas, koska:
1. Jaksot ovat lyhyitä (joten saamme monta per harjoitus)
2. Optimaalinen toimintatapa on yksinkertainen (useimmiten työnnä tangon kallistussuuntaan)

---

## Avaimet takeawayt

| käsite | Pelkkää englantia |
|---------|---------------|
| **käytäntö** | Robotin resepti toimintojen valitsemiseen |
| **Jakso** | Yksi täysi peli alusta loppuun |
| **Palautus G_t** | Koko tuleva palkinto tästä hetkestä |
| **Alennus γ** | Tulevat palkinnot ovat hieman vähemmän tärkeitä kuin välittömät |
| **normalisoi** | Vähennä keskiarvo, jotta signaali on puhtaampi |
| **Varianssi** | Kuinka paljon gradienttiarviot hyppäävät |

---

## Mitä seuraavaksi?

REINFORCE:n suuri heikkous on **varianssi**. Seuraavassa käsikirjoituksessa (`reinforce_baseline.py`),
lisäämme **perustason**, joka vähentää dramaattisesti tätä melua – muuttamatta sitä
Algoritmi oppii keskimäärin.

Keskeinen idea: sen sijaan että kysyisit "oliko tämä toiminta hyvä?" kysymme: "Oliko tämä toiminta **parempi kuin
odotettu**?" Tämä pieni muutos tekee oppimisesta paljon vakaampaa.
