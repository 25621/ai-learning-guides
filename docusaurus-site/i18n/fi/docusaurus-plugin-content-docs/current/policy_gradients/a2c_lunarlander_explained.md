# A2C: Toimija ja arvioija työskentelevät yhdessä

## Suuri idea

REINFORCE odottaa, kunnes peli on täysin ohi ennen päivittämistä. Se on kuin valmentaja, joka
katselee koko jalkapallopeliä hiljaa ja antaa sitten kaiken palautteen vasta lopussa.

**A2C (Advantage Actor-Critic / toimija-arvioija (actor-critic))** antaa palautetta pelin AIKANA – muutaman askeleen välein valmentaja
pysähtyy sanomaan: "Tuo syöttö oli hieno! Tuo taklaus oli huono!"

Tämä on paljon nopeampaa ja tehokkaampaa.

---

## Esittelyssä kaksi roolia

> **Mikä LunarLander on?** Tässä asiakirjassa käytämme **LunarLander**-ympäristöä – fysiikan simulaatiota, jossa ohjaat pientä avaruusalusta ja sinun on laskettava se pehmeästi kuuhun kohdetyynylle käyttämällä kolmea moottoria (vasen, pää ja oikea). Se on vahvistusoppimisen vakiovertailu, joka on saatavilla Gymnasiumin kirjastossa.

### Toimija (Actor) 🎭
**Toimija** on käytäntö (policy) – se päättää mihin toimiin ryhdytään.

> "Olen tässä tilassa. Pitäisikö minun käynnistää vasen vai oikea moottori?"

**Tosielämän esimerkki:** Auton *kuljettaja*, joka kääntää ohjauspyörää ja painaa polkimia.

### Arvioija (Critic) 🎬
**Arvioija** arvioi, kuinka hyvä nykyinen tilanne on – arvo V(s).

> "TÄSSÄ tilassa oleminen on noin +150 pisteen arvoinen tulevaisuuden kokonaispalkkiosta."

**Tosielämän esimerkki:** *navigaattori* istuu kuljettajan vieressä ja sanoo "Olemme hyvällä tiellä -
odotan saapuvamme 30 minuutissa." tai "Olemme menossa ruuhkaan - tämä tulee olemaan hidasta."

### He jakavat aivot
Toteutuksessamme molemmat käyttävät **samaa neuroverkon runkoverkkoa**:

```
          State (8 numbers for LunarLander)
                       ↓
          ┌─────────────────────────┐
          │  Shared Layers          │
          │  [256 neurons] → ReLU   │
          │  [256 neurons] → ReLU   │
          └────────┬────────┬───────┘
                   ↓        ↓
          Actor Head    Critic Head
          [4 outputs]   [1 output]
          (action probs) (V(s))
```

- **ReLU** (Rectified Linear Unit): aktivointifunktio, jota käytetään jokaisen kerroksen jälkeen - se tulostaa `max(0, x)`, pitää positiiviset arvot ja nollaa pois negatiiviset. Tämä antaa verkon oppia epälineaarisia kuvioita.
- **toimintatodennäköisyydet**: todennäköisyys suorittaa jokainen neljästä toiminnosta. Toimija (actor) ottaa näytteitä tästä jakaumasta valitakseen toiminnon jokaisessa vaiheessa.

**Tosielämän esimerkki:** Yksi aivot, kaksi tehtävää – kuten taksinkuljettaja, joka sekä ajaa (toimija)
JA tietää, onko reitti hyvä (arvioija). Aivojen jakaminen tarkoittaa nopeampaa oppimista!

---

## Etu: Oliko tämä odotettua parempi?

Aivan kuten REINFORCE perusviivalla, A2C laskee **edun**:

> A(s, a) = "Todellinen tulos" − "Mitä odotimme"

Mutta tässä "todellinen tulos" tulee kriitikon **n-vaiheen bootstrapista** (**bootstrapping** = käyttämällä kriitikon omaa ennustetta V(s) tulevien vaiheiden arvon arvioimiseen sen sijaan, että odotat todellisen jakson päättymistä – kuten arvioidaksesi loppukokeen pisteet lukukauden puolivälissä nykyisen arvosanasi perusteella):

```
Actual TD return: r_t + γ · r_{t+1} + γ² · r_{t+2} + ... + γⁿ · V(s_{t+n})
Advantage A_t = TD return - V(s_t)
```

**Tosielämän esimerkki:** Odotat tekeväsi 3 maalia tässä pelissä (V(s)). Jos teet 5 maalia,
etusi on +2. Jos teet yhden maalin, etusi on -2.

Positiivinen etu → "toimi auttoi odotettua enemmän → tee se enemmän!"
Negatiivinen etu → "toimi auttoi odotettua vähemmän → tee se vähemmän!"

---

## Miksi käyttää useita rinnakkaisia ympäristöjä?

A2C käyttää **8 kopiota** LunarLanderista samaan aikaan!

**Miksi?** Koska kokemukset yhdestä ympäristöstä ovat **korreloivia** – yksi askel
seuraa edellistä vaihetta tarkasti. Tämä korrelaatio huijaa hermoverkon ajattelemaan
mallit ovat luotettavampia kuin ne ovat.

8 ympäristössä jokainen vaihe antaa 8 itsenäistä kokemusta hyvin erilaisista tilanteista.
Tämä rikkoo korrelaation ja vakauttaa harjoittelun dramaattisesti.

**Tosielämän esimerkki:** Saat lisätietoja säästä, mikä on parempi:
- Yhden kaupungin katselu 8 peräkkäistä tuntia (korreloi - jos aurinko paistoi klo 14, se on todennäköisesti aurinkoinen klo 15)
- Katsotaan 8 kaupunkia samanaikaisesti (sisustettu - erilaiset säämallit, lisätietoja!)

```
Environment 1: [landed on moon, fire left, crash, reset...]
Environment 2: [falling too fast, fire both, hover, land...]
Environment 3: [tilting right, fire right, stabilize, land...]
...
Environment 8: [drifting left, fire left, steady, ...]
```

Kaikki 8 päivittävät verkon samanaikaisesti – 8 kertaa monipuolisempi kokemus päivitystä kohti!

---

## N-vaiheen päivitykset: Älä odota pelin päättymistä

REINFORCE odottaa täyttä jaksoa (voi olla 1000 askelta!).

A2C päivittää joka **n_steps = 128 vaihetta**:

```
Play 128 steps across 8 environments
    → Get 128 × 8 = 1024 experience tuples
    → Compute advantages and returns
    → Update the Actor and Critic
    → Play 128 more steps...
```

**Tosielämän esimerkki:** Opiskelija, joka opiskelee kokeeseen.
- REINFORCE-tyyli: Lue koko oppikirja ja suorita vasta sitten harjoituskokeet.
- A2C-tyyli: Lue 10 sivua, tee tietokilpailu, lue vielä 10 sivua, tee tietokilpailu...

Useampi palaute = nopeampi oppiminen!

---

## Kolme tappiota yhdistettynä

A2C-koulutuksessa käytetään kolmea eri tappiotermiä samanaikaisesti:

**Tappio** (loss) on luku, jonka optimoija yrittää minimoida. Pienempi tappio tarkoittaa verkon
nykyisen käyttäytymisen olevan lähempänä harjoittelun tavoitetta.

### 1. Toimijan menetys (policy gradient)
Tee edullisista toimista todennäköisempää:
```
L_actor = -E[log π(a|s) · A(s,a)]
```
Jos A > 0: lisää toimenpiteen todennäköisyyttä
Jos A < 0: pienennä toimenpiteen todennäköisyyttä

### 2. Arvioijan menetys (arvofunktion MSE)
Tee arvoennusteista tarkempia (**MSE** = Mean Squared Error: neliöi ennustevirheet ja ottaa niistä keskiarvon - neliöinti rankaisee suuria virheitä raskaammin kuin pieniä):
```
L_critic = E[(V(s) - return)²]
```
Kuten minkä tahansa **regressio**-mallin harjoittelu (regressio = jatkuvan luvun ennustaminen, tässä odotettu tuotto V(s)) – minimoi ennustevirhe.

### 3. Entropiabonus (tutkimus)
Pidä käytäntö muuttumasta liian itsevarmaksi liian nopeasti:
```
L_entropy = -H[π(·|s)] = E[log π(a|s)]
```
Korkea entropia = monipuoliset toimintavaihtoehdot = tutkimus
Matala entropia = itsevarma, kapeat valinnat = hyödyntäminen

**Tosielämän esimerkki:** Entropiabonus on kuin opettaja sanoisi: "Älä vain valitse A-vaihtoehtoa jokaiseen
monivalintakysymykseen! Kokeile erilaisia vastauksia, jotta opit, mikä toimii."

```
Total loss = L_actor + 0.5 × L_critic - 0.01 × entropy
```

---

## LunarLander: Vaikeampi haaste

**LunarLander-v3** on Gymnasium (entinen OpenAI Gym) -ympäristö – "v3" on versionumero, joka ilmaisee tämän ympäristön kolmannen version. Agentti ohjaa pientä avaruusalusta, jonka on laskeuduttava turvallisesti määrätylle alustalle kuuhun. Se on paljon vaikeampaa kuin CartPole:
- 8-ulotteinen tila-avaruus (sijainti, nopeus, kulma, jalkakosketus, polttoaine)
- 4 erillistä toimintoa (älä tee mitään, käynnistä vasen moottori, käynnistä päämoottori, käynnistä oikea moottori)
- Palkkio: +100 laskeutumisesta, -100 törmäyksestä, pienet polttoainemaksut

Harjoittelukäyrä osoittaa asteittaista paranemista erittäin negatiivisista palkkioista kohti positiivisia.
LunarLanderin A2C vaatii huomattavaa kokemusta ennen kuin laskeutuja oppii perusvakauden.

---

## Keskeiset yhtälöt

```
n-step return:  G_t = r_t + γ·r_{t+1} + ... + γⁿ·V(s_{t+n})
Advantage:      A_t = G_t - V(s_t)
Actor update:   θ_π ← θ_π - α ∇ L_actor
Critic update:  θ_V ← θ_V - α ∇ L_critic
```

---

## Avaimet takeawayt

| käsite | Pelkkää englantia |
|---------|---------------|
| **toimija** | Käytäntö (policy) päättää, mitä tehdä |
| **arvioija** | Arvofunktio – arvioi, kuinka hyvä tilanne on |
| **Etu** | "Oliko tämä odotettua parempi?" (todellinen - odotettu) |
| **N-vaiheinen paluu** | Katso n askelta tulevaisuuteen ennen kuin käytät arvoa V(s) |
| **Rinnakkaiset envs** | Useita ympäristöjä rinnakkain monipuolisten kokemusten saamiseksi |
| **Entropiabonus** | Kannustaa toimijaa jatkamaan uusien asioiden kokeilua |

---

## Mitä seuraavaksi?

A2C on hieno, mutta siinä on yksi suuri heikkous: se päivittää käytäntöä joskus liian aggressiivisesti.
Yksi huono päivitys voi tuhota kaiken edellisen päivityksen hyvän oppimisen.

**PPO (Proximal Policy Optimization)** korjaa tämän älykkäällä rajoituksella (clipping), joka estää
yksittäistä päivitystä muuttamasta käytäntöä liikaa.

Katso `ppo_scratch.py` PPO:n toteuttamiseksi!
