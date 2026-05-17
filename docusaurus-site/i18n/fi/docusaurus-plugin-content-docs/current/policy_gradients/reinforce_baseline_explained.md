# VAHVISTA perusviivalla: Leikkaa melun läpi

## Ongelma Plain REINFORCE:n kanssa

Kuvittele, että olet opiskelija ja yrität päättää, oliko vastauksesi kokeessa hyvä.

**Huono palaute:** "Sait 7 pistettä!"

Onko 7 hyvä? Jos enimmäismäärä on 10, kyllä! Jos kaikki muut saivat 9, ei! Ilman kontekstia,
et voi sanoa, pitäisikö sinun muuttaa vastaustyyliäsi.

Juuri tämä on REINFORCE-ongelma: se käyttää **raakoja palautuksia** (G_t) arvioidakseen
toimia. 200 pisteen kokonaispalautus voi olla hämmästyttävää tai kauheaa tilanteesta riippuen.

---

## Syötä perusviiva

**perusviiva** b(t) on viitepiste: "Mitä palkintoa **odotan** tässä tilanteessa?"

Sen sijaan, että kysyisimme "Oliko tämä toiminta hyvä?", kysymme:

> **"Oliko tämä toiminta parempi kuin mitä normaalisti odotin?"**

```
Old signal: update ∝ G_t
New signal: update ∝ (G_t - b(s_t))
```

**Tosielämän esimerkki:** Sait 85 pistemäärää matematiikan kokeessa.
- Jos luokan keskiarvo on 60 → vastauksesi oli **25 pistettä keskiarvon yläpuolella** → hienoa!
- Jos luokan keskiarvo on 90 → vastauksesi oli **5 pistettä keskiarvon alapuolella** → vaatii työtä!

**Etu** (G_t - b(s)) on positiivinen, kun onnistuit odotettua paremmin ja
negatiivinen, kun sinulla meni huonommin. Tämä on paljon puhtaampi oppimissignaali!

---

## Mikä on perusviiva?

Luonnollinen perusviiva on **arvofunktio V(s)**:

> V(s) = "Odotettu kokonaispalkkio, jos olen tilassa s ja pelaan nykyisen käytäntöni"

Opimme tämän erillisen **Arvoverkon** avulla (kutsutaan myös perusverkostoksi tai kriitikkoksi):

```
State  →  [128 neurons]  →  [128 neurons]  →  V(s)   (single number)
```

Jokaisen edustajan vieraileman tilan osalta V(s) ennustaa odotetun tuoton. Jos todellinen
tuotto G_t on suurempi kuin V(s), toiminta oli odotettua parempi!

---

## Kaksi verkkoa oppimassa yhdessä

```
Episode happens
     ↓
Compute actual returns G_t
     ↓
         ┌─────────────────────────────┐
         │ Advantage = G_t - V(s_t)    │
         │  +: action was better        │
         │  -: action was worse         │
         └─────────────────────────────┘
              ↓                  ↓
    Update Policy Network   Update Value Network
    (make good actions     (make predictions more
     more/less likely)      accurate next time)
```

**Tosielämän esimerkki:** Kaksi ystävää menevät yhdessä ravintolaan.

- Ystävä 1 (arvoverkosto): "Ennustan, että tämä ruokalaji on 7/10"
- Friend 2 (Policy Network): Kokeile ruokaa ja anna sille arvosana 9/10
- Etu = 9 - 7 = +2 → "Se oli odotettua parempi! Tilaa uudelleen!"

Seuraavalla vierailulla Friend 1 päivittää ennusteensa lähemmäksi 9/10.
Ystävä 2 tilaa todennäköisemmin kyseisen ruuan ensi kerralla.

---

## Miksi tämä vähentää varianssia?

**Matemaattinen todiste (intuitio):**

Ilman perusarvoa: `gradient ∝ ∇log π(a|s) × G_t`

G_t-arvot vaihtelevat paljon jaksosta toiseen:
```
Episode 1: G = [45, 44, 43, ..., 1]   (medium game)
Episode 2: G = [500, 499, ..., 1]      (great game!)
Episode 3: G = [12, 11, ..., 1]        (terrible game)
```

Gradienttiarviot hyppäävät villisti, koska G_t on suuri ja meluisa.

Perustason kanssa: `gradient ∝ ∇log π(a|s) × (G_t - V(s_t))`

Etu (G_t - V(s_t)) on paljon pienempi ja lähellä nollaa:
```
Episode 1: advantage ≈ [-2, +1, -3, ..., 0]   (small, centered)
Episode 2: advantage ≈ [+10, +8, ..., +3]      (this game WAS great)
Episode 3: advantage ≈ [-5, -6, ..., -2]       (this game WAS bad)
```

**Tosielämän esimerkki:** Juoksunopeuden mittaaminen.
- Ilman perusarvoa: "Juostin 8 km/h" (merkityksetön ilman kontekstia)
- Perustasolla: "Juosin 2 km/h keskimääräistä nopeammin" (selvästi hyvä!)

Etuna on aina vertailu – se on luonnollisesti pienempi ja vakaampi.

---

## Ratkaisevaa: Ei ennakkoluuloja!

Perustaso ei muuta MITÄ algoritmi oppii – vain KUINKA NOPEASTI ja VAKAASTI se oppii.

**Miksi?** Koska odotettu etu on aina 0 odotuksessa:

> E[G_t - V(s_t)] = E[G_t] - V(s_t) = V(s_t) - V(s_t) = 0

Mikä tahansa b(t), joka ei riipu toiminnosta, toimii kelvollisena perusviivana!

**Tosielämän esimerkki:** Arviointi käyrällä ei muuta sitä, kuka suoriutui parhaiten – se vain
tekee pisteistä helpompi tulkita. Sijoitus pysyy samana; vain mittakaava muuttuu.

---

## Tulokset

```
No baseline  — Final 100-ep avg: 500.0, grad var: 599.3
With baseline — Final 100-ep avg: 491.4, grad var: 578.8
```

Molemmat menetelmät saavuttavat lähes täydellisen suorituskyvyn CartPolessa, mutta huomaa:
1. **Gradientin varianssi** on mitattavissa (oikealla puolella oleva kaavio näyttää vaihtelun harjoittelussa)
2. Perustasolla agentti saavuttaa korkean suorituskyvyn **luotettavammin** — vähemmän kaatumisia ja alhainen palkkio harjoittelun aikana

Varianssin pieneneminen on dramaattisempaa kovemmissa ympäristöissä (LunarLander, MuJoCo).

---

## Keskeiset yhtälöt

```
Baseline value:   V(s) ← V(s) + α(G_t - V(s))   [minimize MSE]
Policy gradient:  θ ← θ + α ∇log π(a_t|s_t) · (G_t - V(s_t))
Advantage:        A_t = G_t - V(s_t)
```

---

## Avaimet takeawayt

| käsite | Pelkkää englantia |
|---------|---------------|
| **Perustason b(t)** | Odotettu palkkio tilassa s – vertailupisteemme |
| **Etu A_t** | "Oliko tämä toimenpide odotettua parempi?" |
| **Arvoverkosto** | Neuraaliverkko, joka oppii ennustamaan odotettuja tuottoja |
| **Varianssin vähennys** | Vähemmän kohinaa gradienttiarvioissa → vakaampi oppiminen |
| ** Puolueeton** | Perustaso ei muuta tavoitekäytäntöä keskimäärin; se vain tekee oppimissignaalista vähemmän meluisan ja vakaamman |

---

## Mitä seuraavaksi?

Lähtökohta on itse asiassa alku jollekin paljon tehokkaammalle: **Actor-Critic** -menetelmille.

Sen sijaan, että toimija-arvioija laskisi V(t) vain jakson lopussa, hän päivittää
V(t) jokaisessa vaiheessa käyttämällä **Ajallisen eron** oppimista. Tämä tekee päivityksiä
paljon nopeampi ja antaa agentille mahdollisuuden oppia keskeneräisistä jaksoista!

Katso `a2c_lunarlander.py` koko Actor-Critic-toteutukseen.
