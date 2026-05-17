# Deep Q-Network (DQN) alusta alkaen 🧠

## Lineaarisen ongelma

Muistatko lineaarisen kaavamme aikaisemmasta?

> Pisteet = w₁ × kärryn_sijainti + w₂ × kärryn_nopeus + w₃ × tangon_kulma + w₄ × tangon_kulmanopeus

Tämä toimii hyvin CartPolelle, mutta entä videopeli, jossa näet tuhansia
pikseliä? Sille ei voi kirjoittaa yksinkertaista reseptiä!

Tarvitsemme jotain, joka voi tarkastella monimutkaisia tilanteita ja selvittää parhaan toiminnan.
Se jokin on **hermoverkko**.

---

## Mikä on hermoverkko?

Ajattele aivojasi. Miljoonat pienet solut, joita kutsutaan neuroneiksi, puhuvat keskenään. Kun sinä
kosketa jotain kuumaa, hermosolut lähettävät signaaleja: "KUUMA! → Vedä käsi pois NYT!" Jokainen neuroni
välittää tietoa eteenpäin, ja yhdessä he tekevät viisaan päätöksen.

**Tietokoneen hermoverkko** toimii samalla tavalla:

```
Input Layer      Hidden Layer 1    Hidden Layer 2    Output Layer
[cart pos]   →   [128 neurons]  →  [128 neurons]  →  [push LEFT score]
[cart speed] →   [  ...       ]    [  ...       ]    [push RIGHT score]
[pole angle] →
[pole speed] →
```

Jokaisella nuolella on **paino** (kuinka vahva yhteys on). Niitä on tuhansia
nämä painot – ja verkko oppii ne KAIKKI!

**Tosielämän esimerkki:** Ravintolan kokki maistelee ruokaasi ja muokkaa satoja
ainekset kerralla. Jokainen makuhermo on kuin neuroni, ja yhdessä ne kertovat kokille
"lisää enemmän suolaa" tai "vähemmän pippuria". Verkoston kouluttaminen on kuin kokki oppii yli
tuhansia aterioita.

---

## DQN = syvä Q-verkko

**DQN** (Deep Q-Network) keksi DeepMind vuonna 2013. He ottivat vanhan Q-oppimisen
kaavan ja korvasivat Q-taulukon neuroverkolla!

Sen sijaan:
> Q-taulukko[tila][toiminta] = pisteet

Meillä on:
> Q-verkko(tila) → [pisteet_vasemmalle, pistemäärä_oikealle]

Verkko ottaa tilan syötteeksi ja laskee Q-arvot KAIKILLE toiminnoille kerralla.
Tämä on paljon tehokkaampaa kuin niiden laskeminen erikseen!

---

## Tämä skripti: "Naiivi" versio

Tämä skripti näyttää DQN:n **ilman** erityisiä temppuja. Se vain:
1. Näkee valtion
2. Kysyy verkostolta "kuinka hyvä on vasen? kuinka hyvä on oikea?"
3. Suorittaa toiminnan korkeammalla pistemäärällä
4. Ansaitsee palkinnon, päivittää verkkoa

**Tämä on tarkoituksella epävakaa!** Ajattele sitä kuin opiskelija, joka unohtaa heti
edelliset oppituntinsa joka kerta, kun he oppivat jotain uutta. Verkko päivittyy tämän jälkeen
jokainen askel, joka aiheuttaa kaaosta.

**Tosielämän esimerkki:** Kuvittele, että opit kokkaamaan muuttamalla koko reseptiäsi
jokainen purema. Saatat siirtyä "liian suolaisesta" "ei ollenkaan suolaa" ja "liian suolaista"
eikä koskaan sovi oikeaan summaan. Näin täällä tapahtuu!

---

## Mitä näet

Kun juokset `dqn_cartpole.py`:
- Pisteet saattavat hypätä paljon (epävakaa oppiminen)
- Joskus agentti tulee todella hyväksi, sitten unohtaa kaiken
- Tappiokuvassa näkyy villejä heilahteluja

**Tätä odotetaan!** Se osoittaa, MIKSI tarvitsemme parannuksia – kokemus uusinta ja kohde
verkkoja. Ne tulevat seuraavaksi!

---

## ε-ahne temppu 🎲

Robotti ei aina valitse parasta toimintaa. Joskus se poimii satunnaisesti!

Miksi? Koska jos se aina valitsee sen, mikä näyttää parhaalta, se ei ehkä koskaan löydä parempia vaihtoehtoja.

> Todennäköisyydellä ε (epsilon): valitse SATUNNAIS-toiminto (tutki!)
> Todennäköisyydellä 1-ε: valitse PARAS tunnettu toiminta (hyödynnä!)

Aloitamme arvolla ε = 1.0 (100 % satunnainen) ja vähennämme hitaasti arvoon ε = 0.01 (1 % satunnainen).
Tällä tavalla robotti tutkii aluksi paljon ja keskittyy sitten oppimaansa.

**Esimerkki tosielämästä:** Kun vierailet uudessa kaupungissa, voit kokeilla satunnaisia ravintoloita
ensin (tutkia). Hetken kuluttua palaat suosikkeihisi (hyödynnä). Mutta sinä silti
kokeile joskus jotain uutta, jos siellä on piilotettu helmi!

---

## Avainsanasto

| sana | Merkitys |
|------|---------|
| **Neuroverkko** | Kerroksia yhdistettyjä matemaattisia neuroneja, jotka oppivat tiedosta |
| **Syvä** | Useampi kuin yksi piilotettu kerros (siis "syvä" oppiminen) |
| **DQN** | Deep Q-Network — käyttää hermoverkkoa Q-taulukon sijaan |
| **ε-ahne** | Strategia: Tutki joskus satunnaisesti, hyödynnä parasta osaamista toisinaan |
| **Epävakaus** | Verkko "unohtaa", koska päivitykset häiritsevät toisiaan |

---

## Mitä puuttuu (ja miksi sillä on merkitystä)

Tällä naiivilla DQN:llä on kaksi suurta ongelmaa:

1. **Vastaavat päivitykset**: Jokainen kokemus tulee järjestyksessä (vaihe 1, vaihe 2, vaihe 3...).
   Jos vaihe 5 oli huono, KAIKKI lähellä olevat päivitykset sekoittuvat yhteen.
   
2. **Liikkuva kohde**: Verkko vaihtuu jokaisen päivityksen jälkeen. Mutta seuraava päivitys käyttää
   SAMA verkko laskea, mikä kohteen pitäisi olla. Se on kuin ampuisi liikkuvaa kohdetta
   napakymppi!

Nämä ongelmat ratkaistaan seuraavissa **Experience Replay** ja **Target Networks** -skripteissä.
Yhdessä ne tekevät DQN:n heiluvasta aloittelijasta pelien mestarin!
