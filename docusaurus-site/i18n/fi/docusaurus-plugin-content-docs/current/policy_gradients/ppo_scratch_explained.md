# PPO: Turvalliset ja vakaat käytäntöpäivitykset

## Ongelma A2C:n kanssa

Kuvittele, että opettelet tasapainottamaan luudanvartta sormesi. Viikon harjoittelun jälkeen
voit pitää sen ylhäällä 30 sekuntia!

Nyt valmentajasi antaa sinulle neuvoja: "Kallista rannettasi hieman enemmän vasemmalle."

**Hyviä neuvoja → huolellinen muutos → pidä tasapainoa 30 sekuntia ✓**

Mutta entä jos valmentaja ylireagoi ja sanoo: "KALLISTA VÄLITTÖMÄSTI VOIMAKKAASTI VASEMMALLE!"
Ylikorjaat → luudanvarsi putoaa → olet menettänyt viikkojen edistymisen.

Tämä on A2C-ongelma: **suuret gradienttipäivitykset voivat tuhota hyvän käytännön**.

**PPO (Proximal Policy Optimization)** on turvajärjestelmä, joka estää tämän.

---

## Ydinidea: Pysy lähellä sitä, mikä toimi

PPO:n keskeinen rajoitus:

> **"Älä muuta käytäntöä liikaa yhdessä päivityksessä."**

Ennen päivitystä meillä on "vanha" käytäntö π_old.
Päivityksen jälkeen meillä on "uusi" käytäntö π_new.

PPO mittaa, kuinka paljon käytäntö muuttui **todennäköisyyssuhteella**:

```
r(θ) = π_new(a|s) / π_old(a|s)
```

- r = 1,0: käytäntö ennallaan
- r = 1,5: uusi käytäntö valitsee tämän toiminnan 50 % todennäköisemmin
- r = 0,5: uusi käytäntö valitsee tämän toiminnan 50 % epätodennäköisemmin

**Tosielämän esimerkki:** Olet kokki, joka muokkaa reseptiä.
- r = 1,0: sama määrä suolaa kuin ennen
- r = 2,0: kaksinkertainen suola - liian äärimmäinen!
- r = 0,9: 10 % vähemmän suolaa – pieni, turvallinen vaihto

---

## Leikkaamisen temppu

PPO leikkaa suhteen pysyäkseen rajoissa [1-ε, 1+ε] (tyypillisesti ε = 0,2):

```
L_CLIP = E[min(r(θ) · A,  clip(r(θ), 1-ε, 1+ε) · A)]
```

Puretaan tämä:

**Tapaus 1: Toiminta oli HYVÄ (A > 0)**

Haluamme tehdä tämän toiminnon enemmän (r > 1). Mutta rajoitamme, kuinka paljon lisäämme:
```
if r > 1.2: clip to 1.2, no more incentive to push further
```
Tämä estää meitä heilumasta LIIAN pitkälle yhteen suuntaan.

**Tapaus 2: Toiminta oli HUONO (A < 0)**

Haluamme tehdä tämän toiminnon vähemmän (r < 1). Mutta taas päätämme:
```
if r < 0.8: clip to 0.8, no more penalty for going further
```

**Visuaalinen:**
```
ε = 0.2, so the safe ratio window is 0.8 to 1.2.

GOOD action (A > 0): increase the action probability, but stop rewarding it after 1.2
ratio r:       0.6      0.8      1.0      1.2      1.4
incentive:      ↑        ↑        ↑        ↑        -
meaning:     too low     ok      old      max     clipped

BAD action (A < 0): decrease the action probability, but stop rewarding it below 0.8
ratio r:       0.6      0.8      1.0      1.2      1.4
incentive:      -        ↓        ↓        ↓        ↓
meaning:     clipped    max      old       ok    too high
```

The `-` merkitsee tasaisen leikatun alueen. Tällä alueella, jolloin todennäköisyyssuhde on tasainen
äärimmäisempi ei paranna tavoitetta, joten PPO:lla ei ole ylimääräistä kannustinta työntyä pidemmälle.

**Tosielämän esimerkki:** Auton nopeudenrajoitin. Voit kiihdyttää, mutta kun saavutat 120 km/h,
rajoitin käynnistyy eikä anna sinun mennä nopeammin. Se pitää sinut turvassa pysähtymättä
sinua muuttamasta.

---

## Miksi tämä estää katastrofaaliset päivitykset

**Katastrofaalinen päivitys** on, kun yksi suuri käytännön muutos tuhoaa kokonaan kaiken
agentti on oppinut – koulutustunteja on kulunut yhdellä gradienttivaiheella.

Ilman leikkaamista: yksi suuri gradienttiaskel voi muuttaa käytäntöä radikaalisti.
Leikkauksella: gradientti on nolla arvon [1-ε, 1+ε] ulkopuolella, joten käytäntö voi liikkua vain vähän askelta kohti.

**Tosielämän esimerkki:** Hyvä kirurgi tekee pieniä, tarkkoja leikkauksia – ei suuria, lakaisuisia.
PPO on RL-optimoijien "huolellinen kirurgi".

---

## GAE: Smarter Advantage Estimates

PPO käyttää **Generalized Advantage Estimation (GAE)** laskeakseen edun:

```
δ_t = r_t + γ · V(s_{t+1}) - V(s_t)          (TD error)
A_t = δ_t + γλ · δ_{t+1} + (γλ)² · δ_{t+2} + ...
```

GAE:llä on parametri λ (lambda):
- λ = 0: käytä vain yksivaiheista TD-virhettä (pieni varianssi, suuri poikkeama)
- λ = 1: käytä täydellisiä Monte Carlon palautuksia (suuri varianssi, pieni poikkeama)
- λ = 0,95: hyvä tasapaino molempien välillä!

**Tosielämän esimerkki:** Matkan suunnittelu.
- λ=0: katso vain seuraavat 5 mailia (turvallista, mutta pikakuvake saattaa jäädä huomaamatta myöhemmin)
- λ=1: ota huomioon koko 500 mailin matka (lisätietoa, mutta hyvin epävarmaa)
- λ=0,95: katso kauas eteenpäin, mutta paina läheisiä teitä enemmän ← paras tasapaino!

---

## Useita aikakausia: Datan tehokas uudelleenkäyttö

Kerättyään erän kokemusta (käyttöönotto), REINFORCE heittää sen pois YHDEN päivityksen jälkeen.

PPO käyttää kutakin erää uudelleen **K aikakaudella** (yleensä 4–10 läpikulkua samojen tietojen kautta):

```
Collect 512 steps × 4 environments = 2048 transitions
Epoch 1: 32 minibatches × update each
Epoch 2: shuffle, 32 more minibatches × update each
Epoch 3: ...
Epoch 4: ...
```

**Mikä on "minierä"?** Päivittäminen kaikilla 2048 siirtymällä kerralla on hidasta ja
muisti nälkäinen; yhden siirtymän päivittäminen kerrallaan on meluisaa. **minierä** on pieni
pala välillä — tässä 2048 ÷ 32 = **64 siirtymää minierää kohden**. Laskemme yhden
gradienttiaskel per minierä, joten jokainen aikakausi suorittaa 32 pientä, vakaata päivitystä sen sijaan, että
1 valtava. (Tämä on sama minieräidea, jota käytetään kaikkialla syväoppimisessa - katso
[mini-erä gradienttilasku](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Mini-batch_gradient_descent).)

Leikkaus varmistaa, että nämä useat siirrot eivät ylitä – ilman leikkausta, useita
aikakaudet tuhoaisivat käytännön työntämällä sitä liian pitkälle!

**Tosielämän esimerkki:** Oppilaalla on 30 harjoitustehtävää.
- VAHVISTA: tee jokainen ongelma kerran, opi vähän, heitä ne pois
- PPO: tee jokainen tehtävä 4 kertaa (eri kulmat joka kerta), leikkaa muutokset
  niin et muista vääriä kuvioita

---

## Täysi PPO-tappio

```
L = L_CLIP - c₁ · L_entropy + c₂ · L_critic

L_CLIP    = clipped policy gradient
L_entropy = entropy bonus (encourages exploration)  
L_critic  = MSE between V(s) and returns
```

Tyypilliset kertoimet: c₁ = 0,01 (entropia), c₂ = 0,5 (kriittinen)

**Kaksi termiä, jotka kannattaa purkaa:**

- **Policy gradient** — "näyttelijä" puolet tappiosta. Se käyttää gradienttisignaalia
  työnnä käytäntöä kohti toimia, joilla on suurempi hyöty ja pois toimista, joilla on pienempi hyöty
  etu. Tämä on sama ydinidea, joka esiteltiin REINFORCEssa – katso [REINFORCE
  walkthrough](./reinforce_cartpole_explained.md#the-old-way-vs-the-new-way)
  intuitio. PPO vain lisää leikkauskääreen ympärilleen.
- **MSE (Mean Squared Error)** – "kriittinen" puolet tappiosta. Kriitikot V(s) ennustaa
  odotettu tuotto osavaltiosta; vertaamme sen ennustetta todelliseen tuottoon ja
  neliöi eron: `MSE = mean((V(s) - return)²)`. Neliöinti rankaisee suuria virheitä
  enemmän kuin pieniä ja antaa tasaisen, erottuvan signaalin harjoitteluun. (Vakio
  regressiohäviö – katso [keskimääräinen neliövirhe](https://en.wikipedia.org/wiki/Mean_squared_error).)

---

## Tulokset

```
Update  200 | Avg reward: ~120
Update  400 | Avg reward: ~280
Update  800 | Avg reward: ~280-300
```

CartPolen PPO-arvo parantuu tasaisesti, mutta se on yleensä tasanteella noin 280-300.
(**tasanko** tarkoittaa, että oppimiskäyrä tasoittuu – palkinto lakkaa paranemasta jopa harjoittelun aikana
jatkuu. Käytäntö on löytänyt paikallisesti hyvän strategian, mutta se ei edisty enempää.)
Tämä on itse asiassa odotettavissa - PPO on suunniteltu kovempiin, pidempijaksoisiin ympäristöihin.

Mielenkiintoinen havainto: **REINFORCE ratkaisi CartPolen nopeammin!** (500 avg vs 300 avg)

Miksi? CartPole-jaksot ovat lyhyitä (≤500 askelta), joten REINFORCE:n tarkat tuotot ovat erittäin
tarkka. PPO:n bootstrapped arviot lisäävät tarpeetonta monimutkaisuutta. PPO todella loistaa
ympäristöissä, joissa täysien jaksojen odottaminen on epäkäytännöllistä (kuten BipedalWalker).

**Mikä on "BipedalWalker"?** BipedalWalker (erityisesti `BipedalWalker-v3` sisään
[Gymnasium](https://gymnasium.farama.org/environments/box2d/bipedal_walker/)) on a
klassinen RL-ympäristö: 2-jalkainen robotti, jonka on opittava kävelemään eteenpäin
epätasaisessa maastossa putoamatta. Toisin kuin CartPolen kaksi erillistä toimintaa
(VASEN / OIKEA), BipedalWalkerilla on **jatkuvat** toiminnot – neljä vääntömomenttiarvoa, yksi
jokainen jalkanivel, jokainen reaaliluku [-1, 1]. Jaksot voivat kestää tuhansia askeleita,
joka on juuri se järjestelmä, jossa PPO:n tietojen tehokkuus ja vakaus kannattavat.

---

## Keskeiset yhtälöt

```
Ratio:      r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
Clip loss:  L_CLIP = E[min(r_t A_t, clip(r_t, 1-ε, 1+ε) · A_t)]
GAE:        A_t = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}
```

---

## Avaimet takeawayt

| käsite | Pelkkää englantia |
|---------|---------------|
| **Suhde r(θ)** | Kuinka paljon tämän toiminnon käytäntö muuttui |
| **Clip ε** | Turvallisuusraja – älä muuta käytäntöä enempää |
| **GAE** | Älykäs tapa arvioida etuja katsomalla eteenpäin useita vaiheita |
| **Tietotehokkuus** | Jokainen julkaisu kerätään useista rinnakkaisista ympäristöistä (koristeellinen, vakaa kokemus) ja käytetään sitten uudelleen K jaksoissa minieräpäivityksiä – leikkaaminen pitää nämä toistot turvassa. |

---

## Mitä seuraavaksi?

Toistaiseksi kaikissa ympäristöissämme on **erillisiä** toimintoja (työnnä VASEN tai OIKEA).

Todellisten robottien on ohjattava **jatkuvia** toimintoja - kuten "kohdistavat täsmälleen 0,73 newtonin voimaa".

`ppo_continuous.py` laajentaa PPO:n jatkuviin toimiin käyttämällä **Gaussin käytäntöä**,
ja testaa sitä paljon kovemmassa BipedalWalker-v3-ympäristössä!
