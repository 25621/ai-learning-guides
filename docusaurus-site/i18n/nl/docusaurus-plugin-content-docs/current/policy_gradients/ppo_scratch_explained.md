# PPO: veilige en stabiele beleidsupdates

## Het probleem met A2C

Stel je voor dat je leert een bezemsteel op je vinger te balanceren. Na weken oefenen,
je kunt het 30 seconden volhouden!

Nu geeft je coach je het advies: "Leun je pols iets meer naar links."

**Goed advies → voorzichtig wisselen → nog 30 seconden balanceren ✓**

Maar wat als de coach overdreven reageert en zegt: "ONMIDDELLIJK NAAR LINKS!"
Je corrigeert teveel → bezemsteel valt → je bent weken aan voortgang kwijt.

Dit is het A2C-probleem: **grote gradiëntupdates kunnen een goed beleid vernietigen**.

**PPO (Proximal Policy Optimization)** is een veiligheidssysteem dat dit voorkomt.

---

## Het kernidee: blijf dicht bij wat werkte

De belangrijkste beperking van PPO:

> **"Verander het beleid niet te veel in één enkele update."**

Vóór een update hebben we het "oude" beleid π_old.
Na de update hebben we het "nieuwe" beleid π_new.

PPO meet hoeveel het beleid is veranderd met de **kansratio**:

```
r(θ) = π_new(a|s) / π_old(a|s)
```

- r = 1,0: beleid ongewijzigd
- r = 1,5: bij nieuw beleid is de kans 50% groter dat deze actie wordt ondernomen
- r = 0,5: bij nieuw beleid is de kans 50% kleiner dat deze actie wordt ondernomen

**Voorbeeld uit de praktijk:** Je bent een chef-kok die een recept aanpast.
- r = 1,0: dezelfde hoeveelheid zout als voorheen
- r = 2,0: verdubbel het zout – te extreem!
- r = 0,9: 10% minder zout – kleine, veilige verandering

---

## De kniptruc {#the-clipping-trick}

PPO knipt de verhouding af om binnen [1-ε, 1+ε] te blijven (meestal ε = 0,2):

```
L_CLIP = E[min(r(θ) · A,  clip(r(θ), 1-ε, 1+ε) · A)]
```

Laten we dit opsplitsen:

**Geval 1: De actie was GOED (A > 0)**

We willen deze actie vaker doen (r > 1). Maar we beperken hoeveel we verhogen:
`` `
if r > 1.2: clip to 1.2, no more incentive to push further
` ``
Dit voorkomt dat we TE ver in één richting zwaaien.

**Geval 2: De actie was SLECHT (A < 0)**

We willen deze actie minder doen (r < 1). Maar nogmaals, we sluiten af:
`` `
if r < 0.8: clip to 0.8, no more penalty for going further
` ``

**Visueel:**
```
ε = 0,2, dus het veilige-verhoudingsvenster is 0,8 tot 1,2.

GOEDE actie (A > 0): verhoog de actiekans, maar stop met het belonen ervan na 1.2
verhouding r: 0,6 0,8 1,0 1,2 1,4
stimulans: ↑ ↑ ↑ ↑ -
betekenis: te laag ok oude max geknipt

SLECHTE actie (A < 0): verlaag de actiekans, maar stop met het belonen ervan onder de 0,8
verhouding r: 0,6 0,8 1,0 1,2 1,4
prikkel: - ↓ ↓ ↓ ↓
betekenis: geknipt max oud ok te hoog
```

De `-` markeert het vlakke afgekapte gebied. In dat gebied wordt de waarschijnlijkheidsverhouding gelijk
extremer verbetert de doelstelling niet, dus PPO heeft geen extra prikkel om verder te gaan.

**Voorbeeld uit de praktijk:** De snelheidsbegrenzer van een auto. Je kunt accelereren, maar zodra je 120 km/u haalt,
de begrenzer treedt in werking en laat je niet sneller gaan. Het houdt je veilig zonder te stoppen
jij van verhuizen.

---

## Waarom dit catastrofale updates voorkomt

Er is sprake van een **catastrofale update** wanneer één grote beleidswijziging alles volledig vernietigt
agent heeft geleerd: uren training zijn in één enkele gradiëntstap verstreken.

Zonder clippen: één grote gradiëntstap zou het beleid drastisch kunnen veranderen.
Met clippen: de gradiënt is nul buiten [1-ε, 1+ε], dus het beleid kan slechts een klein beetje per stap bewegen.

**Voorbeeld uit de praktijk:** Een goede chirurg maakt kleine, precieze sneden – geen grote, ingrijpende sneden.
PPO is de "zorgvuldige chirurg" van RL-optimizers.

---

## GAE: Slimmere voordeelschattingen {#gae-smarter-advantage-estimates}

PPO gebruikt **Generalized Advantage Estimation (GAE)** om het voordeel te berekenen:

```
δ_t = r_t + γ · V(s_{t+1}) - V(s_t)          (TD error)
A_t = δ_t + γλ · δ_{t+1} + (γλ)² · δ_{t+2} + ...
```

GAE heeft een parameter λ (lambda):
- λ = 0: gebruik alleen eenstaps TD-fout (lage variantie, hoge bias)
- λ = 1: gebruik volledige Monte Carlo-rendementen (hoge variantie, lage bias)
- λ = 0,95: een goede balans tussen beide!

**Voorbeeld uit de praktijk:** Een roadtrip plannen.
- λ=0: kijk alleen naar de volgende 5 mijl (veilig, maar mis mogelijk later een kortere weg)
- λ=1: overweeg de volledige reis van 500 mijl (meer info, maar zeer onzeker)
- λ=0,95: kijk ver vooruit, maar belast nabijgelegen wegen zwaarder ← beste balans!

---

## Meerdere tijdperken: gegevens efficiënt hergebruiken

Na het verzamelen van een heleboel ervaring (uitrol), gooit REINFORCE deze weg na EEN update.

PPO hergebruikt elke batch voor **K-epochs** (meestal passeren 4-10 dezelfde gegevens):

```
Collect 512 steps × 4 environments = 2048 transitions
Epoch 1: 32 minibatches × update each
Epoch 2: shuffle, 32 more minibatches × update each
Epoch 3: ...
Epoch 4: ...
```

**Wat is een "minibatch"?** Updaten met alle 2048-overgangen tegelijk is langzaam en
geheugen-hongerig; het bijwerken van één overgang tegelijk is luidruchtig. Een **minibatch** is een kleine batch
stuk ertussen — hier 2048 ÷ 32 = **64 overgangen per minibatch**. Wij berekenen er één
gradiëntstap per minibatch, zodat elk tijdperk 32 kleine, stabiele updates uitvoert in plaats van
1 grote. (Dit is hetzelfde minibatch-idee dat overal bij deep learning wordt gebruikt – zie
[mini-batch gradiëntdaling](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Mini-batch_gradient_descent).)

De clipping zorgt ervoor dat deze meerdere passages niet voorbijschieten - zonder clipping, meerdere
tijdperken zouden het beleid vernietigen door het te ver door te voeren!

**Voorbeeld uit de praktijk:** Een leerling heeft 30 oefenproblemen.
- VERSTERK: doe elk probleem één keer, leer een beetje, gooi ze weg
- PPO: voer elk probleem 4 keer uit (elke keer vanuit verschillende hoeken), knip uw wijzigingen uit
  zodat je geen verkeerde patronen onthoudt

---

## Het volledige PPO-verlies

```
L = L_CLIP - c₁ · L_entropy + c₂ · L_critic

L_CLIP = afgekapt beleidsverloop
L_entropy = entropiebonus (moedigt verkenning aan)  
L_critic = MSE tussen V(s) en rendementen
```

Typische coëfficiënten: c₁ = 0,01 (entropie), c₂ = 0,5 (criticus)

**Twee termen die het uitpakken waard zijn:**

- **Beleidsgradiënt** — de "actor"-helft van het verlies. Het maakt gebruik van het gradiëntsignaal
  duw het beleid in de richting van acties met een hoger voordeel en weg van acties met een lager voordeel
  voordeel. Dit is hetzelfde kernidee geïntroduceerd in REINFORCE — zie de [REINFORCE
  walkthrough](./reinforce_cartpole_explained.md#the-old-way-vs-the-new-way) voor de
  intuïtie. PPO voegt er gewoon de knipverpakking omheen.
- **MSE (Mean Squared Error)** — de "kritische" helft van het verlies. De criticus V(s) voorspelt
  het verwachte rendement van een staat; we vergelijken de voorspelling met het daadwerkelijke rendement en
  vierkant het verschil: `MSE = mean((V(s) - return)²)` . Kwadrateren bestraft grote fouten
  meer dan kleine en geeft een vloeiend, differentieerbaar signaal voor training. (Standaard
  regressieverlies — zie [gemiddelde kwadratische fout](https://en.wikipedia.org/wiki/Mean_squared_error).)

---

## De resultaten

```
Update  200 | Avg reward: ~120
Update  400 | Avg reward: ~280
Update  800 | Avg reward: ~280-300
```

De PPO op CartPole vertoont een gestage verbetering, maar heeft de neiging rond de 280-300 te blijven hangen.
(Een **plateau** betekent dat de leercurve vlakker wordt – de beloning verbetert zelfs niet meer tijdens de training
gaat door. Het beleid heeft een lokaal goede strategie gevonden, maar boekt geen verdere vooruitgang.)
Dit wordt eigenlijk verwacht: PPO is ontworpen voor hardere omgevingen met langere afleveringen.

Een interessante observatie: **REINFORCE heeft CartPole sneller opgelost!** (500 gem. versus 300 gem.)

Waarom? CartPole-afleveringen zijn kort (≤500 stappen), dus het exacte rendement van REINFORCE is zeer hoog
nauwkeurig. De gebootste schattingen van PPO voegen onnodige complexiteit toe. PPO schittert echt
omgevingen waar wachten op volledige afleveringen onpraktisch is (zoals BipedalWalker).

**Wat is 'BipedalWalker'?** BipedalWalker (specifiek `BipedalWalker-v3` in
[Gymnasium](https://gymnasium.farama.org/environments/box2d/bipedal_walker/)) is een
klassieke benchmark RL-omgeving: een tweepotige robot die moet leren vooruit te lopen
over oneffen terrein zonder te vallen. In tegenstelling tot de twee afzonderlijke acties van CartPole
(LINKS / RECHTS), BipedalWalker heeft **continue** acties - vier koppelwaarden, één voor
elk beengewricht, elk een reëel getal in [-1, 1]. Afleveringen kunnen duizenden stappen duren,
en dat is precies het regime waarbij de data-efficiëntie en -stabiliteit van PPO vruchten afwerpt.

---

## Belangrijkste vergelijkingen

```
Ratio:      r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
Clip loss:  L_CLIP = E[min(r_t A_t, clip(r_t, 1-ε, 1+ε) · A_t)]
GAE:        A_t = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}
```

---

## Belangrijkste afhaalrestaurants

| Concept | Plain English |
|---------|---------------|
| **Verhouding r(θ)** | Hoeveel het beleid op deze actie is veranderd |
| **Clip ε** | De veiligheidsgrens: verander het beleid niet verder dan dit |
| **GAE** | Een slimme manier om de voordelen in te schatten door meerdere stappen vooruit te kijken |
| **Gegevensefficiëntie** | Elke uitrol wordt verzameld uit verschillende parallelle omgevingen (gecorreleerde, stabiele ervaring) en vervolgens hergebruikt voor K-epochs van minibatch-updates - door te knippen blijven deze herhalingspassen veilig |

---

## Wat is het volgende?

Tot nu toe hebben al onze omgevingen **discrete** acties (druk op LINKS of RECHTS).

Echte robots moeten **continue** acties controleren, zoals 'pas precies 0,73 Newton kracht toe'.

`ppo_continuous.py` breidt PPO uit naar continue acties met behulp van een **Gaussiaans beleid**,
en test het in de veel moeilijkere BipedalWalker-v3-omgeving!