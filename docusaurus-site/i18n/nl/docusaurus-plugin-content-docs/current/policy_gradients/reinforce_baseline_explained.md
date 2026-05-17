# VERSTERK met Baseline: door het lawaai heen

## Het probleem met gewone REINFORCE

Stel je voor dat je een student bent en probeert te beslissen of je antwoord op een toets goed was.

**Slechte feedback:** "Je hebt 7 punten!"

Is 7 goed? Als het maximum 10 is, ja! Als alle anderen er 9 hebben, nee! Zonder context,
u weet niet of u uw antwoordstijl moet veranderen.

Dit is precies het probleem met REINFORCE: het gebruikt **ruwe rendementen** (G_t) om te evalueren
acties. Een totaalrendementscore van 200 punten kan, afhankelijk van de situatie, verbazingwekkend of verschrikkelijk zijn.

---

## Voer de basislijn in

Een **basislijn** b(s) is een referentiepunt: "Welke beloning **verwacht** ik in deze situatie?"

In plaats van te vragen "Was deze actie goed?", vragen we:

> **"Was deze actie beter dan wat ik normaal zou verwachten?"**

```
Old signal: update ∝ G_t
New signal: update ∝ (G_t - b(s_t))
```

**Voorbeeld uit de praktijk:** Je scoorde 85 op een wiskundetoets.
- Als het klasgemiddelde 60 is → was je antwoord **25 punten boven het gemiddelde** → geweldig!
- Als het klasgemiddelde 90 is → was je antwoord **5 punten onder het gemiddelde** → werk nodig!

Het **voordeel** (G_t - b(s)) is positief als je het beter deed dan verwacht en
negatief als je het slechter deed. Dit is een veel schoner leersignaal!

---

## Wat is de basislijn?

De natuurlijke basislijn is de **waardefunctie V(s)**:

> V(s) = "Verwachte totale beloning als ik in staat ben en mijn huidige beleid speel"

Dit leren we met een apart **Waardenetwerk** (ook wel het basisnetwerk of criticus genoemd):

```
State  →  [128 neurons]  →  [128 neurons]  →  V(s)   (single number)
```

Voor elke staat die de agent bezoekt, voorspelt V(s) het verwachte rendement. Als de werkelijke
return G_t is hoger dan V(s), de actie was beter dan verwacht!

---

## Twee netwerken die samen leren

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

**Voorbeeld uit de praktijk:** Twee vrienden gaan samen naar een restaurant.

- Vriend 1 (Value Network): "Ik voorspel dat dit gerecht een 7/10 zal zijn"
- Vriend 2 (beleidsnetwerk): u probeert het gerecht en beoordeelt het met 9/10
- Voordeel = 9 - 7 = +2 → "Dat was beter dan verwacht! Bestel hem opnieuw!"

Bij een volgend bezoek werkt Vriend 1 zijn voorspelling bij dichter bij 9/10.
Vriend 2 zal dat gerecht de volgende keer eerder bestellen.

---

## Waarom vermindert dit de variantie?

**Wiskundig bewijs (intuïtie):**

Zonder basislijn: `gradient ∝ ∇log π(a|s) × G_t`

De G_t-waarden variëren sterk van aflevering tot aflevering:
`` `
Episode 1: G = [45, 44, 43, ..., 1]   (medium game)
Episode 2: G = [500, 499, ..., 1]      (great game!)
Episode 3: G = [12, 11, ..., 1]        (terrible game)
` ``

De gradiëntschattingen lopen enorm op omdat G_t groot en luidruchtig is.

Met basislijn: `gradient ∝ ∇log π(a|s) × (G_t - V(s_t))`

Het voordeel (G_t - V(s_t)) is veel kleiner en gecentreerd rond nul:
`` `
Episode 1: advantage ≈ [-2, +1, -3, ..., 0]   (small, centered)
Episode 2: advantage ≈ [+10, +8, ..., +3]      (this game WAS great)
Episode 3: advantage ≈ [-5, -6, ..., -2]       (this game WAS bad)
` ``

**Voorbeeld uit de praktijk:** Je hardloopsnelheid meten.
- Zonder basislijn: "Ik rende 8 km/u" (zinloos zonder context)
- Met baseline: "Ik liep 2 km/u SNELLER dan mijn gemiddelde" (duidelijk goed!)

Het voordeel is altijd een vergelijking: het is van nature kleiner en stabieler.

---

## Cruciaal: geen vooringenomenheid!

De basislijn verandert niets aan WAT het algoritme leert, alleen HOE SNEL en STABIEL het leert.

**Waarom?** Omdat het verwachte voordeel in verwachting altijd 0 is:

> E[G_t - V(s_t)] = E[G_t] - V(s_t) = V(s_t) - V(s_t) = 0

Elke b(s) die niet afhankelijk is van de actie, werkt als een geldige basislijn!

**Voorbeeld uit de praktijk:** Beoordeling op basis van een curve verandert niet wie het beste heeft gepresteerd, alleen wel
maakt de scores gemakkelijker te interpreteren. De ranglijst blijft hetzelfde; alleen de schaal verandert.

---

## De resultaten

```
No baseline  — Final 100-ep avg: 500.0, grad var: 599.3
With baseline — Final 100-ep avg: 491.4, grad var: 578.8
```

Beide methoden bereiken bijna perfecte prestaties op CartPole, maar let op:
1. De **gradiëntvariantie** is meetbaar (grafiek rechts toont de variantie over training)
2. Met de basislijn bereikt de agent **betrouwbaarder** hoge prestaties – minder crashes en een lage beloning tijdens de training

De variantiereductie is dramatischer in hardere omgevingen (LunarLander, MuJoCo).

---

## Belangrijkste vergelijkingen

```
Baseline value:   V(s) ← V(s) + α(G_t - V(s))   [minimize MSE]
Policy gradient:  θ ← θ + α ∇log π(a_t|s_t) · (G_t - V(s_t))
Advantage:        A_t = G_t - V(s_t)
```

---

## Belangrijkste afhaalrestaurants

| Concept | Plain English |
|---------|---------------|
| **Basislijnb(en)** | Verwachte beloning in staten – ons referentiepunt |
| **Voordeel A_t** | "Was deze actie beter dan verwacht?" |
| **Waardenetwerk** | Een neuraal net dat leert verwachte rendementen te voorspellen |
| **Variantiereductie** | Minder ruis in gradiëntschattingen → stabieler leren |
| **Onbevooroordeeld** | De basislijn verandert het doelbeleid gemiddeld niet; het maakt het leersignaal alleen minder luidruchtig en stabieler |

---

## Wat is het volgende?

De basislijn is eigenlijk het begin van iets veel krachtigers: **Actor-Critic**-methoden.

In plaats van V(s) pas aan het einde van een aflevering te berekenen, wordt de Actor-Critic bijgewerkt
V(s) bij elke afzonderlijke stap met behulp van **Temporal Difference**-leren. Hierdoor worden updates uitgevoerd
veel sneller en stelt de agent in staat te leren van onvolledige afleveringen!

Zie `a2c_lunarlander.py` voor de volledige Actor-Critic-implementatie.