# A2C: De acteur en de criticus werken samen

## Het grote idee

REINFORCE wacht tot het spel helemaal afgelopen is voordat het wordt bijgewerkt. Dat is zoiets als een coach die
kijkt een hele voetbalwedstrijd in stilte en geeft aan het eind alle feedback.

**A2C (Advantage Actor-Critic)** geeft feedback TIJDENS het spel — om de paar stappen geeft de coach
pauzeert om te zeggen: "Die pass was geweldig! Die tackle was slecht!"

Dit is veel sneller en efficiënter.

---

## Maak kennis met de twee karakters

> **Wat is LunarLander?** In dit document gebruiken we de **LunarLander**-omgeving: een natuurkundige simulatie waarbij je een klein ruimtevaartuig bestuurt en dit zachtjes moet laten landen op een doelplatform op de maan met behulp van drie motoren (links, hoofd en rechts). Het is een standaard benchmark voor versterkend leren, beschikbaar in de Gymnasium-bibliotheek.

### De acteur 🎭
De **Actor** is het beleid: hij beslist welke actie moet worden ondernomen.

> "Ik verkeer in deze toestand. Moet ik de linkermotor of de rechtermotor afvuren?"

**Voorbeeld uit de praktijk:** De *bestuurder* van een auto die aan het stuur draait en de pedalen indrukt.

### De criticus 🎬
De **Criticus** schat hoe goed de huidige situatie is – de waarde V(s).

> "In DEZE staat zijn is ongeveer +150 punten van de totale toekomstige beloning waard."

**Voorbeeld uit de praktijk:** De *navigator* zit naast de bestuurder en zegt: "We zijn op de goede weg —
verwacht binnen 30 minuten aan te komen." of "We komen in het verkeer terecht, dit gaat langzaam."

### Ze delen een brein
In onze implementatie gebruiken beide dezelfde **dezelfde neurale netwerkbackbone**:

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

- **ReLU** (Rectified Linear Unit): een activeringsfunctie die na elke laag wordt toegepast — deze voert `max(0, x)` uit, waarbij positieve waarden behouden blijven en negatieve waarden worden geëlimineerd. Hierdoor kan het netwerk niet-lineaire patronen leren.
- **actieproblemen**: de waarschijnlijkheid dat elk van de 4 acties wordt uitgevoerd. De acteur neemt een monster uit deze distributie en kiest bij elke stap een actie.

**Voorbeeld uit de praktijk:** Eén brein, twee banen – zoals een taxichauffeur die allebei rijdt (acteur)
EN weet of de route goed is (criticus). Het delen van de hersenen betekent sneller leren!

---

## Het voordeel: was dit beter dan verwacht?

Net als REINFORCE met de basislijn berekent A2C het **Voordeel**:

> A(s, a) = "Werkelijk resultaat" − "Wat we hadden verwacht"

Maar hier komt het "werkelijke resultaat" van de **n-step bootstrap** van de Criticus (**bootstrapping** = gebruik maken van de eigen voorspelling V(s) van de Criticus om de waarde van toekomstige stappen te benaderen, in plaats van te wachten tot de daadwerkelijke aflevering eindigt - zoals het schatten van je eindexamenscore halverwege het semester met behulp van je huidige cijfer):

```
Actual TD return: r_t + γ · r_{t+1} + γ² · r_{t+2} + ... + γⁿ · V(s_{t+n})
Advantage A_t = TD return - V(s_t)
```

**Voorbeeld uit de praktijk:** Je verwacht deze wedstrijd 3 doelpunten te scoren (V(s)). Als je 5 doelpunten scoort,
uw voordeel is +2. Als je 1 doelpunt scoort, is je voordeel -2.

Positief voordeel → "die actie hielp meer dan verwacht → doe het meer!"
Negatief voordeel → "die actie hielp minder dan verwacht → doe het minder!"

---

## Waarom meerdere parallelle omgevingen gebruiken?

Onze A2C gebruikt **8 exemplaren** van LunarLander die tegelijkertijd draait!

**Waarom?** Omdat ervaringen uit één enkele omgeving **gecorreleerd** zijn: één stap
volgt de vorige stap nauwkeurig. Deze correlatie zorgt ervoor dat het neurale netwerk gaat nadenken
patronen zijn betrouwbaarder dan ze zijn.

Met 8 omgevingen geeft elke stap 8 onafhankelijke ervaringen uit zeer verschillende situaties.
Dit verbreekt de correlatie en stabiliseert de training dramatisch.

**Voorbeeld uit de praktijk:** Om meer te weten te komen over het weer, wat beter is:
- Acht opeenvolgende uren naar één stad kijken (gecorreleerd: als het om 14.00 uur zonnig was, is het waarschijnlijk om 15.00 uur zonnig)
- Tegelijkertijd kijken naar 8 steden (gecorreleerd – verschillende weerpatronen, meer informatie!)

```
Environment 1: [landed on moon, fire left, crash, reset...]
Environment 2: [falling too fast, fire both, hover, land...]
Environment 3: [tilting right, fire right, stabilize, land...]
...
Environment 8: [drifting left, fire left, steady, ...]
```

Alle 8 updaten het netwerk tegelijkertijd - 8× meer diverse ervaring per update!

---

## N-Step-updates: wacht niet tot het spel is afgelopen

REINFORCE wacht op een volledige aflevering (kan 1000 stappen zijn!).

A2C wordt elke **n_steps = 128 stappen** bijgewerkt:

```
Play 128 steps across 8 environments
    → Get 128 × 8 = 1024 experience tuples
    → Compute advantages and returns
    → Update the Actor and Critic
    → Play 128 more steps...
```

**Voorbeeld uit de praktijk:** Een student die studeert voor een examen.
- VERSTERK stijl: Lees het hele leerboek en doe DAN oefentoetsen.
- A2C-stijl: lees 10 pagina's, doe een quiz, lees nog 10 pagina's, doe een quiz...

Vaker feedback = sneller leren!

---

## Drie verliezen gecombineerd

A2C traint met drie verliestermen tegelijk:

Een **verlies** is het getal dat de optimalisatie probeert te minimaliseren. Kleiner verlies betekent dat van het netwerk
Het huidige gedrag ligt dichter bij het trainingsdoel.

### 1. Acteurverlies (beleidsgradiënt)
Maak voordelige acties waarschijnlijker:
```
L_actor = -E[log π(a|s) · A(s,a)]
```
Indien A > 0: verhoog de waarschijnlijkheid van die actie
Indien A < 0: verklein de waarschijnlijkheid van die actie

### 2. Criticusverlies (waardefunctie MSE)
Waardevoorspellingen nauwkeuriger maken (**MSE** = Mean Squared Error: kwadraat van de voorspellingsfout en het gemiddelde ervan — kwadrateren bestraft grote fouten zwaarder dan kleine):
```
L_critic = E[(V(s) - return)²]
```
Zoals het trainen van elk **regressie**-model (regressie = voorspellen van een continu getal, hier de verwachte retour V(s)) – minimaliseer de voorspellingsfout.

### 3. Entropiebonus (verkenning)
Zorg ervoor dat het beleid niet te snel te zelfverzekerd wordt:
```
L_entropy = -H[π(·|s)] = E[log π(a|s)]
```
Hoge entropie = diverse actiekeuzes = verkenning
Lage entropie = zelfverzekerd, beperkte keuzes = uitbuiting

**Voorbeeld uit de praktijk:** De entropiebonus is als een leraar die zegt: "Raad niet zomaar een A bij elke
meerkeuzevraag! Probeer verschillende antwoorden, zodat je leert wat werkt."

```
Total loss = L_actor + 0.5 × L_critic - 0.01 × entropy
```

---

## LunarLander: een moeilijkere uitdaging

**LunarLander-v3** is een Gymnasium-omgeving (voorheen OpenAI Gym) — "v3" is het versienummer dat de derde revisie van deze omgeving aangeeft. De agent bestuurt een klein ruimtevaartuig dat veilig moet landen op een aangewezen plek op de maan. Het is veel moeilijker dan CartPole:
- 8-dimensionale toestandsruimte (positie, snelheid, hoek, beencontact, brandstof)
- 4 discrete acties (niets doen, links vuren, hoofd vuren, rechts vuren)
- Beloning: +100 voor landen, -100 voor crashen, kleine brandstofboetes

De trainingscurve laat een geleidelijke verbetering zien van zeer negatieve beloningen naar positieve beloningen.
A2C op LunarLander vereist aanzienlijke ervaring voordat de lander basisstabiliteit leert.

---

## Belangrijkste vergelijkingen

```
n-step return:  G_t = r_t + γ·r_{t+1} + ... + γⁿ·V(s_{t+n})
Advantage:      A_t = G_t - V(s_t)
Actor update:   θ_π ← θ_π - α ∇ L_actor
Critic update:  θ_V ← θ_V - α ∇ L_critic
```

---

## Belangrijkste afhaalrestaurants

| Concept | Plain English |
|---------|---------------|
| **Acteur** | Het beleid – beslist wat er moet gebeuren |
| **Criticus** | De waardefunctie: beoordeelt hoe goed de situatie is |
| **Voordeel** | "Was dit beter dan verwacht?" (werkelijk - verwacht) |
| **N-stap retour** | Kijk n stappen in de toekomst voordat je start met V(s) |
| **Parallelle omgevingen** | Meerdere omgevingen voor decorgerelateerde, diverse ervaringen |
| **Entropiebonus** | Moedigt de acteur aan om nieuwe dingen te blijven proberen |

---

## Wat is het volgende?

A2C is geweldig, maar heeft één grote zwakte: het actualiseert het beleid soms te agressief.
Eén enkele slechte update kan al het goede van een eerdere update vernietigen.

**PPO (Proximal Policy Optimization)** lost dit op met een slimme "veiligheidsclip" die dit voorkomt
elke enkele update om het beleid te veel te veranderen.

Zie `ppo_scratch.py` voor de PPO-implementatie!