# VERSTERK: Een robot leren betere keuzes te maken

## Wat proberen we te doen?

Stel je voor dat je een robot hebt die een videogame speelt. Elke seconde moet de robot kiezen:
**"Moet ik op de knop drukken of niet?"**

In plaats van elke situatie in een tabel te onthouden (zoals Q-learning), willen we dat de robot leert
een **recept** — een reeks regels die direct zeggen: "Voer in deze situatie deze actie uit."

Dit recept wordt een **beleid** (π, pi) genoemd. Ter versteviging
leren betekent π 'de regel voor het kiezen van acties'.

---

## De oude manier versus de nieuwe manier {#the-old-way-vs-the-new-way}

**Oude manier (Q-learning / DQN):** Ontdek hoe GOED elke actie is (Q-waarden) en kies vervolgens de beste.
> "Duwen naar LINKS heeft score 7, duwen naar RECHTS levert score 5 → duw naar LINKS!"

**Nieuwe manier (beleidsgradiënt):** Leer direct welke actie u moet KIEZEN.
> "Als de paal naar rechts kantelt, duw dan naar RECHTS met 80% kans, druk naar LINKS met 20% kans."
*(Het woord **Gradiënt** verwijst naar de wiskundige ‘stap’ die we nemen om deze kansen langzaam in de goede richting aan te passen.)*

**Voorbeeld uit de praktijk:** Leren fietsen.
- De oude manier: bereken de *exacte score* voor '5 graden naar links leunen' versus '7 graden naar links leunen'.
- De nieuwe manier: gewoon oefenen totdat je **lichaam** het leert – druk op de voet die goed voelt!

---

## Hoe werkt VERSTERKEN?

REINFORCE ziet de robot vervolgens een volledig spel van begin tot eind spelen (één **aflevering**).
vraagt: "Welke acties hebben tot een goede score geleid? Laten we dat vaker doen!"

### Stap voor stap

**1. Speel een aflevering**

De robot maakt keuzes en verzamelt ervaring:
```
Step 1: State = [pole tilting right] → Action = push RIGHT → Reward = +1
Step 2: State = [pole nearly balanced] → Action = push RIGHT → Reward = +1
Step 3: State = [pole tilting left] → Action = push LEFT  → Reward = +1
...
Step 47: State = [pole fell!] → Episode over
```

**2. Bereken rendement**

Bereken voor elke stap G_t — de **totale beloning vanaf dat moment**:
```
G_at_step_47 = 1
G_at_step_46 = 1 + 0.99 × 1 = 1.99
G_at_step_45 = 1 + 0.99 × 1.99 = 2.97
...
G_at_step_1  = 47 (roughly — higher return because it was from the start)
```

De γ = 0,99 **kortingsfactor** betekent dat beloningen ver in de toekomst iets minder tellen.

**Voorbeeld uit de praktijk:** Een gouden ster krijgen op schooldag 1 voelt spannender dan weten
je *kan* er een krijgen op dag 100. Toekomstige beloningen worden enigszins "verdisconteerd".

**3. Update het beleid**

Voor elke ondernomen actie:
> Als G_t HOOG was (die actie leidde tot een geweldig resultaat): **doe het meer!**
> Als G_t LAAG was (die actie leidde tot een slecht resultaat): **doe het minder!**

De wiskunde: `loss = -log_prob(action) × G_t`

Het nemen van de gradiënt en het bijwerken van het beleid is hetzelfde als tegen de robot zeggen:
*"Die actie die je hebt ondernomen bij stap 20? Je zou het de volgende keer 5% vaker moeten doen!"*

---

## Wat is een beleidsnetwerk?

In plaats van een tabel gebruiken we een **neuraal netwerk** om het beleid weer te geven.

```
Observation      Policy Network       Action Probabilities
[cart pos]  →   [128 neurons]  →  →  [push LEFT: 30%]
[cart speed] →  [128 neurons]        [push RIGHT: 70%]
[pole angle] →
[pole speed] →
```

Het netwerk geeft voor elke actie **waarschijnlijkheden** weer. Vervolgens monsteren we:
> Gooi een dobbelsteen → 1-30: druk op LINKS, 31-100: druk op RECHTS

**Voorbeeld uit de praktijk:** Een weer-app zegt: "70% kans op regen." Je WEET niet dat het zal regenen – jij
beslissen op basis van waarschijnlijkheid. De robot doet hetzelfde!

---

## Normalisatie: waarom we het gemiddelde aftrekken

Voordat we G_t gebruiken om bij te werken, normaliseren we:
```
G_normalized = (G - mean(G)) / std(G)
```

**Waarom?** Stel je voor dat alle beloningen positief zijn (wat in CartPole het geval is: altijd +1 per stap).
Zonder normalisatie ziet ELKE actie er "goed" uit en is het updatesignaal verwarrend.

Na normalisatie zijn sommige rendementen positief (boven het gemiddelde → pushen meer), en sommige zijn dat ook
negatief (onder het gemiddelde → minder pushen). Het signaal wordt veel schoner!

**Voorbeeld uit de praktijk:** Je leraar beoordeelt op basis van een curve. Als de gemiddelde score 70 is en je hebt
85, dat is geweldig! Maar als het gemiddelde 90 is en jij 85, dan is dat onder het gemiddelde. De ruwe score
alleen vertelt niet het hele verhaal.

---

## Het probleem: hoge variantie

REINFORCE heeft een grote zwakte: **variantie**. De retourzendingen G_t zijn erg luidruchtig.

**Voorbeeld uit de praktijk:** Stel je voor dat je een chef-kok beoordeelt door slechts EEN maaltijd uit elk restaurant te proeven.
Soms had de chef-kok een slechte dag, soms waren de ingrediënten niet goed. Eén maaltijd is niet genoeg
om betrouwbaar te weten of het restaurant goed is!

REINFORCE wacht op een VOLLEDIGE aflevering voordat de update wordt uitgevoerd. De ene aflevering kan heel veel geluk hebben, de andere
heel veel pech. De hellingen springen alle kanten op.

Dit is de reden waarom de leercurve (in de plot) er grillig uitziet:
- Sommige runs krijgen 500 (geweldig!)
- Sommige runs dalen tot 50 (verschrikkelijk!)

Ondanks het lawaai leert REINFORCE uiteindelijk, maar er is geduld voor nodig.

---

## De resultaten

```
Episode  100 | Avg reward (last 100):  43.1
Episode  200 | Avg reward (last 100): 193.9
Episode  500 | Avg reward (last 100): 408.4
Episode 1000 | Avg reward (last 100): 500.0  ← Solved!
```

De robot leert de paal in evenwicht te houden voor de maximale 500 stappen — OPGELOST!

Ondanks de variantieproblemen is REINFORCE op CartPole effectief omdat:
1. Afleveringen zijn kort (dus we krijgen er veel per trainingsrun)
2. Het optimale beleid is eenvoudig (meestal duwen in de richting waarin de paal kantelt)

---

## Belangrijkste afhaalrestaurants

| Concept | Plain English |
|---------|---------------|
| **Beleid** | Het recept van de robot voor het kiezen van acties |
| **Aflevering** | Eén volledig spel van begin tot eind |
| **Retour G_t** | Totale toekomstige beloning vanaf dit moment |
| **Korting γ** | Toekomstige beloningen tellen iets minder mee dan directe beloningen |
| **Normaliseren** | Trek het gemiddelde af, zodat het signaal schoner is |
| **Variantie** | Hoeveel de gradiëntschattingen rondspringen |

---

## Wat is het volgende?

De grote zwakte van REINFORCE is **variantie**. In het volgende script ( `reinforce_baseline.py` ),
we voegen een **basislijn** toe die deze ruis dramatisch vermindert — zonder te veranderen wat de
algoritme leert gemiddeld.

Het kernidee: in plaats van te vragen "was deze actie goed?" we vragen "was deze actie **beter dan
verwacht**?" Die kleine verandering maakt het leren veel stabieler.