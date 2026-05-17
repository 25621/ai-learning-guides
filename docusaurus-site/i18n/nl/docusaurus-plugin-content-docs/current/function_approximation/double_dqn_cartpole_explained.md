# Double DQN: het probleem van overmoed oplossen 🤔

## Het probleem: DQN denkt dat het beter is dan het is

Stel je voor dat je wordt gevraagd: "Wat is het beste restaurant van de stad?"

Je zou kunnen zeggen: "Pizza Palace is geweldig - het is absoluut een 10/10!" Maar dat heb je alleen gedaan
ben er twee keer geweest. Je weet eigenlijk niet of het *echt* een 10/10 is. Misschien wel
overschatten omdat je tijdens die twee bezoeken geluk hebt gehad met een goede pizza.

Hetzelfde probleem doet zich voor bij DQN: de agent **overschat Q-waarden**.

---

## Waarom overschat DQN?

Wanneer DQN het doel berekent, doet het het volgende:
> doel = beloning + γ × **max** Q(next_state)

De `max` is het probleem! Wanneer u het maximum van verschillende luidruchtige schattingen kiest, kunt u
kies bijna altijd degene met de grootste willekeurige fout (opwaartse bias).

**Voorbeeld uit de praktijk:** Je laat 5 vrienden de hoogte van een gebouw raden. Hun
de gissingen zijn: 40 m, 38 m, 45 m (geluksgok!), 39 m, 41 m. De werkelijke hoogte is 40 meter.
Als je `max(guesses)` = 45m gebruikt, zit je er ver naast! Het maximum aan luidruchtige gissingen
is vrijwel altijd een overschatting.

Gedurende duizenden updates blijft DQN trainen in de richting van deze overdreven opgeblazen doelen,
leren dat de dingen beter zijn dan ze in werkelijkheid zijn. Dit kan het leerproces vertragen of veroorzaken
de agent om overmoedige, slechte beslissingen te nemen.

---

## De dubbele DQN-fix

**Double DQN** (Hasselt et al., 2016) splitst de `max` in twee stappen:

**Stap 1 — Welke actie?** Gebruik het **online netwerk** om de beste actie te kiezen:
> beste_actie = argmax Q_online(volgende_staat)

**Stap 2 — Wat is de waarde ervan?** Gebruik het **doelnetwerk** om die actie te evalueren:
> doel = beloning + γ × Q_target(volgende_staat, beste_actie)

```
Vanilla DQN:   target = r + γ × max_a Q_target(s', a)
                                 ↑ same network picks AND evaluates → biased

Dubbele DQN: best_a = argmax_a Q_online(s', a) ← online keuzes
               doel = r + γ × Q_doel(en', beste_a) ← doel evalueert
                                 ↑ verschillende netwerken → minder bevooroordeeld
```

**Voorbeeld uit de praktijk:** In een sollicitatiegesprek laat je de sollicitant geen cijfer geven
eigen prestatietest (dat is het typische DQN-probleem!). In plaats daarvan de kandidaat
*nomineert* hun beste werk, en een *afzonderlijke* examinator beoordeelt het.
Twee verschillende mensen = eerlijkere beoordeling!

---

## Waarom helpt scheiding?

De twee netwerken (online en doel) hebben verschillende gewichten omdat het doel dat is
minder vaak bijgewerkt. Ze hebben verschillende ‘meningen’ over welke actie het beste is.

Wanneer ze het er niet mee eens zijn:
- Online zegt: "Actie A ziet er geweldig uit!"
- Doel zegt: "Eigenlijk is Actie A alleen maar oké - ongeveer 7 waard, niet 10"

Door de VALUE-schatting van het doelnetwerk te gebruiken voor de GEKOZEN actie van het online netwerk,
we krijgen een eerlijker, minder opgeblazen getal.

---

## Codeverschil: slechts één regel!

De enige codewijziging van vanille naar dubbele DQN zit in de doelberekening:

```python
# Vanilla DQN:
q_next = target_net(s_next).max(dim=1).values

# Dubbele DQN:
best_action = online_net(s_next).argmax(dim=1, keepdim=True) # kies met online
q_next = target_net(s_next).gather(1, best_actions) # evalueren met doel
```

Er veranderen slechts twee lijnen, maar de impact op de stabiliteit en nauwkeurigheid is aanzienlijk!

---

## Wat de vergelijking laat zien

Wanneer u `double_dqn_cartpole.py` uitvoert, ziet u twee plots:

**Plot 1: Leercurven**
- Zowel vanille als dubbele DQN zouden CartPole moeten oplossen
- Dubbele DQN convergeert vaak sneller en stabieler
- CartPole is zo eenvoudig dat het verschil bescheiden is; bij Atari is het dramatischer

**Plot 2: Q-waardeschattingen**
- Vanilla DQN: Q-waarden stijgen in de loop van de tijd (overschatting)
- Dubbele DQN: Q-waarden blijven bescheidener en nauwkeuriger

De grafiek van de overschatting van de Q-waarde is het belangrijkste inzicht: het laat het typische DQN-leren zien
opgeblazen waarden die uiteindelijk de prestaties schaden.

---

## Hoeveel beter is dubbele DQN?

| Metric | Vanilla DQN | Double DQN |
|--------|------------|------------|
| Nauwkeurigheid Q-waarde | Overschatting | Nauwkeuriger |
| Learning stability | More variance | Less variance |
| CartPole performance | Good | Slightly better |
| Atari-optreden (50 spellen) | Basislijn | +2,6× meer games op menselijk niveau |

Bij complexe Atari-spellen maakte Double DQN een veel groter verschil dan bij CartPole
(omdat Atari veel luidruchtiger schattingen van de Q-waarde heeft).

---

## De familie van DQN-verbeteringen

Double DQN is slechts een van de vele verbeteringen aan vanilla DQN. Het "Regenboog" -papier
(2017) combineerde 6 verbeteringen:

1. **Dubbele DQN** (overschatting opgelost) ← dit script!
2. **Geprioriteerde herhaling** (leer meer van verrassende ervaringen)
3. **Duelleringnetwerken** (scheid "hoe goed is deze staat?" van "wat is de beste actie?")
4. **Retouren in meerdere stappen** (kijk verder in de toekomst)
5. **Distributionele RL** (leer de volledige verdeling van de rendementen, niet alleen het gemiddelde)
6. **NoisyNets** (geleerde verkenning in plaats van [ε-greedy](../foundations/multi_armed_bandit_explained.md#the-epsilon-greedy-strategy))

Rainbow combineerde ze allemaal en behaalde de beste Atari-prestatie van zijn tijd!

---

## Sleutel Woordenschat

| Word | Meaning |
|------|---------|
| **Overschatting** | Q-waarden zijn hoger dan de werkelijke waarden (overdreven optimistisch) |
| **Dubbele DQN** | Gebruikt online netwerk voor actieselectie, doelnetwerk voor evaluatie |
| **Ontkoppeling** | Het scheiden van twee taken die door hetzelfde netwerk zijn uitgevoerd |
| **Vooroordeel** | Een systematische fout in één richting (altijd te hoog of altijd te laag) |
| **Regenboog** | Een DQN-variant die 6 verbeteringen combineert voor maximale prestaties |

---

## Samenvatting: Fase 3 Reis

Je hebt nu de volledige fase 3-voortgang voltooid:

| Algorithm | What it adds | Why it helps |
|-----------|-------------|-------------|
| Lineaire Q | Neuraal net → eenvoudige formule | Verwerkt continue toestanden |
| Naive DQN | Full neural network | Learns complex patterns |
| + Herspeelbuffer | Willekeurige geheugenbemonstering | Verbreekt correlaties |
| + Doelnetwerk | Bevroren kopie voor doelen | Stabiliseert de "bullseye" |
| Atari DQN | CNN + framestapeling | Leert van pixels! |
| Dubbele DQN | Apart selecteren/evalueren | Vermindert overschatting |

Elke stap loste een specifiek probleem op. Dat is hoe echt onderzoek werkt – één voorzichtigheid
verbetering in één keer!