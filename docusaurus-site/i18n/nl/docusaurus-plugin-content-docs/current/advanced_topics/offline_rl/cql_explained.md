# Conservatieve Q-Learning (CQL) 🛡️

## Wat is het?

Stel je voor dat je leert geld te beleggen door een gigantisch grootboek uit het verleden te lezen
aandelenhandel door andere mensen. Het grootboek bevat aankopen, verkopen en vasthouden -
maar **geen verslag van enige transactie die niemand daadwerkelijk heeft uitgevoerd**.

Stel je nu eens voor dat een overmoedige leerling naar het grootboek kijkt en zegt:
*"Wat als iemand elke maandag loten had gekocht? Dat zou het geval zijn geweest
het was een geweldige ruil!"*

Het probleem: **het grootboek bevat geen gegevens over loterijaankopen op maandag**, dus de
De student hallucineert alleen maar. Toch ziet die gehallucineerde handel er prachtig uit
papier, dus het ‘beleid’ van de student blijft het willen doen.

Dat hallucinatieprobleem is **distributieverschuiving**: een offline leerling
houdt van acties die de dataset nooit heeft getest, omdat er geen gegevens voor zijn
spreken het optimisme tegen. CQL is de remedie.

---

## Hoe Q-Learning offline misgaat

Het normale doel van Q-learning is:

```
target(s, a) = r + γ · max_{a'} Q(s', a')
```

Dat `max_{a'}` is het gevaar. Wanneer de dataset nooit actie registreerde `a'` 
in status `s'` *raadt* het netwerk slechts een Q-waarde – en neurale netwerken
hebben de neiging om Q te **overschatten** voor onzichtbare input. Het doel erft de
overschat, het netwerk leert dat grotere aantal te voorspellen, enzovoort
de volgende stap die we **extrapoleren** (projecteren nog verder dan wat dan ook).
data-ondersteuning) nog hoger. Het beleid jaagt op een spook.

Als je meer gegevens zou kunnen blijven verzamelen, zou dit zichzelf corrigeren (de
fantoomactie blijkt in werkelijkheid slecht te zijn). Maar **in offline RL jij
kan niet meer gegevens verzamelen.** Het fantoom is voor altijd.

---

## De truc van CQL

CQL (Kumar et al., 2020) voegt een boetetermijn toe aan het verlies:

```
cql_loss(s)  =  log Σ_a exp Q(s, a)   -   Q(s, a_dataset)
```

Twee stukken:

1. ** `log Σ_a exp Q(s, a)` ** (lees: *"log-sum-exp over alle acties"*) is een
   **zacht maximum** over alle acties — een soepel, differentieerbaar
   benadering van `max` waarbij elke actie in één keer wordt overwogen in plaats van
   moeilijk één winnaar selecteren. Door het te bestraffen, krimpen de Q-waarden
   **over de hele linie** (alle voorspellingen pushen
   gelijkmatig naar beneden) — vooral voor acties met de *hoogste* Q, en dat is precies
   waar de hallucinaties leven.
2. ** `- Q(s, a_dataset)` ** beloont hoge Q voor de actie van de dataset
   daadwerkelijk geregistreerd – waardoor de Q-waarden binnen de distributie worden beschermd tegen de
   krimp erboven.

Netto-effect: **Q wordt naar beneden getrokken bij niet-geziene acties, omhoog getrokken bij geziene acties
acties.** De geleerde Q wordt een *ondergrens* voor de ware Q. De
** `argmax` ** beleid (de regel die simpelweg de actie met de hoogste Q kiest)
stopt met het achtervolgen van fantomen.

Volledig verlies:

```
L  =  Bellman_MSE   +   α · cql_loss
```

(Waar ** `Bellman_MSE` ** de standaardfout is van normaal Q-learning,
het meten van de mate waarin de huidige inschatting van het netwerk het niet eens is met de zijne
toekomstige gok).

`α` is de conservatismeknop. Te klein → distributieverschuiving kruipt terug
in. Te groot → het middel is zo conservatief dat het nooit verder verbetert dan de
gegevens.

---

## Voorbeelden uit het echte leven

- **Conservatieve schaakcoach.** Je kunt alleen al van partijen leren
  gespeeld. Een roekeloze coach zegt: "Deze hypothetische zet zonder
  precedent zou briljant kunnen zijn!"  CQL is de coach die zegt: "We hebben geen
  gegevens daarover – laten we vasthouden aan zetten die echte spelers hebben geprobeerd."
- **Restaurantmenukeuzes.** Yelp-recensies hebben nooit betrekking op de
  items buiten het menu. Een naïef beleid zou de off-menu-items aanbevelen
  gebaseerd op gehallucineerde vijfsterrenbeoordelingen. CQL beveelt alleen aan wat is
  genoeg keer besteld om te vertrouwen.
- **Robot grijpt uit boomstammen.** De robot heeft een video van het grijpen van kopjes,
  flessen en boeken – maar nooit een mes. CQL weigert dit vol vertrouwen
  adviseren "pak het mes bij het mes."

---

## Wat onze code doet

Het script `cql.py`:

1. **Laadt de vier datasets** gebouwd door `d4rl_dataset.py` .
2. **Kiest `medium-replay` ** als trainingsset: dit is het meest realistisch
   (gemengde kwaliteit) en het meest schadelijk voor naïeve methoden.
3. **Traint drie agenten puur offline**, behalve onder identieke omstandigheden
   voor `α`:
   - `α = 0` → naïeve offline DQN (geen straf — de gebroken basislijn)
   - `α = 1.0` → milde CQL
   - `α = 5.0` → sterke CQL
4. **Evalueert elke 2500 gradiëntstappen** door gretig uit te rollen
   in de echte omgeving (10 afleveringen). Dit is het *enige* omgevingscontact;
   training zelf ziet nooit de omgeving.
5. **Plott leercurven** naar `outputs/cql.png` .

---

## Wat je moet zien

Een typische run drukt zoiets af als:

```
Final evaluation returns (avg over 10 episodes, greedy):
  naive offline DQN (alpha=0)         ->  ~30-150  (unstable; often crashes)
  CQL (alpha=1.0)                     ->  ~300-450
  CQL (alpha=5.0)                     ->  ~450-500
```

In de leercurvegrafiek:

- De **rode bocht** ( `α = 0` ) stijgt vroeg en valt vervolgens vaak **van een klif**
  zodra distributie-shift-hallucinaties het **Bellman-doelwit** infecteren
  (het nummer dat we gebruiken als het "juiste antwoord" bij het trainen van het Q-netwerk:
   `r + γ · max Q(s', ·)` ). Wanneer fantoom Q-waarden dat doel vervuilen,
  elke gradiëntstap maakt de zaken erger. Het **Bellman-verlies** (de MSE
  tussen de voorspelling van het Q-netwerk en het Bellman-doel) ziet er goed uit –
  dat is het **verraad** van het probleem: het netwerk is perfect
  consistent is met zijn eigen verkeerde overtuigingen, dus het verlies geeft geen waarschuwing.
- De **oranje curve** ( `α = 1.0` ) stijgt langzamer, maar **blijft omhoog**.
- De **groene curve** ( `α = 5.0` ) is de meest stabiele en meestal de beste.

Het Bellman-verliespaneel laat nog een veelzeggend verhaal zien: het verlies van de naïeve DQN kan blijven bestaan
klein, terwijl het beleid verschrikkelijk is, omdat het netwerk intern is
consistent met zijn eigen hallucinaties.

---

## Waar CQL in het veld zit

CQL was een *grote* deal omdat het een principiële, eenvoudige oplossing bood
distributieverschuiving. De afstamming:

```
DQN (online)
   │
   ▼
Naive offline DQN  ── breaks because of distribution shift
   │
   ▼
CQL (Kumar 2020)    ── add a conservative penalty: Q is a lower bound
   │
   ▼
IQL (Kostrikov 2021)  ── never query Q on un-seen actions in the first place
   │
   ▼
Decision Transformer (Chen 2021)  ── skip Q entirely, treat RL as sequence modelling
                                      (predict the *next action* given past states and
                                       a desired total return, exactly like an LLM
                                       predicts the next word)
```

Elke stap in deze lijn is een ander antwoord op dezelfde vraag:
**hoe voorkom ik dat ik mijn Q-netwerk vragen stel over dingen die het nog nooit heeft gezien?**

---

## Sleutelwoorden om te onthouden

| Word | Meaning |
|------|---------|
| **Distributieverschuiving** | Het getrainde beleid wil acties buiten de data |
| **Buiten distributie (OOD)** | Een (s, a) paar dat de dataset nooit heeft geregistreerd |
| **Echte vraag** | Het werkelijke verwachte toekomstige rendement voor het ondernemen van actie `a` in staat `s`, als we dit perfect konden meten |
| **Conservatieve Q** | Een aangeleerde Q-functie die probeert onder de echte Q te blijven in plaats van te veel te beloven |
| **Logsumexp** | Een vloeiende, differentieerbare benadering van `max` |
| **Alfa (α)** | De conservatismeknop van CQL – hoe moeilijk het is om Q onder druk te zetten bij OOD-acties |

---

## Samenvatting van één zin

> **CQL voegt een ‘pessimismestraf’ toe die hoge Q-waarden voor acties bestraft
> de dataset heeft het nooit geprobeerd, dus het beleid kan er niet verliefd op worden
> hallucinaties.**