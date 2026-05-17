# Een geleerd model voor planning (MPC) gebruiken 🔮

## Het grote idee {#the-big-idea}

Je hebt een **wereldmodel** (een neuraal netwerk dat de toekomst voorspelt).
Wat nu?

Het meest directe gebruik is **planning**: vraag op elk moment aan het model "wat".
zou er gebeuren als ik *dit* plan probeerde? *dat* plan? *dat andere* plan?"
Voer vervolgens het plan uit dat het beste lijkt, maar **slechts de allereerste stap**.
Omdat het model niet perfect is, voeren we slechts één actie uit, observeren we de daadwerkelijke nieuwe staat vanuit de echte omgeving en plannen we vervolgens opnieuw vanaf het begin.

Deze truc heeft een naam: **Model Predictive Control** (MPC).

---

## Een analogie uit het echte leven {#a-real-life-analogy}

Je bent in een restaurant en kijkt naar het menu. Je verplicht je niet tot een vijfgangenmenu
bestel ter plekke - u bestelt het eerste gerecht en kijkt dan hoe vol u zich voelt
opnieuw beslissen over het dessert.

Of: je rijdt op een bochtige weg. Je legt de stuurinput niet vast
de volgende 30 seconden - je kijkt voortdurend vooruit, plant een paar seconden, neemt
de volgende stuuractie en maak een nieuwe planning.

Die **plan-ver/act-near/re-plan** lus is MPC.

---

## Hoe "willekeurig fotograferen" werkt {#how-random-shooting-works}

Er zijn meer geavanceerde planners, bijvoorbeeld:
- **CEM** (Cross-Entropy Method): verfijn iteratief een verdeling over plannen door elke ronde alleen de top-k te behouden.
- **MCTS** (Monte Carlo Tree Search): bouw een zoekboom geleid door simulatiestatistieken, gebruikt door AlphaGo en MuZero.
- **Op gradiënten gebaseerde planners**: differentieer de voorspellingen van het model met betrekking tot de acties en volg de gradiënt direct.

We gebruiken de eenvoudigste die werkt: **willekeurig fotograferen**.

```
Given the current state s:
    1. Sample N=200 random action sequences of length H=15.
    2. For each sequence, simulate it through the world model from s, summing
       a shaped reward at each step.   (200 dreams in parallel — fast!)
    3. Find the sequence with the highest predicted total reward.
    4. Execute that sequence's FIRST action in the real environment.
    5. Observe the real next state.  Discard the rest of the plan.
    6. Go to step 1 — re-plan from scratch.
```

200 plannen × 15 stappen = 3.000 ingebeelde transities per echte stap. De wereld
model voert ze allemaal uit in een enkele batch-voorwaartse doorgang van het neurale netwerk – meestal
een paar milliseconden.

---

## Waarom elke stap opnieuw plannen? {#why-re-plan-every-step}

Omdat het model imperfect is. Fouten worden groter tijdens een implementatie (zoals te zien is in het diagram dat is gegenereerd door `world_model.py` en is opgeslagen in `outputs/world_model.png`). Het plan bij stap 0 is alleen betrouwbaar voor de eerste paar zetten; bij stap 15 hallucineert het model. Wij vertrouwen dus alleen op de
**eerste zet** en vernieuw het plan vervolgens met de laatste echte status.

Dit is dezelfde reden waarom mensen geen schaakplan met 100 zetten schrijven en zich eraan houden
het – de omstandigheden veranderen, en hoe verder je raadt, hoe minder het overeenkomt
realiteit.

---

## Een rimpel: de beloning moet de planner iets vertellen {#a-wrinkle-the-reward-has-to-tell-the-planner-something}

In CartPole is de echte beloning `+1` voor elke stap totdat de paal valt. Het model
zal `+1, +1, +1, ...` voor bijna elk plan getrouw voorspellen, omdat
willekeurige plannen eindigen zelden snel binnen het model – en dat geldt ook voor elk plan
scoort hetzelfde. De planner heeft niets om uit te kiezen.

De oplossing: vervang de echte beloning door een **soepele proxy** tijdens de planning:

```python
reward_proxy(state) = 1
                    - |pole_angle| / 0.21          # pole upright?  (1=yes)
                    - 0.1 * |cart_position| / 2.4  # cart centred?  (1=yes)
```

Plannen die *zouden* eindigen met een omgevallen paal krijgen nu zichtbaar slechtere scores dan
plannen die overeind blijven. De planner kan ze rangschikken.

> **Les uit het echte leven.** Een vlak beloningssignaal — "je hebt nog een seconde overleefd" —
> is nutteloos voor kortetermijnplanning. Dichte, gevormde signalen helpen.

---

## Wat onze code doet {#what-our-code-does}

`model_based_planning.py`:

1. **Laadt** de wereldmodelgewichten die zijn opgeslagen door `world_model.py` . (Als ze dat zijn
   ontbreekt, er wordt er meteen één omgeschoold.)
2. **Voert 20 afleveringen** van MPC uit op de echte CartPole-v1.
3. **Voert ook 20 afleveringen uit** met een uniform willekeurig beleid als basislijn.
4. **Plots** naast elkaar en drukt gemiddelden af.

### Wat je zou moeten zien als je het uitvoert {#what-you-should-see-when-you-run-it}

| Beleid | Gemiddelde beloning (overleefde stappen) |
|--------|-----------------------------:|
| Willekeurig        | ~22 (typisch voor CartPole – hengel valt snel) |
| MPC (de onze)    | ~ 150-500 (varieert per zaad; veel afleveringen bijna 500) |
| Maximaal mogelijk  | 500 |

Die **5-25× verbetering** wordt bereikt zonder beleidsnetwerk, zonder waarde
functie, en geen verdere opleiding. Gewoon een wereldmodel + 200 dromen per stap.

De plot `outputs/model_based_planning.png` toont twee gekleurde balken per aflevering
- MPC is bijna altijd groter dan Random, en veel afleveringen raken de
Plafond met 500 treden.

---

## Strengths of Model-Based Planning {#strengths-of-model-based-planning}

- **Voorbeeld efficiënt.** Al het leren is gedaan op basis van één willekeurige batch
  overgangen. Er was geen verdere interactie met het milieu nodig om tot een bruikbaar beleid te komen.
- **Gemakkelijk opnieuw te targeten.** Wilt u de agent op een andere manier besturen? Verander de
  beloningsproxy - geen omscholing. (Probeer de karsnelheid voor de lol te maximaliseren.)
- **Interpreteerbaar.** U kunt de plannen inzien die de agent heeft overwogen, de
  voorspelde trajecten en de scores.

## Zwakke punten (en wat mensen eraan doen) {#weaknesses-and-what-people-do-about-them}

- **Willekeurig schieten is dom.** Het proeft blindelings plannen. Voor hoger
  afmetingen, schakel over naar **CEM** (Cross-Entropy Method - zie hierboven) of
  **iLQR** (Iteratieve lineaire-kwadratische regelaar, een klassieke optimale controle
  methode die het model als lokaal lineair benadert en oplost
  analytisch) of volledige **gradiëntgebaseerde** planner verbetert acties door gradiënten te volgen via een
  differentieerbaar model.
- **Samengestelde modelfout.** Lange horizonten verschuiven. Mensen gebruiken **probabilistisch
  ensembles** (verschillende modellen getraind op dezelfde gegevens, zoals in PETS, Chua et
  al. 2018), zodat de planner onenigheid kan opmerken en plannen kan bestraffen
  model is onzeker.
- **Echte beloning is wat we uiteindelijk willen.** Beloningsvorming helpt, maar voor
  Bij complexere taken leren mensen een **waardefunctie** getraind *in* de
  wereldmodel – een geleerde criticus die het rendement op de lange termijn van welk model dan ook schat
  staat zonder dat een handgemaakte proxy nodig is. Beide **Dreamer** (die traint
  een acteur-criticus geheel in latente verbeelding) en **MuZero** (welke paren
  MCTS met een leerwaardenetwerk) gebruiken dit idee.

---

## Hoe dit aansluit op moderne systemen {#how-this-connects-to-modern-systems}

Het exacte recept dat u zojuist hebt uitgevoerd — **geleerde dynamiek + planning** — is het
ruggengraat van enkele van de sterkste RL-systemen in modern AI-onderzoek:

- **MuZero** (DeepMind): combineert een geleerd wereldmodel met Monte Carlo Tree Search. Het beheerste Go, schaken, shogi en Atari zonder de regels vooraf nodig te hebben.
- **Dreamer / DreamerV3** (Hafner et al.): traint een beleid *binnen* een geleerde
  Wereldmodel in de latente ruimte (wat betekent dat het model onbewerkte beelden of toestanden comprimeert tot een compacte, abstracte representatie voordat de toekomst wordt voorspeld). Het behaalt state-of-the-art (top) prestaties in meer dan 100 benchmarks.
- **PETS / PlaNet / TD-MPC**: dit zijn families van algoritmen (onderzoekslijnen) die precies dit idee opschalen naar complexe continue controletaken zoals robotica.

Je hebt – in een paar honderd regels – het kleinste lid van die familie gebouwd.

---

## Sleutelwoorden {#key-words}

| Term | Plain English |
|------|---------------|
| **MPC**           | Modelleren van voorspellende controle: plan vooruit, handel één keer, plan opnieuw |
| **Willekeurige opname** | Scoor veel willekeurige plannen, kies de beste |
| **Horizon (H)**   | Met hoeveel stappen kijkt het plan vooruit |
| **N monsters**     | Hoeveel kandidaatplannen overwegen we per stap |
| ** Terugwijkende horizon ** | Herplanning bij elke stap in plaats van vast te houden aan één plan |
| **Beloningsproxy / vormgeving** | Een soepele surrogaatbeloning die de planner een nuttig signaal geeft om te optimaliseren |

---

## Samenvatting van één zin {#one-sentence-summary}

> **Als je eenmaal een wereldmodel hebt, is plannen niets anders dan ‘honderd toekomsten dromen’,
> kies de beste eerste stap, herhaal."**

Dat is het hele geheim van modelgebaseerde RL.