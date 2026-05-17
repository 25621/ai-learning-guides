# PPO voor continue controle: BipedalWalker laten lopen

## Discrete versus continue acties

Tot nu toe heeft elke omgeving die we hebben opgelost **discrete** acties:
- CartPole: duw LINKS of duw RECHTS (2 keuzes)
- LunarLander: vuur niets / links / hoofd / rechts (4 keuzes)

Maar robots in de echte wereld hebben **continue** acties nodig:
- Een mensachtige robot: "hoe hard je elk gewricht moet duwen" (elke waarde van -1 tot +1)
- Een auto: "hoeveel je het stuur precies moet draaien" (elke hoek van -30° tot +30°)
- Een arm: "pas precies 2,3 Newton toe in deze richting"

**Voorbeeld uit de praktijk:** Typen op een toetsenbord = discreet (druk op A, B, C...).
Schrijven met een potlood = continu (beweeg de hand 2,3 cm naar rechts, druk 40 g kracht...).

---

## Het Gaussiaanse beleid voor continue acties

Voor doorlopende acties, in plaats van een categorische verdeling (kies uit N categorieën),
we gebruiken een **Normale (Gaussiaanse) verdeling**:

```
Action ~ Normal(μ, σ)
```

Waar:
- **μ (mu, mean)**: het centrum van de distributie – de actiewaarde waar het netwerk "naar streeft"
- **σ (sigma, standaarddeviatie)**: de spreiding — hoeveel willekeur/verkenning moet worden toegevoegd

```
        Probability
             │
        0.4 ─┤      ██████
             │    ████████████
        0.2 ─┤  ██████████████████
             │████████████████████████
             └──────────────────────── Action value
           -1  -0.5   0   0.5   1
                      ↑
                   mean μ
```

**Voorbeeld uit de praktijk:** Een ervaren boogschutter richt op het midden van het doel (μ).
Hun pijlen landen niet allemaal op precies dezelfde plek – er is enige spreiding (σ).
Terwijl ze oefenen, worden ze nauwkeuriger (σ neemt af) terwijl ze gecentreerd blijven op de roos.

---

## Ons Gaussiaanse acteur-criticusnetwerk

```
State (24 numbers) → [256 neurons] → [256 neurons] →
    ├── Actor: 4 mean values  (μ₁, μ₂, μ₃, μ₄)
    │          + 4 log_std params (shared across all states!)
    └── Critic: 1 value (V(s))
```

De `log_std` (logaritme van de **standaardafwijking** — een maatstaf voor spreiding of onzekerheid)
is een **leerbare parameter** — niet toestandsafhankelijk.
Dit houdt het simpel en laat de verkenning tijdens de training toch veranderen.

**Waarom log_std in plaats van std?** Standaardafwijking moet positief zijn. Het gebruik van `log_std` is toegestaan
het netwerk om een reëel getal (positief of negatief) uit te voeren, dan passen we toe
 `exp(log_std)` — de exponentiële functie, die het omgekeerde is van de logaritme — to
een gegarandeerd positieve standaard herstellen. Dit voorkomt dat de std ooit negatief of nul wordt.

---

## Berekeningslogwaarschijnlijkheid voor continue acties

Voor discrete acties: `log_prob = log(P(action=LEFT))`

Voor continue acties beschrijft de **Normale verdeling** een vloeiende klokvormige curve
rond het gemiddelde. Een enkele exacte waarde heeft een waarschijnlijkheid nul bij continue wiskunde, zo gebruiken we
de curvehoogte bij die waarde, de **pdf** genoemd (kansdichtheidsfunctie):
`` `
log_prob = Σᵢ log[Normal(μᵢ, σᵢ).pdf(aᵢ)]
` ``

`log` betekent natuurlijke logaritme. Het verandert kleine dichtheidswaarden in stabiele getallen die dat ook zijn
gemakkelijker voor neurale netwerken om te optimaliseren. We tellen alle actiedimensies op (4 voor
BipedalWalker), omdat de volledige actie één vector met vier getallen is.

**Voorbeeld uit de praktijk:** Wat is de kans dat het morgen precies 5,732...°C wordt?
Voor continu weer zou je naar de normale distributiecurve kijken en zien hoe hoog deze is
op dat exacte punt. Waarschijnlijker temperaturen (dichtbij het gemiddelde) hebben een grotere waarschijnlijkheid.

---

## BipedalWalker: een wandeluitdaging

BipedalWalker-v3 is een 2D-robot die moet leren lopen zonder te vallen:

```
          O (head)
         /│\
        / │ \
       /  │  \
      L   │   R   ← two legs, each with a knee joint
     / \  │  / \
    ●   ● │ ●   ●  ← 4 motors (hip/knee for each leg)
```

**Statusruimte (24 cijfers):**
- Romp: hoek, hoeksnelheid, horizontale snelheid, verticale snelheid (4 cijfers)
- Gewrichten: 4 motoren (2 heupen, 2 knieën) die elk zorgen voor hoek en snelheid, plus 2 grondcontactsensoren (één voor elk been) (10 cijfers)
- 10 LIDAR-bereiksensoren (afstandsmetingen die de grond voor zich zien) (10 cijfers)

**Actieruimte (4 doorlopende waarden, elk in [-1, 1]):**
De actiewaarden regelen het **koppel** (de rotatiekracht uitgeoefend door de motoren) voor precies 4 gewrichten (er worden geen acties rechtstreeks op de romp uitgeoefend):
- Been 1 Heupmoment, Been 1 Kniemoment, Been 2 Heupmoment, Been 2 Kniemoment

**Beloningen:**
- +300 voor het bereiken van het doel (rechterkant)
- -100 voor omvallen (de grond raken met lichaam)
- Kleine beloning per stap voorwaartse vooruitgang
- Kleine boete voor elk motorgebruik (beloningsefficiëntie)

**Opgelost wanneer:** Gemiddelde beloning > 300 over 100 afleveringen

---

## Belangrijkste verschil met discrete PPO

Alles is hetzelfde BEHALVE:

| | Discrete PPO | Continuous PPO |
|---|---|---|
| **Beleid** | Categorisch(logits) | Normaal(μ, σ) |
| **Steekproef** | actie = monster uit {0,1,...,N} | actie = μ + σ × ruis |
| **log_prob** | logboek P(actie=k) | Σ log Normaal(μᵢ, σᵢ).pdf(aᵢ) |
| **Klem** | Niet nodig | Klemacties tot [-1, 1] |

**Logits** zijn ruwe, niet-genormaliseerde scores voor afzonderlijke acties. Een categorisch beleid leidt tot bekeringen
ze omzetten in kansen met **softmax** — een functie die een willekeurige reeks getallen en nodig heeft
drukt ze samen in een geldige waarschijnlijkheidsverdeling (alle waarden positief, opgeteld tot 1).
Logits [2,0, 1,0, 0,5] worden bijvoorbeeld waarschijnlijkheden [0,59, 0,24, 0,17]. Continue PPO gebruikt **geen** softmax voor de actie zelf,
omdat de actie niet uit een vast menu wordt gekozen. In plaats daarvan geeft het beleid het gemiddelde weer
en standaarddeviatie van een normale verdeling, en bemonstert er vervolgens koppels met reële waarde uit.

**Clamp** betekent dat een waarde binnen een geldig bereik wordt geforceerd. De code gebruikt `action.clamp(-1, 1)`, dus de
omgeving ontvangt nooit een motorcommando buiten de toegestane grenzen.

**Clip** in PPO betekent iets anders: PPO knipt de waarschijnlijkheidsratio binnen het verlies,
zoals uitgelegd in de [PPO-knipsectie](./ppo_scratch_explained.md#the-clipping-trick).
Actieklemming beschermt de omgevingsinterface; PPO-clipping beschermt de beleidsupdate.

---

## Vanaf nul beginnen: wat de agent leert

**Vroege training (negatieve beloningen):** De robot zwaait willekeurig en valt onmiddellijk.
Elke aflevering eindigt binnen enkele seconden in een crash.

**Midden training:** De robot ontdekt dat bewegende benen afwisselend voorwaartse vooruitgang creëren.
Het begint kleine, ongemakkelijke stappen te zetten – de beloning wordt minder negatief.

**Late training:** Er ontstaat een soepele, efficiënte **loop**. Een gang is een herhaalde beweging
patroon, zoals afwisselende linker- en rechterstappen. De robot past zich dynamisch aan oneffen terrein aan door gebruik te maken van de LIDAR-sensoren om zijn stappen in realtime aan te passen.

**Voorbeeld uit de praktijk:** Een baby die leert lopen:
1. Valt onmiddellijk (negatieve beloning)
2. Zet een stap, valt (iets minder negatief)
3. Zet een paar stappen (kleine positieve beloning)
4. Loopt door de kamer (grote positieve beloning!)

---

## Waarom BipedalWalker PPO nodig heeft (niet VERSTERKEN)

- **BipedalWalker-afleveringen** kunnen maximaal 1600 stappen bevatten (veel langer dan CartPole!)
- **Beloningen zijn schaars** — beloningen voor vooruitgang zijn klein per stap
- **REINFORCE heeft** duizenden volledige afleveringen nodig om een bruikbaar signaal te krijgen

PPO's n-staps updates met [GAE (Generalized Advantage Estimation)](./ppo_scratch_explained.md#gae-smarter-advantage-estimates) laten de robot leren van onvolledige afleveringen:
> "Ook al viel ik na vijftig stappen, toch lieten die stappen ENIGE vooruitgang zien.
> Laat me een rendementsschatting van 50 stappen gebruiken in plaats van te wachten tot de aflevering is voltooid.

---

## Resultaten

Na 500 updates (≈ 1 miljoen omgevingsstappen):
- De robot maakt zichtbare vooruitgang van willekeurig zwaaien naar een voorwaartse beweging
- Consistente verbetering van de leercurve
- Volledige convergentie om> 300 te belonen vereist meer training (5-10 miljoen stappen)

De leercurve toont de karakteristieke "S-curve" van continue regeling:
1. Langzame initiële vooruitgang (leerstabiliteit)
2. Snelle verbetering (loopontdekking)
3. Geleidelijke verfijning (loopoptimalisatie)

---

## Belangrijkste afhaalrestaurants

| Concept | Plain English |
|---------|---------------|
| **Gaussiaans beleid** | In plaats van uit een menu te kiezen, werp een pijltje naar een reeks waarden |
| **μ (gemiddeld)** | Waar het beleid ‘beoogt’ |
| **σ (standaard)** | Hoeveel willekeur/exploratie het beleid gebruikt |
| **log_std als leerbare parameter** | Een globale verkenningssnelheid die wordt bijgewerkt door op gradiënt gebaseerde optimalisatie (gradiënt *stijging* bij beloning, of gelijkwaardige gradiënt *daling* bij het PPO-verlies) — net als elk ander netwerkgewicht |
| **Continu controle** | Controle van reële waarden (koppels, krachten, hoeken) |

---

## Wat is het volgende?

PPO heeft veel **hyperparameters**: instellingen die u kiest voordat de training begint (in tegenstelling tot
*parameters* zoals netwerkgewichten, die automatisch worden geleerd). Voorbeelden zijn onder meer
 `clip_eps` , leersnelheid, aantal tijdperken en batchgrootte.

Hoe gevoelig is PPO voor deze keuzes?  `ppo_hyperparams.py` voert experimenten uit
het systematisch variëren van elke hyperparameter en toont het effect op de leersnelheid en stabiliteit.