# Een wereldmodel trainen: de agent leren dromen 🌍

## Wat is een ‘wereldmodel’?

Een **wereldmodel** is de *interne kopie van het universum* van de agent. Geef het een
staat en een actie, en het voorspelt wat er daarna zal gebeuren:

```
(state, action)  ──►  Neural Network  ──►  (next_state, reward)
```

Het is niet de echte wereld; het is een **simulator die de agent voor zichzelf heeft gebouwd**
kijken naar de werkelijkheid en leren deze na te bootsen.

Eenmaal getraind laat het model de agent 'wat-als'-vragen stellen zonder te nemen
elke echte actie:

> *"Als ik nu twee keer naar links en dan twee keer naar rechts duw, waar kom ik dan terecht? Zal de paal
> vallen?"*

De agent kan binnen zijn model honderd plannen overdenken in de tijd die daarvoor nodig is
om één echte stap te zetten. Dat is het hele punt.

---

## Een analogie uit het echte leven

Denk na over hoe *jij* een puzzel oplost. Je verplaatst niet elk stuk fysiek
in elke gleuf. Je **stel je voor** wat er gebeurt als stuk A hier komt. Als dat
mentale simulatie ziet er verkeerd uit, je wijst het af voordat je een vinger opsteekt.

Je brein heeft een aangeleerd wereldmodel – opgebouwd uit jarenlang zien hoe objecten
gedragen - waarmee u resultaten kunt simuleren voordat u zich vastlegt.

Andere voorbeelden:

- **Een schaker** stelt zich een zet voor die meerdere beurten vooruit is.
- **Een bestuurder** die denkt: "Als ik nu rem, heeft de auto erachter voldoende ruimte."
- **Een kind** stapelt blokken: "Als ik de grote er bovenop leg, zal de toren dat doen
  wiebelen." (Ze leerden dit model door eerder torens omver te werpen.)

In elk geval geldt: **een mentaal model + verbeeldingskracht = betere beslissingen met minder risico**.

---

## Hoe bouwt de agent zijn model op?

Het **kijkt gewoon**. Specifiek:

1. **Verzamel gegevens.** Laat elk beleid (zelfs willekeurig) interacteren met de werkelijkheid
   omgeving een tijdje. Bewaar elke overgang:
   `` `
   (state, action, reward, next_state)
   ` ` `
2. **Train a neural network** to predict ` next_state ` and ` beloning ` from
   ` (status, actie)`. Dit is begeleid leren: elke opgeslagen transitie is een
   gelabeld voorbeeld waarbij de invoer is "wat de agent zag en deed" en de
   label is "wat er feitelijk daarna gebeurde."
3. **Valideren.** Bewaar 10% van de gegevens en controleer de voorspellingen van het model
   tegen de echte. Een lage fout betekent dat het model de
   de **dynamiek** van de omgeving: hoe toestanden veranderen na acties.

De truc die we gebruiken: in plaats van `next_state` rechtstreeks te voorspellen, voorspel je de
**delta** `next_state − state` . De meeste natuurkunde is incrementeel ("de kar bewoog een
klein beetje"), en kleine doelen zijn vriendelijker voor neurale netwerken.

---

## Onze opstelling

| Choice | Value | Why |
|--------|-------|-----|
| Omgeving | `CartPole-v1` | 4D-status, 2 acties – eenvoudig te modelleren |
| Gegevens | 20.000 overgangen van een willekeurig beleid | Brede dekking van de staatsruimte |
| Netwerk | MLP, 2 × 128 ReLU verborgen | MLP = Multi-Layer Perceptron (standaard "vanille" neuraal netwerk). Twee verborgen lagen van 128 neuronen die gebruik maken van ReLU-activaties. Voldoende capaciteit, snel te trainen. |
| Verlies | MSE op `(delta_state, reward)` | MSE = Mean Squared Error (gemiddelde van kwadratische voorspellingsfouten). Standaard regressieverlies. |
| Optimalisatie | Adam, lr = 1e-3, 30 tijdperken | Adam = adaptieve optimizer (past de leersnelheid per parameter automatisch aan). Off-the-shelf betekent dat er geen speciale afstemming nodig is. |

De hele training eindigt in een paar seconden op de CPU.

---

## Hoe ziet ‘goed’ eruit?

Twee diagnostiek is van belang:

### 1. Nauwkeurigheid in één stap (validatie MSE)

Dit is "hoe goed voorspelt het model EEN stap in de toekomst?" Na 30
tijdperken zou u validatie MSE moeten zien in het bereik **1e-4 tot 1e-3**. Dat is
klein – poolhoeken en wagenposities zijn tot op enkele decimalen nauwkeurig.

### 2. **Samenstellingsfout** bij de implementatie van k-step

Dit is de *echte* test. Neem een staat, voer deze door het model en neem vervolgens
de voorspelling ervan en voer deze terug via het model — voor `k` stappen op rij.
De fout wordt groter omdat elke stap een beetje ruis toevoegt bovenop de vorige
voorspelling.

```
Step  1:  L2 error ≈ 0.01   (almost perfect)
Step  5:  L2 error ≈ 0.05
Step 10:  L2 error ≈ 0.15
Step 20:  L2 error ≈ 0.40   (visibly drifting)
```

*(L2-fout = Euclidische afstand tussen de voorspelde volgende toestand en de werkelijke toestand —
zie het als "hoe ver is de schatting van het model in de 4D-toestandsruimte?") *

**Waarom dit belangrijk is.** Als we 15 stappen vooruit plannen met het model, is de *exacte*
staat bij stap 15 zal verkeerd zijn – maar als de relatieve rangorde van ‘goede plannen
versus slechte plannen" blijft behouden, planning werkt nog steeds. (Dit is wat
 `model_based_planning.py` exploits.)

De grafiek in `outputs/world_model.png` toont beide diagnostiek naast elkaar: de
De curve voor trainingsverlies gaat mooi naar beneden op een logschaal, en de uitrolfout
curve gaat mooi omhoog.

---

## Waarom de *Delta* voorspellen?

Vergelijk twee manieren om hetzelfde probleem aan het netwerk te verwoorden:

| Target | Typical magnitude | Easy or hard? |
|--------|-----------------:|--------------|
| `next_state`        | 0–2,4 (wagenpositie) | Het netwerk moet de positie **en** de kleine verandering reproduceren |
| `next_state - state`| ~0,02            | Netwerk leert gewoon de kleine verandering |

Het voorspellen van de delta betekent ook: als het netwerk nullen oplevert (als ongetrainde beginner
netwerk vaak doet), is de voorspelling eenvoudigweg "niets verplaatst" - een verstandige, veilige standaard voor een single
tijdstap. Als u daarentegen de absolute `next_state` rechtstreeks zou voorspellen, zou dit in eerste instantie volledig willekeurige waardes opleveren, waardoor de vroege training zeer onstabiel zou zijn.

---

## Wat dit ons koopt

Een getraind wereldmodel vormt de basis voor:

- **Planning** — zoek naar ingebeelde actiescènes (zie
   `model_based_planning.py`).
- **Dyna-achtige augmentatie** — train een Q-netwerk op basis van ingebeelde gegevens
  vermenigvuldig de monsterefficiëntie.
- **Nieuwsgierigheid/verkenning**: bezoek geeft aan dat het model niet goed kan voorspellen.
- **Dreamer / World-Models papieren** — train een *beleid* volledig binnen de
  model zonder enige interactie in de echte wereld, afgezien van de initiële gegevensverzameling.

---

## Grenzen en waarschuwingen

- **Afwijking buiten distributie.** Het model kent alleen het deel van de wereld waarin het zich bevindt
  heeft gezien. Plan te agressief en je komt in regio's terecht waar het model nog nooit is geweest
  bezocht - voorspellingen daar zijn pure fantasie.
- **Samengestelde fout.** Plannen over lange **horizons** (veel stappen in de toekomst) is onbetrouwbaar vanwege de opeenstapeling van fouten, zoals de grafiek laat zien.
  Moderne systemen pakken dit aan door gebruik te maken van **probabilistische ensembles** (meerdere modellen trainen en controleren of ze het eens zijn, zoals in PETS of Dreamer), zodat de planner
  weet precies *hoe onzeker* het model bij elke stap is en kan risicovolle, onbekende paden vermijden.
- **Stochastische omgevingen.** Een standaard deterministische regressor voorspelt alleen het *gemiddelde* gemiddelde
  uitkomst en mist volledig de verspreiding van mogelijke uitkomsten. Complexe, realistische omgevingen vereisen probabilistisch inzicht
  modellen (zoals die met Gaussiaanse outputs, of **latente stochastische modellen** — netwerken die
  coderen van de wereldtoestand als een waarschijnlijkheidsverdeling in een gecomprimeerde ruimte,
  door ze echte willekeur te laten vastleggen in plaats van deze weg te middelen) om onzekerheid en willekeur accuraat weer te geven.

---

## Sleutelwoorden

| Term | Plain English |
|------|---------------|
| **Wereldmodel** | Een neuraal netwerk dat de omgeving nabootst |
| **Dynamiek** | De functie `(s, a) → s'` |
| **Beloningsmodel** | De functie `(s, a) → r` (vaak gebundeld) |
| **Voorspelling in één stap** | Wat het model oplevert vanuit een echte toestand |
| **Uitrol** | Herhaalde voorspellingen in één stap, waarbij de output weer wordt ingevoerd |
| **Samenstellingsfout** | Kleine fouten die tijdens een implementatie groter worden |

---

## Samenvatting van één zin

> **Een wereldmodel is een kleine neurale kopie van het universum die de agent kan maken
> raadplegen – en van binnen dromen – voordat je een echte actie riskeert.**

Vervolgens: `model_based_planning.py` zet dit model in voor daadwerkelijke besluitvorming.