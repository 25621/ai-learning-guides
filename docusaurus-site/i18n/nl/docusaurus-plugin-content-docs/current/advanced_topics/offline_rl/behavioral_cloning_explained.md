# Gedragsklonen (BC) 🐒

## Wat is het?

Stel je voor dat je wilt leren tennissen. Je kijkt honderden uren
neem Wimbledon-wedstrijden op en **kopieer gewoon wat de spelers doen**. Jij
denk er niet over na of hun schot het *beste* schot was - je matcht gewoon
jouw lichaamspositie ten opzichte van die van hen en zwaai op dezelfde manier.

Dat is gedragsklonen. **Geen beloning. Geen planning. Gewoon imitatie.**

In RL-termen: neem de dataset van `(state, action)` paren en train a
neuraal netwerk om de actie van de staat te voorspellen, precies zoals een
beeldclassificatiemodel voorspelt kat-vs-hond. Het ‘label’ is wat dan ook
actie die de gegevensverzamelaar heeft ondernomen.

---

## Hoe het verschilt van "echte" offline RL

| Approach | Uses rewards? | Can beat the data? |
|----------|---------------|---------------------|
| **BC**   | ❌ nee         | ❌ nee — komt in het beste geval overeen met de gemiddelde gegevenskwaliteit |
| **CQL** (en vrienden) | ✅ ja | ✅ ja — kan goede overgangen uit gemengde gegevens samenstellen |

BC is de ‘begeleide leervisie’ van RL. Het is vaak ongelooflijk eenvoudig
verrassend sterk, en de universele basislijn. **Als een offline RL
algoritme kan BC niet verslaan op dezelfde dataset, het heeft niets gedaan.**

---

## Voorbeelden uit het echte leven

- **Leren rijden vanaf dashcambeelden.** Kijk naar de weg, voorspel
  de stuurwielhoek die de mens gebruikte. Twee historische voorbeelden:
  - **ALVINN (1989)** — de allereerste neurale netwerkdriver; een kleine 3-laags
    netwerk getraind op camera- en laserinvoer om een busje over snelwegen te sturen.
  - **NVIDIA PilotNet (2016)** — een moderne, diepgaande CNN waarop end-to-end is getraind
    dashcambeelden; puur door het aanhouden van de rijstrook en de basisbesturing
    het imiteren van menselijke chauffeurs, geen met de hand ontworpen regels.
- **Leerling die een meesterkok nadoet.** "Wat de chef ook doet, ik doe het."
  Werkt prima als de chef-kok geweldig is; levert een slechte chef-kok op als de chef-kok dat ook is
  slecht.
- **GitHub Copilot.** Automatisch aanvullen is getraind om te voorspellen "welke code
  zou een menselijk type het volgende zijn?" — pure imitatie van broncodelogboeken.
- **Je oudere broer of zus nabootsen.** Kinderen doen dit jaren eerder dan zij
  begin te redeneren over *waarom* de oudere broer of zus doet wat hij doet.

---

## De wiskunde (één regel)

Minimaliseer voor elke `(s, a)` in de dataset:

```
loss = -log π(a | s)        (cross-entropy)
```

Dat is het. Het beleid `π` is slechts een MLP die actielogits uitvoert;
training is identiek aan MNIST. Laten we het jargon opsplitsen:
- ** `π` (Pi):** Het standaardsymbool voor 'beleid': de regel of het neurale netwerk dat beslist wat te doen.
- **MLP (Multi-Layer Perceptron):** Een standaard, standaard neuraal netwerk.
- **Logits:** De ruwe, niet-genormaliseerde scores die het netwerk uitspuwt voordat we ze in waarschijnlijkheden omzetten.
- **Cross-entropie:** De standaardformule voor het bestraffen van een model wanneer het een lage waarschijnlijkheid aan het juiste antwoord toekent.
- **MNIST:** De beroemde beginnersdataset met handgeschreven cijfers.

Het trainen van een agent om een ​​spel te spelen via BC is letterlijk identiek aan het trainen van een netwerk om handgeschreven cijfers in MNIST te herkennen. In MNIST is de invoer een afbeelding en de uitvoer een cijfer (0-9). In BC is de invoer de spelstatus en de uitvoer de actie (bijvoorbeeld "naar links gaan").

---

## Wat onze code doet

Het script `behavioral_cloning.py`:

1. **Laadt alle vier datasets** gebouwd door `d4rl_dataset.py` 
   (`random`, `medium`, `expert`, `medium-replay`).
2. **traint voor elke dataset een afzonderlijk BC-beleid** voor een gradiënt van 10.000
   stappen van kruis-entropie. De beloningskolom wordt volledig genegeerd.
3. Elke 2500 stappen **evalueert** het huidige beleid door het uit te rollen
   gretig in de echte CartPole-v1-omgeving (gemiddeld 20 afleveringen).
4. Percelen:
   - Een staafdiagram: uiteindelijk BC-rendement per dataset.
   - Een leercurvegrafiek: hoe elke BC-variant tijdens de training stijgt.

---

## Wat je moet zien

Een typische run wordt afgedrukt:

```
Final evaluation returns:
  BC on random          ->    ~20  ± a few   (≈ random play)
  BC on medium          ->   ~150  ± large   (≈ the medium policy)
  BC on expert          ->   ~480  ± small   (≈ the expert policy)
  BC on medium-replay   ->    ~60  ± large   (≈ the AVERAGE of mixed data)
```

Het staafdiagram maakt het verhaal duidelijk: **De terugkeer van BC volgt die van de dataset
gemiddeld rendement.** Het kan niet boven dat plafond uitkomen, omdat het niet mogelijk is
geven de voorkeur aan de ‘goede’ delen van een gemengde dataset boven de ‘slechte’ delen – beide zijn dat ook
even geldige imitatiedoelen.

Dat is de clou: **BC erft het dataplafond.**

---

## BC versus CQL – de schoonste vergelijking

Op de dataset **medium-replay** (het meest realistische geval van gemengde kwaliteit):

| Method | Approx final return | Why? |
|--------|-------------------:|------|
| BC     | ~60   | Imiteert het *gemiddelde* van mislukte vroege runs + latere goede runs |
| CQL    | ~400+ | Gebruikt beloningen om de voorkeur te geven aan high-Q-overgangen; hecht een goed beleid aan uit gemengde gegevens |

Dus CQL **verslaat de gegevens**, BC **komt overeen met de gegevens**. Dat is het geheel
reden offline RL is een onderzoeksveld en niet alleen maar 'imiteren'
leren". Wanneer gegevens van gemengde kwaliteit zijn (wat bij echte logs altijd het geval is),
beloningsbewuste methoden herstellen meer.

Op **expert**-gegevens draait de vergelijking om: BC komt overeen met expert (~480). Je vraagt ​​je misschien af ​​waarom CQL hier een 'band' heeft in plaats van te verliezen. Omdat CQL is ontworpen om *conservatief* te zijn en acties te bestraffen die niet in de dataset voorkomen, doet het uiteindelijk precies wat de expert deed. Het kan de expert niet verslaan (omdat de maximaal mogelijke score al is behaald), maar het breekt ook niet actief de strategie van de expert. Het hangt gewoon samen met de prestaties van BC.

Dit is de beroemde afweging tussen ‘datakwaliteit en algoritme’:

```
                            EXPERT  data  →  BC wins, CQL ties
   Algorithm sophistication  ↑         
                            MIXED   data  →  CQL clearly beats BC
                            
                            RANDOM  data  →  Everybody fails; need exploration
```

---

## Waar BC leeft in modern RL

- **Vooropleiding voor online RL.** Veel moderne systemen (RT-1, Voyager,
  game-playing bots) begin met BC bij demonstraties en verfijn vervolgens
  online met PPO/SAC.
- **RLHF.** Stap 1 van InstructGPT is fijnafstelling onder toezicht — pure BC aan
  door mensen geschreven reacties. PPO + beloningsmodel komt later.
- **DAgger (Ross et al., 2011).** Een slimme uitbreiding om het **samengestelde fout**-probleem op te lossen.
  *Waarom zijn samengestelde fouten een probleem als BC perfect kloneert?* Zelfs als een BC-model 99% nauwkeurig is, gebeurt die fout van 1% uiteindelijk. Wanneer dat het geval is, komt de agent in een toestand terecht die hij nog nooit eerder heeft gezien in de perfect aangestuurde dataset. Omdat het in de war is, maakt het een grotere fout, waardoor het nog verder afwijkt van de bekende gegevens, wat uitmondt in een totale mislukking (zoals van een klif afrijden).
  *De oplossing:* We kunnen de expert gewoon vragen om voor altijd te blijven rijden, maar de tijd van een expert is duur. In plaats daarvan laat DAgger het BC-beleid bepalen. Wanneer het beleid een fout maakt en in een vreemde toestand terechtkomt, pauzeren we, vragen we de expert "wat zou jij *hier* doen?", en voegen dat toe aan de dataset. We ondervragen de expert alleen maar opnieuw over de staten die de BC-beleidsbezoeken afleggen, omdat we alleen de expert nodig hebben om ons te leren hoe we kunnen herstellen van onze eigen specifieke fouten, in plaats van ze altijd te ondervragen.
- **Decision Transformer (Chen et al., 2021).** Een "slimme" BC die
  conditioneert de actievoorspelling op een gewenste *return-to-go*,
  waardoor offline RL in wezen wordt omgezet in een voorspelling van het volgende token.

---

## Sleutelwoorden om te onthouden

| Word | Meaning |
|------|---------|
| **Imitatieleren** | Overkoepelende term voor "kopieer de demonstrator"; BC is het eenvoudigste lid |
| **Samenstellingsfout** | Een kleine BC-fout brengt je naar toestanden die de dataset nooit heeft gezien, waar fouten zich kunnen verergeren |
| **Demonstratiegegevens** | Trajecten opgesteld door een expert, gebruikt als BC-trainingsset |
| **Dataplafond** | Het rendement van BC wordt begrensd door het gemiddelde rendement in de dataset |
| **Dolk** | Een interactieve oplossing voor samengestelde fouten |

---

## Samenvatting van één zin

> **Gedragsklonen is slechts leren onder toezicht (toestand, actie)
> paren — sterk als de gegevens goed zijn, hulpeloos als de gegevens gemengd zijn.**