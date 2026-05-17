# PPO-hyperparametergevoeligheid: wat is het belangrijkst?

## Waarom hyperparameters belangrijk zijn

Stel je voor dat je een chocoladetaart bakt. Het recept vraagt om:
- 2 eieren
- 200 g bloem
- 1 theelepel bakpoeder
- 35 minuten bij 180°C

Als je 10 eieren gebruikt, explodeert de cake. Als je 0,1 theelepel bakpoeder gebruikt, rijst het niet.
Als je 10 minuten op 300°C bakt, brandt het aan de buitenkant en is het van binnen rauw.

**Hyperparameters in PPO lijken op ingrediënten en oveninstellingen.** De juiste combinatie werkt
prachtig; Verkeerde instellingen kunnen het leren volledig verhinderen.

Dit script test systematisch 3 belangrijke hyperparameters door er slechts ÉÉN tegelijk te wijzigen,
voer elke instelling uit met 3 verschillende willekeurige zaden en vergelijk de resultaten.

---

## De drie experimenten

### Experiment 1: Clip Epsilon (ε)

```
ε = 0.05   (very conservative — only tiny policy changes allowed)
ε = 0.2    (standard — balanced safety and speed)
ε = 0.4    (aggressive — allows large policy changes)
```

**Wat regelt ε?**

ε is de grootte van het "veiligheidsvenster" rond het oude beleid:
`` `
ratio must stay in [1 - ε,  1 + ε]
ε=0.05: ratio in [0.95, 1.05]  ← tiny changes
ε=0.2:  ratio in [0.80, 1.20]  ← standard  
ε=0.4:  ratio in [0.60, 1.40]  ← large changes
` ``

**Voorbeeld uit de praktijk:** Beschouw ε als 'hoe ver je de auto in één beweging mag sturen'.
- ε=0,05: Net als rijden op ijs — slechts kleine aanpassingen
- ε=0,2: Normaal rijden — redelijke bochten
- ε=0,4: Autocoureur — agressieve besturing, risico op **spin-out** (controleverlies omdat de verandering te drastisch is, zoals een auto die van de weg slipt)

**Verwachte resultaten:**
- ε=0,05: Langzaam maar stabiel leren (te voorzichtig)
- ε=0,2: Goede balans (de **"Goudlokje"-waarde** — niet te klein, niet te groot, precies goed — genoemd naar het sprookje waarin Goudlokje de pap plukt die niet te warm en niet te koud is)
- ε=0,4: Kan snel leren, maar kan **doorschieten en oscilleren** (doorschieten = voorbij het optimale beleid gaan; oscilleren = er heen en weer omheen stuiteren zonder te settelen, als een slinger die te ver in beide richtingen zwaait)

---

### Experiment 2: Leersnelheid

```
lr = 1e-4  (slow but stable)
lr = 3e-4  (standard)
lr = 1e-3  (fast but risky)
```

**Wat houdt leersnelheidscontrole in?**

De leersnelheid is vergelijkbaar met de "stapgrootte" bij het beklimmen van een heuvel (elke stap = één update van de gewichten van het neurale netwerk, waarbij deze enigszins in de richting wordt verplaatst die de beloning verbetert):
- Te klein: het duurt een eeuwigheid om de top te bereiken (convergeert langzaam)
- Te groot: je schiet voorbij de piek en valt aan de andere kant naar beneden (**divergeert** – de trainingsbeloning stort in of fluctueert wild in plaats van gestaag te verbeteren)
- Precies goed: gestage vooruitgang richting de top

**Voorbeeld uit de praktijk:** Een gitaarsnaar stemmen.
- lr=1e-4: Kleine draaiingen van de stemknop** (de knop waaraan je draait om een snaar strakker of losser te maken) — duurt een eeuwigheid maar precies
- lr=3e-4: Normale stemming — vind de juiste toonhoogte in een paar beurten
- lr=1e-3: Grote **rukjes** (plotseling hard trekken) aan de pin — kan de snaar **breken** (volledig breken, net zoals te grote updates de training onomkeerbaar kunnen onderbreken)!

**Verwachte resultaten:**
- lr=1e-4: Uiteindelijk goed maar erg langzaam
- lr=3e-4: Beste prestatie in het algemeen
- lr=1e-3: Snelle aanvankelijke vooruitgang, daarna instabiliteit

---

### Experiment 3: Tijdperken bijwerken (K)

```
K = 3   (conservative — few passes through each batch)
K = 10  (standard)
K = 20  (aggressive — many passes through each batch)
```

**Wat controleren update-epochs?**

Na het verzamelen van een **uitrol** (= het spel een tijdje spelen om nieuwe ervaring op te doen – zoals een leerling die een huiswerksessie doet voordat hij deze doorneemt), bundelt PPO die ervaring in een **batch** (= de volledige set status-, actie- en beloningstuples van die uitrol). Vervolgens voert het K **passes** uit (= volledige sweeps door de batch, waarbij elke pass het netwerk één keer bijwerkt) over dezelfde gegevens.
Meer tijdperken = meer leren uit elke batch persen, maar het risico lopen **overfitting naar verouderde gegevens** (= het onthouden van patronen die waar waren onder het oude beleid, maar niet langer geldig zijn zodra het beleid is bijgewerkt, zoals een student die het examen van vorig jaar uit zijn hoofd leert en niet slaagt voor een nieuw examen).

**Voorbeeld uit de praktijk:** Een leerling oefent met een reeks van 20 wiskundeproblemen.
- K=3: Doe elk probleem 3 keer → leer nog steeds, overdrijf de oefenset niet
- K=10: Voer elk probleem 10 keer uit → goede beheersing van deze specifieke problemen
- K=20: Voer elk probleem 20 keer uit → **oplossingen onthouden zonder wiskunde echt te begrijpen** (= het model past perfect bij de specifieke batch, maar verliest het vermogen om te generaliseren)!

> ⚠️ **"Maar de resultaten voor K=20 zien er goed uit — wat zou mij dat schelen?"**
> De clipping-truc van PPO beperkt de mate waarin het beleid per passage kan veranderen, zodat K=20 geen plotselinge ineenstorting zal veroorzaken.
> De agent is zich echter nog steeds stilletjes aan het overmatig aanpassen aan gegevens die niet langer weerspiegelen wat het huidige beleid feitelijk zou ervaren.
> Dit **vertraagt ​​het leren op de lange termijn**: bij elke uitrol leert de agent minder dan zou moeten, omdat later steeds verouderde informatie wordt gerecycled.
> De schade is geleidelijk en niet dramatisch – en dat is precies de reden waarom deze bij korte experimenten gemakkelijk over het hoofd wordt gezien.

De clipping voorkomt catastrofale overfitting, maar te veel tijdperken kunnen het algehele leerproces nog steeds vertragen.

**Verwachte resultaten:**
- K=3: Minder efficiënt (een deel van het leerpotentieel gaat per batch verloren)
- K=10: Goede balans
- K=20: Risico dat het beleid **te veel vertrouwen heeft in verouderde gegevens** (= de updates van het netwerk worden aangestuurd door ervaringen die niet langer overeenkomen met wat het huidige beleid zou tegenkomen, waardoor de steekproefefficiëntie stilletjes wordt uitgehold)

---

## Hoe de resultaten te lezen

De grafiek toont drie grafieken, die elk één hyperparameter variëren:

```
Left graph:    Clip Epsilon — which ε learns fastest?
Middle graph:  Learning Rate — which lr is most stable?
Right graph:   Update Epochs — which K finds the best policy?
```

Elke regel is de **gemiddelde beloning over 3 zaden** (om willekeur te verminderen).

**Waar moet je op letten:**
1. **Leersnelheid:** Welke lijn bereikt het snelst een hoge beloning?
2. **Eindprestatie:** Welke lijn behaalt de hoogste eindbeloning?
3. **Stabiliteit:** Welke lijn heeft de minste oscillatie?

Een goede hyperparameter brengt deze drie in evenwicht!

---

## Methodologie: wetenschappelijke experimenten

Dit experiment maakt gebruik van het **ablatiestudie**-ontwerp (= een methode waarbij je één component tegelijk verwijdert of varieert om de individuele impact ervan te meten – genoemd naar de wetenschappelijke praktijk van het selectief verwijderen van weefsel om de functie ervan te bestuderen):
1. Kies standaardwaarden: ε=0,2, lr=3e-4, K=10
2. Wijzig EEN parameter tegelijk
3. Houd al het andere vast
4. Vergelijk resultaten

Dit vertelt ons het effect van ELKE parameter afzonderlijk.

**Voorbeeld uit de praktijk:** Testen of een nieuwe meststof planten helpt:
- Verander de meststof, houd al het andere hetzelfde (dezelfde grond, water, zonlicht)
- Als de planten beter groeien → kunstmest hielp!

---

## Gemeenschappelijke bevindingen in de praktijk

| Hyperparameter | Too Small | Sweet Spot | Too Large |
|----------------|-----------|------------|-----------|
| **ε (clip)** | Langzame convergentie | ε ≈ 0,2 | Instabiliteit |
| **lr** | Te langzaam | 2,5e-4 tot 3e-4 | Divergentie |
| **K (tijdperken)** | **Afvalgegevens** (uitrol weggooien voordat volledig signaal wordt geëxtraheerd) | K=4-10 | Overfitting voor verouderde implementatiegegevens |
| **n_stappen** | Te luidruchtig | 128-2048 | **OOM Geheugenfouten** (gebruikt te veel RAM) |
| **batchgrootte** | Te luidruchtig | 32-256 | **OOM Geheugenfouten** (gebruikt te veel RAM) |

Deze ‘sweet spots’ kunnen verschuiven afhankelijk van de omgeving!

---

## Het belangrijkste inzicht: PPO is relatief robuust

Vergeleken met eerdere algoritmen (zoals DQN zonder doelnetwerken) is PPO relatief
robuust voor hyperparameterkeuzes. Het clipmechanisme zorgt voor een natuurlijk vangnet.

**Voorbeeld uit de praktijk:** Een auto met **ABS** (Antiblokkeerremsysteem – een veiligheidsvoorziening die voorkomt dat de wielen blokkeren tijdens hard remmen, zodat de bestuurder de controle behoudt) remt versus zonder:
- Zonder ABS (DQN): Eén verkeerde afslag (slechte hyperparameter) en je spint
- Met ABS (PPO): De auto corrigeert zichzelf – redelijke hyperparameters werken allemaal prima

Deze robuustheid is een belangrijke reden waarom PPO in de praktijk het populairste RL-algoritme is!

---

## Belangrijkste afhaalrestaurants

| Concept | Plain English |
|---------|---------------|
| **Ablatieonderzoek** | Verander één ding tegelijk om het effect ervan te zien |
| **Clip epsilon ε** | Veiligheidsgrens — 0,2 is meestal het beste |
| **Leerpercentage** | **Stapgrootte** — hoeveel het gewicht van het netwerk wordt aangepast na elke batch (denk aan de grootte van elke voetstap wanneer je naar een doel loopt). **2,5e-4 tot 3e-4** is de wetenschappelijke notatie voor 0,00025 tot 0,0003 — dit zijn dimensieloze vermenigvuldigers, geen tijdswaarden |
| **Epoches K bijwerken** | Hoe vaak moet elke batch opnieuw worden gebruikt: 4-10 is standaard |
| **Willekeurige zaden** | Elk experiment wordt herhaald met verschillende **willekeurige zaden** (= het startnummer dat wordt ingevoerd in de generator voor willekeurige getallen, die alle willekeurige keuzes in de training controleert). Door meerdere zaden te gebruiken, wordt duidelijk of de resultaten consistent zijn of gewoon geluk hebben gehad |

---

## Samenvatting: Beleidsgradiëntmethoden in één oogopslag

```
REINFORCE              A2C                    PPO
     │                  │                      │
Full episodes     N-step updates         N-step + clipping
Simple but noisy  Faster but unstable    Stable + efficient
Best for easy     Medium difficulty      Hard environments
environments      environments           (industry standard)
```

**Als je uit deze fase maar ÉÉN algoritme leert, leer dan PPO.** Het is de basis van:
- OpenAI's ChatGPT-training (RLHF maakt gebruik van PPO)
- DeepMind's AlphaGo-vervolgacties
- Meest moderne robotica-onderzoek
- AI's voor het spelen van videogames