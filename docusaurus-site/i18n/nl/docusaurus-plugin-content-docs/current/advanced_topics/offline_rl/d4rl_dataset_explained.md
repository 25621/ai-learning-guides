# D4RL Benchmark-gegevenssets 📦

## Wat is het?

Stel je voor dat je een robot wilt leren pannenkoeken om te draaien. Laten oefenen op a
een echte kachel voor een maand zou langzaam, gevaarlijk en duur zijn. Maar dat heb je wel gedaan
tien jaar opgenomen video van chef-koks die pannenkoeken omdraaien (sommige goed, sommige slecht,
sommige willekeurig). Kun jij de robot leren van *alleen die gegevens*, zonder ooit
een echte pan laten aanraken?

Dat is **offline versterkend leren**. De agent leert van een vaste
dataset van ervaringen uit het verleden – geen levende omgeving. Het moeilijkste is dat
de agent kan nooit tot het einde *uitproberen* wat hij heeft geleerd.

Om dit eerlijk te kunnen bestuderen, had de onderzoeksgemeenschap een *standaard nodig
gegevensset*. Dat is **D4RL** (**D**atasets voor **D**eep **D**ata-**D**riven
**R**einforcement **L**earning): een verzameling vooraf opgenomen overgangen voor
klassieke controletaken, uitgebracht door UC Berkeley in 2020. Elk papier traint
op dezelfde bytes, zodat de resultaten vergelijkbaar zijn.

---

## Wat zit er in een D4RL-gegevensset?

Voor elke taak levert D4RL **vier kwaliteitsniveaus**:

| Level | Where the data comes from | Why it matters |
|-------|--------------------------|----------------|
| **willekeurig**        | Een beleid dat acties uniform en willekeurig kiest | In het slechtste geval: kun je nog iets nuttigs leren? |
| **medium**        | Een gedeeltelijk getraind beleid (ongeveer de helft van de expertscore) | Realistisch: de meeste gelogde gegevens zijn middelmatig |
| **deskundige**        | Een bijna geconvergeerd beleid | In het beste geval: kunt u het bronbeleid matchen? |
| **medium herhaling** | De *volledige herhalingsbuffer* die wordt gebruikt om het mediumbeleid te trainen | Gemengd: bevat vroege mislukkingen EN latere successen |

Het verschil tussen `medium` en `medium-replay` is cruciaal:
- ** `medium` ** wordt gegenereerd door een enkel, vast 'gemiddeld' beleid te nemen en dit veel games te laten spelen. Alle gegevens weerspiegelen dit constante, gemiddelde vaardigheidsniveau.
- ** `medium-replay` ** is een historisch logboek. Het bevat alle ervaringen die zijn verzameld *tijdens het leren* vanaf het begin tot het gemiddelde niveau. Het combineert **slecht en goed**
overgangen – precies hoe een logboek uit de echte wereld eruit ziet (de eerste van een robot).
onhandige pogingen *en* het latere verfijnde gedrag ervan, allemaal in één emmer).

---

## Voorbeelden uit de praktijk van offline datasets

- **Medische dossiers.** Jaren aan tupels (patiëntstatus, behandeling, resultaat).
  Je kunt behandelingen op levende mensen niet randomiseren, maar je kunt wel leren:
  beter beleid uit het logboek.
- **Chatlogboeken van de klantenservice.** Miljoenen (user_message, agent_reply,
  tevredenheid) verslagen. Train een betere assistent zonder meer moeite te doen
  gebruikers.
- **Autonoom rijdende wagenparkgegevens.** Elke Tesla / Waymo-auto uploadt zijn eigen
  drijft. De vloot is een gigantische dataset met gemiddelde herhaling.
- **Aanbevelingssystemen.** Kliklogboeken van vorig jaar zijn een bevroren dataset:
  U kunt dezelfde advertenties niet opnieuw aan dezelfde gebruikers weergeven.

In alle vier de gevallen **kun je de omgeving niet om een nieuw monster vragen.** De
dataset is wat je hebt. Voor altijd.

---

## Wat onze code doet

De echte D4RL-datasets worden opgenomen op MuJoCo (Multi-Joint dynamics met
Contact) voortbewegingstaken
(zoals HalfCheetah, Hopper, Walker2d, Ant – dit zijn geavanceerde 3D-fysica-simulaties waarbij virtuele robots leren lopen en rennen). MuJoCo is zwaar om te installeren, dus wij
maak dezelfde **dezelfde structuur met vier niveaus opnieuw op CartPole-v1** — de standaard
beginnersomgeving uit eerdere fasen. De lessen worden direct overgedragen.

Het script `d4rl_dataset.py`:

1. **Traint een DQN** (Deep Q-Network, een standaard RL-algoritme) op CartPole totdat deze de taak oplost (retour ≥ 475).
2. **Maakt foto's van twee controlepunten** onderweg:
   - "medium" — het moment waarop het recente rendement de 150 overschreed
   - "expert" — het moment waarop het recente rendement de 475 overschreed
3. **Maakt een momentopname van de volledige herhalingsbuffer van het mediumbeleid** – elke transitie
   het ooit heeft gezien. Dat is onze "medium-replay" dataset.
4. **Introduceert drie nieuwe beleidsregels** voor elk 10.000 transities:
   - `random` — uniform willekeurig
   - `medium` — het gemiddelde controlepunt + ε=0,10 ruis
   - `expert` — het expertcontrolepunt + ε=0,02 ruis
5. **Bewaart vier `.npz`-bestanden** (het gecomprimeerde array-formaat van NumPy) in
    `outputs/` , elk met arrays `obs / action / reward / next_obs / terminal` .

Deze vier bestanden zijn de invoer voor `cql.py` en `behavioral_cloning.py` .

---

## Wat u moet zien als u het uitvoert

Een samenvatting in platte tekst, afgedrukt op de console en opgeslagen in
 `outputs/d4rl_summary.txt`:

```
dataset         |   N    |  mean return  |  min  |  max
------------------------------------------------------------
random          | 10000  |          ~22  |    ~9 |   ~80
medium          | 10000  |         ~180  |   ~50 |  ~500
expert          | 10000  |         ~490  |  ~400 |   500
medium-replay   | 10000  |          ~60  |    ~9 |  ~200
```

Het genereert ook een histogram ( `outputs/d4rl_returns.png` ) dat laat zien hoe de
vier datasets overlappen elkaar. De belangrijkste kenmerken om op te merken:

- **Willekeurige** clustert rond de 20 (de gemiddelde lengte van een willekeurige CartPole-aflevering).
- **Expert**-clusters aan het 500-plafond.
- **Medium** zit daar tussenin, met hoge variantie.
- **Medium-replay** heeft een lange rechterstaart: deze bestaat voornamelijk uit vroege mislukte runs (laag rendement), maar heeft een staart die zich uitstrekt tot hogere rendementen, zoals de agent heeft geleerd.

---

## Waarom de dataset ertoe doet

Op welke dataset u uw offline algoritme ook traint, u zet een
*plafond* van wat mogelijk is:

- **Van `expert` ** — zelfs een dom algoritme als BC (Behavioral Cloning, dat de gegevens precies kopieert) kan het goed doen,
  omdat alle gegevens goed zijn.
- **Van `random` ** — je hebt een slim algoritme nodig dat *aan elkaar kan hechten*
  zeldzame goede transities (een weg naar succes vinden door korte reeksen goede acties uit verschillende pogingen te combineren). BC zal volledig mislukken.
- **Van `medium-replay` ** — de meest realistische en interessante.
  Goede algoritmen (zoals **CQL** — Conservatieve Q-Learning, die vermijdt
  overmoedig zijn over acties die het nog nooit heeft gezien) kan soms **kloppen
  de gemiddelde datakwaliteit** omdat ze structuur uit gemengd halen
  signalen. Domme algoritmen (BC) gaan terug naar het gemiddelde.

We zullen precies dit verhaal zien in de volgende twee scripts.

---

## Sleutelwoorden om te onthouden

| Word | Meaning |
|------|---------|
| **OfflineRL**         | Trainen vanuit een vaste dataset; geen omgevingsinteractie toegestaan |
| **Gedragsbeleid**   | Het beleid dat de dataset *produceerde* |
| **Datasetkwaliteit**    | Hoe goed het gedragsbeleid was (random / medium / expert) |
| **Herhalingsbuffer**      | De volledige geschiedenis van overgangen gezien tijdens een trainingsrun |
| **Distributieverschuiving** | De kloof tussen de acties in de gegevensset en de acties die uw getrainde beleid wil ondernemen. Omdat de dataset nooit laat zien wat er gebeurt als het nieuwe beleid iets probeert dat niet is vastgelegd, kunnen de waardeschattingen van het algoritme voor die nieuwe acties gevaarlijk verkeerd zijn. |

---

## Samenvatting van één zin

> **D4RL bevriest RL in een benchmark voor begeleid leren: dezelfde bytes
> voor iedereen, geen vals spelen met de omgeving, moge het beste algoritme winnen.**