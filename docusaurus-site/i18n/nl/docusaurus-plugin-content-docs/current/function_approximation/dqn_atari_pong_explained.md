# DQN op Atari Pong 🏓

## Wat is Ataripong?

Pong is een van de oudste videogames ooit gemaakt: het lijkt op digitaal tafeltennis! Twee
peddels stuiteren een bal heen en weer. Je wint een punt als de tegenstander de bal mist.
Het spel eindigt als iemand 21 punten heeft behaald.

In onze versie bestuurt de AI één peddel. De tegenstander (computer) bestuurt de ander.
Het spel begint altijd bij score −21 (slechtst mogelijke). Een goede agent bereikt 0 of +21.

---

## Waarom is Pong moeilijk voor een AI?

In CartPole kon de robot de cijfers direct ZIEN (poolhoek, karsnelheid...).
In Pong ziet het alleen maar **rauwe pixels**: duizenden kleine gekleurde puntjes op een scherm!

```
CartPole input: [0.02, −0.14, 0.01, −0.23]   ← 4 numbers, easy!
Pong input:     [pixel grid: 210×160×3]        ← 100,800 numbers, MUCH harder!
```

De robot moet uit pixels het volgende uitzoeken:
- Waar is mijn peddel?
- Waar is de bal?
- Beweegt de bal naar links of naar rechts?
- Hoe snel?

Mensen doen dit automatisch (we hebben een geweldig zicht!). Voor een AI is dit een enorme uitdaging.

---

## Beweging zien: framestapelen 🎬

Eén frame (screenshot) vertelt je niet of de bal naar links of rechts beweegt. Je hebt nodig
om MEERDERE frames te zien om beweging te begrijpen - net zoals een flipbook alleen werkt wanneer
je bladert door vele pagina's.

**Frame Stacking:** Voer de laatste 4 frames tegelijkertijd in het netwerk.

```
Frame 1: ball at position 40
Frame 2: ball at position 43    → Stack these 4 frames → Network sees MOTION!
Frame 3: ball at position 46
Frame 4: ball at position 49
```

Het netwerk kan nu concluderen: "bal beweegt naar rechts met snelheid 3"

**Voorbeeld uit de praktijk:** Een film kijken in plaats van naar één frame kijken. Eén frame van een auto
ras is slechts een wazig beeld. Bekijk 4 frames en je ziet welke auto sneller is!

---

## Zien met een CNN 🔍

Voor pixelinvoer gebruiken we een speciaal neuraal netwerk dat een **Convolutioneel Neuraal Netwerk wordt genoemd
(CNN)**. In plaats van naar alle pixels tegelijk te kijken, gebruikt CNN schuifvensters om te detecteren
patronen – zoals ogen die een beeld scannen.

```
Raw pixels (84×84×4 frames)
       ↓
Conv Layer 1 (8×8 filter, stride 4) → finds edges and shapes
       ↓
Conv Layer 2 (4×4 filter, stride 2) → finds objects (paddles, ball)
       ↓
Conv Layer 3 (3×3 filter, stride 1) → finds relationships
       ↓
Flatten → 512 neurons → Q-values (one per action)
```

**Voorbeeld uit de praktijk:** Wanneer je in een menigte naar je vriend zoekt, merken je hersenen dit als eerste op
vormen (een persoon), dan gelaatstrekken (haarkleur) en vervolgens details (hun gezicht). CNN's werken de
op dezelfde manier – van eenvoudige patronen tot complexe patronen!

---

## Voorverwerking: de wereld verkleinen

Pong-frames zijn 210 x 160 pixels in kleur. Dat is te groot! We verwerken elk frame voor:

1. **Grijswaarden** — kleur doet er niet toe voor Pong (de bal is toch altijd wit)
2. **Formaat wijzigen naar 84×84** — kleiner = snellere training, maar nog steeds duidelijk genoeg om te zien
3. **Normaliseren naar [0,1]** — deel de pixelwaarden door 255, zodat het kleine getallen zijn

**Voorbeeld uit de praktijk:** Zoals het maken van een fotokopie op 50% formaat. De belangrijke details
(bal, peddels) zijn nog steeds zichtbaar, alleen kleiner. Het kopieerapparaat maakt zich er niet druk om
kleuren ook!

---

## Beloning knippen: alle games gelijk behandelen ✂️

In Pong krijg je +1 voor scoren, −1 voor gescoord worden.
In sommige andere Atari-spellen kunnen de scores duizenden bedragen!

We **beloningen** beperken tot [−1, +1], zodat het netwerk zich niets aantrekt van de omvang van de beloningen.
Dezelfde code kan in ELK Atari-spel worden getraind zonder de beloningsschalen aan te passen.

---

## Hoe lang duurt de training?

| Training Duration | What the Agent Learns |
|---|---|
| 100K stappen | Meestal willekeurig, reageert nauwelijks |
| 1M stappen | Begint soms richting de bal te bewegen |
| 5M stappen | Geeft enkele schoten terug |
| 10M stappen | Competitief spel, misschien wat winnen |
| 20M+ stappen | Verslaat vaak de computertegenstander |

Onze demo voert **300.000 stappen** uit – genoeg om te zien hoe de trainingsarchitectuur werkt
observeer vroeg leren, maar niet genoeg om het spel onder de knie te krijgen.

**Voorbeeld uit de praktijk:** Piano leren duurt maanden. Een oefensessie van 10 minuten
laat zien dat je het goed doet, maar verwacht nog geen concerten!

---

## Wat onze code heeft gevonden

Na 300K stappen op Pong:
- De makelaar begint met scores rond −20 (keert de bal nauwelijks terug)
- Tegen het einde verbetert het doorgaans tot ongeveer −15 tot −10
- De leercurve vertoont een geleidelijke verbetering door willekeurig spelen

Om echte competitieve Pong-prestaties te zien, moet je ~10M+ stappen rennen met een GPU.
De implementatie is compleet en correct; het heeft alleen meer tijd nodig!

---

## Sleutel Woordenschat

| Word | Meaning |
|------|---------|
| **CNN** | Convolutioneel Neuraal Netwerk – gespecialiseerd voor beeldinvoer |
| **Framestapelen** | Meerdere opeenvolgende frames invoeren om beweging vast te leggen |
| **Voorverwerking** | Transformeren van onbewerkte frames (grijstinten, formaat wijzigen, normaliseren) voordat ze naar het netwerk worden gevoerd |
| **Beloning knippen** | Beloningen beperken tot [−1, +1] om in verschillende games te werken |
| **ALE** | Arcade Learning Environment — de bibliotheek met Atari-spellen |

---

## De historische prestatie

Toen DeepMind in 2015 DQN publiceerde, was de wereld verbaasd. EEN ENKEL algoritme,
met DEZELFDE architectuur, 49 verschillende Atari-spellen leren spelen - waarvan er vele
bovenmenselijk niveau — alleen van onbewerkte pixels en een score!

Vóór DQN dachten mensen dat je de strategie voor elk spel met de hand moest coderen.
DQN liet zien dat een algemene leerling het allemaal zelf kan uitzoeken.
Het was een historisch moment in AI!