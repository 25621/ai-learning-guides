# Trainen op Montezuma's Revenge 🏛️🔑

## Waarom dit spel beroemd is (in RL-kringen) {#why-this-game-is-famous-in-rl-circles}

In 2015 leerde DeepMind's DQN tientallen Atari-spellen spelen
bovenmenselijk niveau van onbewerkte pixels. Het haalde de krantenkoppen. Maar begraven in de
resultatentabel was een spel waarin DQN **0** scoorde – hetzelfde als doen
helemaal niets: **Montezuma's Revenge**.

Waarom? Kijk eens wat het spel van je vraagt ​​in de allereerste kamer:

1. Beklim *een* ladder.
2. Loop over een richel.
3. Spring over een rollende schedel (mis het → je gaat dood).
4. Beklim *op* nog een ladder.
5. Pak de sleutel.

Dat zijn grofweg **100 nauwkeurige knopaanslagen**, en de game geeft je dat
**geen enkel punt** totdat de sleutel in handen is. Het beloningssignaal is a
vlak, karakterloos **nul** gedurende de gehele openingsscène.

Een normale RL-agent leert door zich aan te passen aan beloningen die hij daadwerkelijk ontvangt.
Als de beloning overal nul is, valt er niets te leren
van* — het is alsof je de bodem van een perfect vlakke vallei probeert te vinden
gevoel voor de afdaling. Dus DQN draaide gewoon rond
startplatform voor altijd. Montezuma werd *de* maatstaf voor **hard
verkenning**: het spel dat je alleen kunt verslaan als je *slim* verkent, niet
willekeurig.

De doorbraak kwam in 2018 met **Random Network Distillation (RND)** —
en de truc was precies het onderwerp van werkitem 1: voeg een **intrinsiek toe
nieuwsgierigheidsbonus** zodat de agent *zichzelf* beloont voor het bereiken van nieuwe schermen,
en plotseling heeft het een compact signaal dat het dieper het niveau in trekt. RND
kreeg een bovenmenselijke score op Montezuma. (Later: Go-Explore, Agent57, …)

## Voorbeelden uit de praktijk van een spaarzame beloning in "Montezuma-stijl". {#real-life-examples-of-montezuma-style-sparse-reward}

- **Een combinatieslot / een speurtocht met cryptische aanwijzingen.** Geen gedeeltelijke
  krediet. Je staat op nul totdat je plotseling bij de prijs bent.
- **Een paper geaccepteerd krijgen, of een startup winstgevend maken.** Maandenlang
  geen externe beloning, dan (misschien) een grote.
- **Een speedrun-route voor videogames.** Tientallen frame-perfecte invoer op een rij
  zonder feedback totdat de truc werkt of niet.
- **Escape rooms.** De kamer vertelt je bijna niets totdat je vastgeketend bent
  meerdere ontdekkingen bij elkaar.

In al deze gevallen is "gewoon willekeurige dingen proberen" hopeloos. Dat moet
*systematisch* verkennen - en een intern "ooh, dat is nieuw, ga door"
signaal is wat je systematisch houdt.

## Waarom we hier niet echt trainen op Pixel Montezuma {#why-we-dont-actually-train-on-pixel-montezuma-here}

Het *echte* ding goed doen betekent:

- een convolutioneel netwerk om het 210×160 RGB-scherm te zien,
- frame-stacking (zodat de agent kan zien in welke richting de schedel beweegt),
- een RND-module (nog twee netwerken: een vast willekeurig "doel" en een getraind
  "voorspeller"),
- en **tientallen miljoenen omgevingsframes** — vele GPU-uren.

Dat is een onderzoeksproject, geen onderwijsscript. Dus `montezuma_revenge.py` 
doet in plaats daarvan twee eerlijke dingen:

### 1. Het *raakt* van het echte spel (als `ale-py` is geïnstalleerd) {#1-it-touches-the-real-game-if-ale-py-is-installed}

Het laadt `ALE/MontezumaRevenge-v5` via Gymnasium, voert een **uniform- uit
willekeurige agent voor 2000 stappen**, en rapporteert de totale spelbeloning. De
Het nummer dat het afdrukt is bijna altijd **0,0** — de abstracte zinsnede "sparse
beloning" omgezet in een concreet, op zichzelf staand feit. Als de Atari
pakket niet is geïnstalleerd, wordt de opdracht `pip install` uit één regel afgedrukt en
gaat verder.

### 2. Het traint een tabellarische agent op een *schaalmodel*: `MiniMontezumaEnv` {#2-it-trains-a-tabular-agent-on-a-scale-model-minimontezumaenv}

Dit is een kleine rasterwereld met hetzelfde *skelet* als Montezuma's eerste
kamer:

```
###############
#S....#.......#
#.....#.......#
#.....#...T...#     S = start
#.....D.......#     K = key      D = door (only passable while carrying the key)
#..K..#.......#     T = treasure (the ONLY tile that gives a reward)
###############
```

Om te winnen moet je: naar de **sleutel** lopen (~6 zetten), deze oppakken; lopen naar de
**deur** (~4 zetten) — die nu opengaat; loop er doorheen en bereik de
**schat** (~5 zetten). Ongeveer **15 perfecte zetten**, met **nul feedback
tot aan de schat**. De vlag `has_key` maakt dus deel uit van de status van de agent
zodra je de sleutel pakt, is er een hele tweede kamer met *nieuwe* staten
ontdek — net zoals er nieuwe schermen verschijnen in het echte spel.

Vervolgens trainen we een gewone **tabelvormige Q-leerling** twee keer:

| Agent | Result on MiniMontezuma |
|-------|-------------------------|
| **geen nieuwsgierigheid (epsilon-hebzuchtig)** | Return blijft op **0** voor alle 1.500 afleveringen. Het bereikt zelfs nooit de sleutel. (Klinkt dit bekend? Dat is DQN in het echte spel.) |
| **met een nieuwsgierigheidsbonus voor voorspellingsfouten** | Bereikt de schat binnen ongeveer 20–25 afleveringen en leert vervolgens de **optimale route van 15 stappen**. (Dat is het RND-idee, verkleind om in een Q-tabel te passen.) |

De figuur toont de twee leercurves naast elkaar, plus de werkelijke
route die de nieuwsgierige agent heeft geleerd, getekend op het raster (start → sleutel → deur →
schat). Het script drukt die route ook af als ASCII-frames.

## De les {#the-lesson}

> **"Schaarse beloning" is geen eigenaardigheid van een vreemd Atari-spel — het is de
> standaard in elke wereld waar succes een lange, specifieke reeks van handelingen vereist
> acties.** Een agent die alleen beloningen biedt (vanille DQN) kan letterlijk niet worden verkregen
> gestart: er is geen verloop om te volgen. Een nieuwsgierigheidsbonus produceert
> één – een compact, zelf gegenereerd ‘dit is nieuw, ga door’-signaal – en
> dat signaal is wat de agent door de woestijn van nullen naar de voert
>eerste echte beloning. Alles daarna (RND, Go-Explore, Agent57) is een
> opgeschaalde, neurale netwerkversie van dezelfde zet.

## Sleutelwoorden om te onthouden {#key-words-to-remember}

| Word | Meaning |
|------|---------|
| **Moeilijke verkenning** | Problemen waarbij je alleen slaagt door slim te verkennen; willekeurige verkenning mislukt |
| **Schaarse beloning** | De beloning is bijna overal nul; je krijgt het pas na een lange correcte reeks |
| **Montezuma's wraak** | De Atari-game waarop klassieke deep-RL-agenten (DQN, A3C) 0 scoorden – de canonieke benchmark voor harde verkenning |
| **RND (willekeurige netwerkdestillatie)** | De methode uit 2018 die Montezuma versloeg met behulp van een nieuwsgierigheidsbonus voor voorspellingsfouten |
| **Ga verkennen** | "Denk aan veelbelovende staten, keer ernaar terug en verken van daaruit" - nog een Montezuma-kraker |
| **Schaalmodel** | Een kleine, goedkope omgeving die de *structuur* van een moeilijk probleem behoudt, zodat je het snel kunt bestuderen |

## Samenvatting van één zin {#one-sentence-summary}

> **Montezuma's Revenge is het spel dat RL leerde dat je nooit beloningen krijgt
> ontvangen kan je niets leren" - en de oplossing, toen en nu, is een
> nieuwsgierigheidsbonus waarmee de agent zichzelf kan belonen voor het verkennen totdat hij dit doet
> vindt de echte prijs.**