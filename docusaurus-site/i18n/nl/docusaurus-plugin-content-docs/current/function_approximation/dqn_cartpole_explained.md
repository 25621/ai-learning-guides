# Deep Q-Network (DQN) vanaf Scratch 🧠

## Het probleem met lineair

Ken je onze lineaire formule van vroeger nog?

> Score = w₁ × kar_positie + w₂ × kar_snelheid + w₃ × pool_hoek + w₄ × pool_snelheid

Dit werkt prima voor CartPole, maar hoe zit het met een videogame waarin je duizenden ziet
pixels? Daar kun je geen eenvoudig recept voor schrijven!

We hebben iets nodig dat naar ingewikkelde situaties kan kijken en de beste actie kan bedenken.
Dat iets is een **neuraal netwerk**.

---

## Wat is een neuraal netwerk?

Denk aan je hersenen. Miljoenen kleine cellen, neuronen genaamd, praten met elkaar. Wanneer jij
Als je iets warms aanraakt, sturen neuronen signalen: "HEET! → Trek NU je hand weg!" Elk neuron
geeft informatie door en samen nemen ze een slimme beslissing.

Een **neuraal netwerk op een computer** werkt op dezelfde manier:

```
Input Layer      Hidden Layer 1    Hidden Layer 2    Output Layer
[cart pos]  →    [128 neurons]  →  [128 neurons]  →  [push LEFT score]
[cart speed] →   [  ...       ]    [  ...       ]    [push RIGHT score]
[pole angle] →
[pole speed] →
```

Elke pijl heeft een **gewicht** (hoe sterk die verbinding is). Er zijn duizenden
deze gewichten – en het netwerk leert ze allemaal!

**Voorbeeld uit de praktijk:** Een chef-kok in een restaurant proeft uw eten en past er honderden aan
ingrediënten in één keer. Elke smaakpapil is als een neuron en samen vertellen ze het aan de chef
'voeg meer zout toe' of 'minder peper'. Het trainen van het netwerk is als het leren van een chef-kok
duizenden maaltijden.

---

## DQN = Diep Q-netwerk

**DQN** (Deep Q-Network) werd in 2013 door DeepMind uitgevonden. Ze namen de oude Q-learning
formule en verruilde de Q-tabel voor een neuraal netwerk!

In plaats van:
> Q-tabel[staat][actie] = score

Wij hebben:
> Q-netwerk(staat) → [score_for_left, score_for_right]

Het netwerk neemt de status als invoer en voert Q-waarden uit voor ALLE acties tegelijk.
Dit is veel efficiënter dan ze afzonderlijk te berekenen!

---

## Dit script: de "naïeve" versie

Dit script toont DQN **zonder** speciale trucs. Het is gewoon:
1. Ziet de staat
2. Vraagt het netwerk: "hoe goed is links? hoe goed is rechts?"
3. Voert de actie uit met de hoogste score
4. Krijgt een beloning, werkt het netwerk bij

**Dit is opzettelijk onstabiel!** Zie het als een leerling die het meteen vergeet
hun vorige lessen elke keer dat ze iets nieuws leren. Het netwerk wordt daarna bijgewerkt
elke stap, wat chaos veroorzaakt.

**Voorbeeld uit de praktijk:** Stel je voor dat je leert koken door daarna je hele recept te veranderen
elke hap. U kunt van 'te zout' naar 'helemaal geen zout' naar 'veel te zout' gaan
en neem nooit genoegen met het juiste bedrag. Dat is wat hier gebeurt!

---

## Wat je zult zien

Wanneer u `dqn_cartpole.py` uitvoert:
- De scores kunnen erg rondspringen (onstabiel leren)
- Soms wordt de agent heel goed en vergeet dan alles
- De verliesgrafiek vertoont wilde schommelingen

**Dit wordt verwacht!** Het laat zien WAAROM we verbeteringen nodig hebben: ervaar herhaling en doel
netwerken. Die komen hierna!

---

## De ε-Greedy-truc 🎲

De robot kiest niet altijd de beste actie. Soms wordt het willekeurig gekozen!

Waarom? Want als het altijd kiest wat het beste lijkt, zal het misschien nooit betere opties ontdekken.

> Met waarschijnlijkheid ε (epsilon): kies een RANDOM-actie (verken!)
> Met kans 1-ε: kies de BEST bekende actie (exploit!)

We beginnen met ε = 1,0 (100% willekeurig) en nemen langzaam af naar ε = 0,01 (1% willekeurig).
Op deze manier onderzoekt de robot eerst veel en concentreert zich vervolgens op wat hij heeft geleerd.

**Voorbeeld uit de praktijk:** Wanneer u een nieuwe stad bezoekt, kunt u willekeurige restaurants uitproberen
eerst (verkennen). Na een tijdje ga je terug naar je favorieten (exploit). Maar jij nog steeds
Probeer af en toe iets nieuws, voor het geval er een verborgen juweeltje is!

---

## Sleutel Woordenschat

| Word | Meaning |
|------|---------|
| **Neuraal netwerk** | Lagen van verbonden wiskundige neuronen die leren van gegevens |
| **Diep** | Meer dan één verborgen laag (vandaar ‘deep’ learning) |
| **DQN** | Deep Q-Network — maakt gebruik van een neuraal net in plaats van Q-table |
| **ε-hebzuchtig** | Strategie: soms willekeurig verkennen, andere keren de beste kennis exploiteren |
| **Instabiliteit** | Het netwerk blijft ‘vergeten’ omdat updates met elkaar interfereren |

---

## Wat ontbreekt (en waarom het ertoe doet)

Deze naïeve DQN heeft twee grote problemen:

1. **Gecorreleerde updates**: elke ervaring komt in volgorde (stap 1, stap 2, stap 3...).
   Als stap 5 slecht was, raken ALLE updates in de buurt door elkaar.
   
2. **Bewegend doel**: na elke update verandert het netwerk. Maar de volgende update gebruikt
   hetzelfde netwerk om te berekenen wat het doel zou moeten zijn. Het is alsof je op een bewegende schiet
   roos!

Deze problemen worden in het volgende opgelost door **Experience Replay** en **Target Networks**
scripts. Samen veranderen ze DQN van een wankele beginner in een kampioenspelspeler!