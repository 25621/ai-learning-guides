# Staatswaardefuncties 🗺️

## Wat is een ‘staat’?

Denk eens aan het spelen van een bordspel. Je staat op elk moment op *één*
vierkant van het bord. Dat vierkant is jouw **staat** – het is waar je bent
nu.

In ons 4×4-rasterspel zijn er 16 vierkanten (staten). Elk vierkant is een plaats
waar de agent op kan staan.

---

## Wat is een "waarde"?

Hier is de magische vraag: **"Als ik nu op dit plein sta,
Hoeveel schatten kan ik verwachten te verzamelen voordat het spel eindigt?"**

Dat antwoord is de **waarde** van die staat!

Een vierkant met een **hoge waarde** betekent: "Dit is een geweldige plek, waarschijnlijk wel
verzamel hier veel schatten!"

Een vierkant met een **lage waarde** betekent: "Oh, vanaf hier gaat het meestal
slecht."

**Voorbeeld uit de praktijk:** Stel je voor dat je verstoppertje speelt. Als je je verstopt
achter een grote boom (een geweldige plek) is je kans om te winnen groot — dat is een
staat van hoge waarde! Als je je midden in een lege kamer verstopt, doe je dat wel
waarschijnlijk gevonden worden – dat is een staat van lage waarde.

---

## Onze rasterwereld

Dit is het spelbord dat we gebruikten:

```
S  .  .  .      S = Start
.  H  .  H      H = Hole (reward -1, game ends)
.  .  .  H      G = Goal (reward +1, game ends)
H  .  .  G      . = Empty safe square
```

- Als je **G** (Doel) bereikt: krijg je **+1 punt** 🎉
- Als je op **H** (gat) stapt: krijg je **-1 punt** 😢
- Andere stappen: **0 punten**

We gebruikten γ (gamma) = 0,99, wat betekent dat toekomstige beloningen bijna net zoveel tellen
als onmiddellijke beloning. (Een snoepje morgen is bijna net zo lekker als snoepje vandaag!)

---

## Twee verschillende plannen (beleid)

We hebben twee beleidsvormen getest en de waarde van elk vierkant voor elk berekend:

### Beleid 1: Uniform willekeurig
Willekeurig omhoog, omlaag, naar links of naar rechts met gelijke kansen.

```
Values (Uniform Random Policy):
-0.912  -0.932  -0.912  -0.942
-0.929   (H)   -0.898   (H)
-0.901  -0.801  -0.696   (H)
 (H)   -0.630  -0.104   (G)
```

Bijna overal is het **negatief** – het willekeurige beleid valt zo in gaten
Vaak is het erg om ergens te zijn!

---

### Beleid 2: gericht op het doel
Ga liever naar rechts en naar beneden (in de richting van het doel), maar ga toch soms anders
richtingen.

```
Values (Biased-Toward-Goal Policy):
-0.838  -0.895  -0.814  -0.961
-0.798   (H)   -0.665   (H)
-0.595  -0.143  -0.213   (H)
 (H)    0.254   0.673   (G)
```

Nu hebben de vierkanten bij **Doel** **positieve waarden** (0,254 en 0,673)!
Door het slimme beleid zijn die pleinen fijne plekken om te vertoeven.

---

## Wat de kleuren in ons beeld betekenen

In onze visualisatie:
- **Groene vierkanten** = hoge waarde (geweldige plekken om te zijn)
- **Rode vierkantjes** = lage waarde (vermijd deze!)
- **Gele vierkantjes** = ergens daar tussenin

Je kunt het **gradiënt** zien: de waarden worden groener naarmate je dichter bij het doel komt
en roder in de buurt van Holes.

---

## Waarom geven we om waarden?

Waarden vormen de *basis* van versterkend leren! Zodra je de waarde kent
van elke staat kun je slimme beslissingen nemen:

> "Ik sta op vierkant A. Ik kan naar vierkant B (waarde = 0,5) of vierkant C (waarde = -0,3) gaan.
> Ik ga naar B – het heeft een hogere waarde!"

Dit is precies hoeveel RL-algoritmen (zoals Q-learning) leren goed te maken
beslissingen nemen zonder dat de regels bekend zijn.

**Voorbeeld uit de praktijk:** Stel je voor dat je kiest in welke rij je moet staan
supermarkt. Elke regel is een 'status'. De waarde van die toestand is hoe snel
je komt door het afrekenen. Je kijkt naar de lijnen (observeert staten) en kiest
degene met de hoogste waarde (kortste wachttijd + minste items).

---

## Hoe we de waarden berekenden

We hebben **Iteratieve beleidsevaluatie** gebruikt, wat als volgt werkt:

1. Begin: denk dat alle waarden 0 zijn.
2. Update: bereken voor elk vierkant waarop de waarde *moet* gebaseerd zijn
   waar het beleid u vervolgens naartoe brengt.
3. Herhaal dit totdat de waarden niet meer veranderen (convergeren).

Wiskundig: **V(s) = Σ_a π(a|s) × [R(s,a) + γ × V(volgende_toestand)]**

In gewoon Engels: "De waarde van dit vierkant = de gemiddelde beloning die ik krijg
op dit moment + een klein beetje van de waarde van waar ik ook terechtkom."

---

## Sleutelwoorden om te onthouden

- **Staat**: waar je nu bent (één vakje op het bord)
- **Waarde V(s)**: verwachte totale beloning vanaf staat s
- **Beleid**: uw plan voor wat u in elke staat moet doen
- **Kortingsfactor γ**: hoeveel u geeft om toekomstige beloningen (0,99 = veel!)
- **Beleidsevaluatie**: waarden berekenen voor elke staat onder een bepaald beleid

Het grote idee: **Sommige plaatsen zijn beter dan andere – en de waardefunctie
vertelt je precies hoe goed elke plek is!**