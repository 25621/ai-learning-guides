# Lineaire Q-Learning voor CartPole 🎪

## Wat is CartPole?

Stel je een bezemsteel voor die rechtop op je vinger balanceert. Als u uw vinger naar links of rechts beweegt
een klein beetje, dan voorkom je dat de bezemsteel valt. Dat is **CartPole**!

Een kleine robot zit op een karretje (een doos op wielen) en er steekt een paal bovenop.
De robot kan de kar alleen **links** of **rechts** duwen. Het moet leren dat vast te houden
paal zo lang mogelijk in balans houden — net zoals je een bezemsteel balanceert!

De robot kan 4 dingen over de wereld zien:
1. Waar de winkelwagen is
2. Hoe snel de kar beweegt
3. Hoeveel de paal leunt
4. Hoe snel de paal leunt

---

## Het grote probleem: te veel staten!

Kent u Q-learning uit Fase 2 nog? Er werd een grote tafel gebruikt om te onthouden hoe goed elke actie was
is in elke situatie (staat). Dat werkte prima voor Frozen Lake – er waren er maar 16
vierkanten op het ijs.

Maar CartPole is anders! De kar kan zich op **elke positie** bevinden en met **elke snelheid** bewegen,
met de paal in **elke hoek**. Er zijn feitelijk **oneindig veel mogelijke toestanden**! Dat kunnen we niet
maak een tabel met oneindige rijen. We zouden een notitieboekje ter grootte van een heelal nodig hebben!

**Voorbeeld uit de praktijk:** Stel je voor dat je leert fietsen. Je kunt niet alles onthouden
mogelijke wiebelen - er zijn er te veel! In plaats daarvan leer je een **regel**: "Als ik naar links leun, 
duw naar rechts; als ik naar rechts leun, duw dan naar links." Een eenvoudige regel werkt voor ALLE wiebels.

---

## De oplossing: een magische formule

**Lineaire functiebenadering** vervangt de gigantische tabel door een **kleine formule**:

> **Score(situatie, actie) = w₁ × kar_positie + w₂ × kar_snelheid + w₃ × pool_hoek + w₄ × pool_snelheid**

- De `w`-nummers worden **gewichten** genoemd; het zijn net knoppen waaraan je kunt draaien
- We leren **verschillende gewichten voor elke actie** (push-links en push-rechts)
- De formule geeft een score voor hoe goed elke actie op dit moment is

**Voorbeeld uit de praktijk:** Denk aan een eenvoudig recept: "1 kopje bloem + 2 eieren + ½ kopje boter."
De gewichten (1, 2, ½) vertellen je hoeveel elk ingrediënt ertoe doet. We leren de
recept voor goede beslissingen!

---

## Hoe leert het?

De robot probeert dingen, krijgt feedback en past de gewichten aan:

1. **Robot duwt de kar** (kiest de actie met de hoogste score)
2. **Fysica gebeurt** (de paal kantelt een beetje, de kar beweegt)
3. **Robot krijgt een beloning** (+1 voor elke stap dat de paal omhoog blijft, 0 als hij valt)
4. **Robot vraagt:** "Was het werkelijke resultaat beter of slechter dan ik had voorspeld?"
5. **Robot past de gewichten aan** om de volgende keer dichter bij de realiteit te komen

Dit is de **Semi-Gradient TD Update** — een mooie naam voor "een klein duwtje in de rug bij het recept"
gebaseerd op de verrassing."

> **Nieuw gewicht = Oud gewicht + Leersnelheid × (Wat er werkelijk is gebeurd − Wat ik voorspelde) × Functie**

---

## Wat onze code heeft gevonden

Wanneer je `linear_q_cartpole.py` uitvoert, doet de robot het volgende:

- Begint verschrikkelijk (de paal valt in 10-30 stappen)
- Leert geleidelijk goede gewichten na 3.000 pogingen
- Houdt de paal uiteindelijk 100–400+ stappen in evenwicht!

De grafiek laat de **leercurve** zien: hoe de score in de loop van de tijd beter wordt.
Het zal hobbelig zijn (leren gaat nooit soepel!), maar de trend zou omhoog moeten gaan.

---

## Waarom dit cool (en beperkt!) is

**Cool:** Een kleine formule met slechts 8 cijfers (4 gewichten × 2 acties) kan een paal in evenwicht brengen!
Geen gigantische tafel nodig.

**Beperkt:** De formule is te simpel voor complexe taken. Er wordt altijd uitgegaan van grotere aantallen
betekenen grotere effecten (wat niet altijd waar is). Voor hardere games zoals Atari hebben we dit nodig
**neurale netwerken** — dat is wat DQN doet!

---

## Sleutel Woordenschat

| Word | Meaning |
|------|---------|
| **Functie** | Eén meetbaar ding over de wereld (bijvoorbeeld de poolhoek) |
| **Gewicht** | Hoeveel een kenmerk de beslissing beïnvloedt |
| **Lineair** | De formule is alleen maar vermenigvuldigen en optellen (geen ingewikkelde curven) |
| **Semi-gradiënt** | Werk de gewichten bij door de richting van minder fouten te volgen |
| **Functiebenadering** | Een formule gebruiken in plaats van een tabel |

---

## Wat is het volgende?

Lineaire benadering is als het gebruik van een rechte liniaal om een curve te tekenen; het werkt prima
eenvoudige vormen, maar geen complexe. Voor Atari-spellen met miljoenen mogelijke situaties,
we hebben **Deep Q-Networks (DQN)** nodig: neurale netwerken die veel complexer kunnen leren
patronen. Dat staat in het volgende bestand!