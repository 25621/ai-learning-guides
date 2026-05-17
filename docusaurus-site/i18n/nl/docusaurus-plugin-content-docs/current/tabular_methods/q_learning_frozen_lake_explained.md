# Q-Learning Agent voor Frozen Lake🧊

## Wat is het?

Stel je een bevroren vijver voor met glad ijs. Er is een **Startveld** en een **Doelveld**
met enkele **gaten** in het midden. Als je in een gat valt, begin je opnieuw!

Het ijs is glad, dus zelfs als je naar rechts probeert te lopen, kun je naar boven of naar beneden glijden.
Een **Q-Learning-agent** is een robot die leert – door steeds opnieuw te proberen – hoe hij eruit moet komen
Begin met doelen zonder erin te vallen!

---

## Waar staat de ‘Q’ in Q-Learning voor?

De **"Q"** staat voor **"Kwaliteit"** — specifiek de *kwaliteit* van het nemen van een bepaald product
handelen in een bepaalde situatie.

Zie het als een restaurantbeoordeling: "Hoe goed (kwaliteit) is het om de pizza bij DIT te bestellen
restaurant?" Q(s, a) vraagt: "Hoe goed is het om actie **a** te ondernemen als ik in de toestand **s** verkeer?"

Een hoge Q-waarde betekent: "Goede keuze! Deze actie leidt tot veel beloning."
Een lage Q-waarde betekent: "Slecht idee! Deze actie leidt meestal tot problemen."

**Voorbeeld uit de praktijk:** Stel je voor dat je een kind bent en besluit of je snoep wilt eten voor het avondeten.
Uw Q-waarde voor 'eet nu snoep' is op dit moment misschien hoog (het smaakt heerlijk!) maar over het algemeen laag
(moeder raakt van streek, je voelt je later ziek). Q-learning leert rekening te houden met die toekomst
gevolgen – niet alleen het onmiddellijke gevoel!

---

## Het grote idee: een magische tabel met scores

Q-Learning bouwt een grote tafel genaamd de **Q-table**. Elke rij is een vierkant op het ijs,
en elke kolom is een actie (links, rechts, omhoog, omlaag). De cijfers binnenin zijn **scores**:
"Hoe goed is het om deze actie vanaf dit plein te ondernemen?"

Elke keer dat de robot een zet probeert:
1. Het krijgt feedback (is het gevallen? heeft het het doel bereikt?)
2. Het werkt de score in de tabel bij met behulp van deze formule:

> **Nieuwe score = Oude score + Leerpercentage × (Wat er echt is gebeurd − Wat ik had verwacht)**

De robot vraagt ​​eigenlijk: "Was deze zet beter of slechter dan ik dacht?"

**Voorbeeld uit de praktijk:** Denk aan een baby die leert lopen. Elke keer dat ze een stap proberen te zetten en vallen,
ze leren dat "die stap slecht was." Elke keer dat ze slagen, herinneren ze zich "dat werkte!" Na
Na veel pogingen weten ze hoe ze moeten lopen. Q-learning doet hetzelfde, maar dan met een tafel!

---

## Wat Q-Learning speciaal maakt: het valt buiten het beleid!

Hier is iets slims: wanneer Q-Learning zijn tabel bijwerkt, gaat het er *altijd van uit dat dit het geval zal zijn
de volgende keer* de perfecte zet, zelfs als hij tijdens de training soms willekeurige bewegingen verkent.

Dit maakt Q-Learning **buiten het beleid**: de strategie die het *leert* (kies altijd de bekendste
actie) staat los van de strategie die het *volgt* tijdens de training (kies soms een willekeurige
actie om te verkennen). Concreet gebruikt de Q-tabelupdate de *maximale* Q-waarde van de volgende
staat – de theoretisch beste – zelfs als de daadwerkelijke volgende zet van de robot willekeurig zal zijn.

In duidelijke bewoordingen: de robot kan willekeurig naar links afdwalen om te verkennen, maar hij leert nog steeds
berekent alsof het vervolgens de *beste* actie zou ondernemen. Deze scheiding maakt Q-Learning mogelijk
convergeert naar de optimale strategie, ongeacht hoeveel deze onderzoekt.

---

## Wat onze code heeft gevonden

We hebben voor **50.000 afleveringen** getraind op het gladde 4×4 Frozen Lake:

| Metric | Result |
|--------|--------|
| Succespercentage van hebzuchtige evaluaties | **73,1%** |
| Mijlpaaldoelstelling (>70%) | ✓ **GESLAGEN** |

Het ijs is erg glad, dus zelfs de beste polis kan niet 100% van de tijd winnen!

De geleerde Q-tabel laat zien dat de agent het door heeft: ga naar beneden en naar rechts terwijl je de gaten ontwijkt.

---

## Voorbeelden uit het echte leven

- **Zelfrijdende auto**: via proefritten leren welke rijstroken je op kruispunten moet nemen.
- **Aanbevelingssystemen**: leren welke films u kunt aanbevelen op basis van de vraag of gebruikers deze leuk vonden
  eerdere suggesties.
- **Videogame AI**: een personage dat leert navigeren door een doolhof door vele paden te proberen.

---

## Sleutelwoorden om te onthouden

- **Q-tabel**: de tabel met "hoe goed is elke actie in elke staat"
- **Q(s, a)**: de score voor het ondernemen van actie a in staat s
- **Beloning**: wat de agent krijgt na het ondernemen van een actie (+1 voor het bereiken van het doel, anders 0)
- **Buiten het beleid**: leert de optimale strategie, zelfs tijdens willekeurige verkenningen
- **ε-greedy** (ε = epsilon): Voer meestal de bekendste actie uit; soms willekeurig verkennen
- **Kortingsfactor γ** (γ = gamma): Hoeveel toekomstige beloningen waard zijn (zoals nu de voorkeur geven aan geld versus later)

Het grote idee: **Q-Learning bouwt een "spiekbriefje" voor elke situatie en blijft verbeteren
totdat hij overal de beste zet kent.**