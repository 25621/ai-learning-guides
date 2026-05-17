# Frozen Lake met een willekeurig beleid 🧊

## Wat is bevroren meer?

Stel je voor dat je met je vrienden op een **bevroren vijver** speelt.

Het ijs is grotendeels veilig, maar sommige plekken hebben **gaten** — als je op een gat stapt,
je valt erin en het spel is voorbij! Aan het ene uiteinde van de vijver ligt een **cadeau** 🎁.
Jouw taak is om van het **begin** naar het **cadeau** te glijden zonder erin te vallen.

Zo ziet het bevroren meer eruit (4 vierkanten × 4 vierkanten):

```
S  F  F  F
F  H  F  H
F  F  F  H
H  F  F  G
```

- **S** = Start (waar je begint)
- **F** = Bevroren ijs (veilig!)
- **H** = Hole (invallen, game over 😨)
- **G** = Doel — het cadeau! 🎁

---

## Het lastige deel: glad ijs!

Als je op een echte bevroren vijver *naar rechts* probeert te lopen, laat het gladde ijs
je soms *omhoog* of *omlaag* glijden in plaats daarvan! Dat maakt dit moeilijk.

Zelfs als je *naar rechts* wilt gaan, kan het spel je ergens anders heen schuiven.
Dit heet **stochasticiteit** – een mooi woord voor ‘dingen gaan niet altijd
zoals je het gepland had.”

---

## Wat is een willekeurig beleid?

Een **beleid** is gewoon een plan: "In deze situatie zal ik DEZE actie uitvoeren."

Een **willekeurig beleid** betekent: "Ik heb helemaal geen plan! Ik kies gewoon
een willekeurige richting – omhoog, omlaag, naar links of naar rechts – alsof ik aan een spinner draai!"

Het is als een baby die op ijs loopt, zonder enig idee waar het cadeau is.

---

## Wat onze code heeft gevonden

We hebben het willekeurige beleid geprobeerd voor **1.000 spellen**:

| Result | Value |
|--------|-------|
| **Keren dat het cadeau bereikt werd** | 11 van de 1.000 (1,1%) |
| **Gemiddelde stappen per spel** | 7,5 stappen |
| **Snelste spel** | 2 stappen |
| **Langste spel** | 33 stappen |

Meestal viel de willekeurige wandelaar snel in een gat.
Slechts 1 op de 100 spellen eindigde met het vinden van het cadeau!

---

## Waarom is dit nuttig?

Ook al is het willekeurige beleid verschrikkelijk, het geeft ons een **basislijn** –
een startpunt om mee te vergelijken.

Wanneer we later een *slim* beleid bouwen (met behulp van Q-learning of andere algoritmen),
we kunnen zeggen: "Onze slimme agent slaagt 75% van de tijd - veel beter dan de
willekeurige wandelaar is 1%!"

**Voorbeeld uit de praktijk:** Stel je voor dat je je klaslokaal op een nieuwe school probeert te vinden
door bij elke gang willekeurig links of rechts af te slaan. Misschien kom je daar wel
uiteindelijk, maar het zou lang duren! Een slim beleid is als het hebben van een kaart.

---

## Wat de Heatmap laat zien

Op onze foto laat de **heatmap** zien welke vierkanten de willekeurige wandelaar heeft bezocht
meestal:

- Het **Start**-plein wordt veel bezocht (elk spel begint daar).
- Pleinen bij de **gaten** worden minder bezocht (de wandelaar valt er vaak eerder in
  hen bereiken).
- Het **Doel** wordt zeer zelden bezocht omdat de willekeurige wandelaar bijna nooit wordt bezocht
  maakt het daar.

Dit vertelt ons iets belangrijks: het willekeurige beleid blijft hangen in de buurt van de
begint en verkent nooit echt het hele meer.

---

## Sleutelwoorden om te onthouden

- **Beleid**: uw plan voor wat u in elke situatie moet doen
- **Willekeurig beleid**: geen plan - kies gewoon een willekeurige actie!
- **Baseline**: een slecht resultaat dat we ter vergelijking gebruiken (hoeveel beter kunnen we doen?)
- **Stochastisch**: dingen gaan niet altijd zoals je wilt (zoals glad ijs!)
- **Succespercentage**: hoe vaak hebben we gewonnen? (Hier: 1,1% – erg laag!)

Het grote idee: **Een willekeurig beleid is een startpunt. Echt leren betekent
een beter plan bouwen!**