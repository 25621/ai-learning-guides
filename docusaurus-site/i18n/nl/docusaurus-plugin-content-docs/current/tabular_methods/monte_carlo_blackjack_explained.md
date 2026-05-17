# Monte Carlo Controle voor Blackjack 🃏

## Wat is het?

Heb je ooit een kaartspel gespeeld waarbij je moest beslissen: **"Neem ik nog een kaart,
of ben ik blij met wat ik heb?"**

**Blackjack** (ook wel "21" genoemd) is precies dat! U wilt dat uw kaarten zo dicht mogelijk bij elkaar passen
mogelijk naar 21, zonder er overheen te gaan. Als je over de 21 gaat, "bust" je en verlies je!

**Monte Carlo-controle** is hoe een robot Blackjack leert spelen — door *duizenden te spelen
volledige games* en onthouden wat werkte en wat niet.

---

## Het grote idee: leer van complete verhalen

Het woord "Monte Carlo" komt van het beroemde casino in Monaco. In wiskunde betekent dit:
**gebruik willekeurige experimenten om iets te leren**.

Zo werkt het:

1. **Speel een volledige game** (een volledige aflevering) met welke strategie je ook hebt
2. **Kijk eens wat er gebeurde**: Heb je gewonnen? Verliezen? Tekenen?
3. **Achteruit werken**: Was het slaan (een kaart pakken) op 17-jarige leeftijd een goed idee? Hoe zit het met 14?
4. **Update uw geheugen**: onthoud of elke beslissing tot winnen of verliezen heeft geleid

Doe dit voor **500.000 spellen** en je zult heel goed worden!

**Voorbeeld uit de praktijk:** Stel je voor dat je leert koken door 500.000 maaltijden te bereiden. Elke keer jij
onthoud precies wat je hebt gedaan - en of de maaltijd goed smaakte. Na genoeg pogingen, jij
weet: "Het toevoegen van te veel zout bij deze stap maakte het altijd slecht." Monte Carlo werkt op dezelfde manier!

---

## Belangrijkste verschil met SARSA en Q-Learning

SARSA en Q-Learning updaten hun kennis **na elke stap** (zelfs halverwege de aflevering).
Monte Carlo wacht tot de **hele aflevering is afgelopen** en kijkt dan terug naar alles.

| Method | Updates when? | Needs complete episode? |
|--------|---------------|------------------------|
| **TD (SARSA, Q-Learning)** | Na elke stap | Nee |
| **Monte-Carlo** | Na elke volledige aflevering | Ja |

Dit maakt Monte Carlo eenvoudiger te begrijpen, maar het kan pas leren als elke aflevering is afgelopen.

---

## De Blackjack-staat

De robot kijkt elke beurt naar 3 dingen:
1. **Mijn kaarttotaal** (12 tot 21)
2. **Welke kaart laat de dealer zien?** (Aas tot en met 10)
3. **Heb ik een bruikbare Aas?** (Een Aas kan tellen als 1 of 11)

Op basis van deze 3 stukjes informatie beslist het: **Hit (pak een kaart) of Stick (stop)**?

---

## Wat onze code heeft gevonden

Na **500.000 spellen** Blackjack:

| Outcome | Percentage |
|---------|------------|
| **overwinningen** | **43,1%** |
| **Trekt** | 8,9% |
| **Verliezen** | 48,0% |

Dit komt dicht in de buurt van de wiskundig optimale "basisstrategie" (ongeveer 42-43% winst)!
De robot leerde wanneer hij moest slaan en wanneer hij moest vasthouden – gewoon door spelletjes te spelen en te onthouden.

Het geleerde beleid laat zien:
- **Hit** (neem een kaart) als je totaal laag is (het is onwaarschijnlijk dat je kapot gaat)
- **Blijf staan** als uw totaal hoog is (u kunt failliet gaan als u nog een kaart pakt)
- Als je een **bruikbare Aas** hebt, kun je agressiever zijn (deze kan indien nodig overschakelen van 11 naar 1)

---

## Voorbeelden uit het echte leven

- **Weervoorspelling**: Monte Carlo-simulaties voeren duizenden 'wat als'-scenario's uit
  om het weer van morgen te voorspellen.
- **Aandelenmarktmodellering**: analisten simuleren duizenden mogelijke toekomsten om te schatten
  risico.
- **Leren schaken**: een speler bekijkt hele partijen (niet slechts enkele zetten).
  begrijpen welke strategie tot winnen heeft geleid.

---

## Sleutelwoorden om te onthouden

- **Aflevering**: Eén compleet spel van begin tot eind
- **Return (G)**: totale beloning verzameld vanaf een punt in het spel tot het einde
- **MC bij elk bezoek**: update de score voor een staat telkens wanneer u deze in een aflevering bezoekt
- **Geen bootstrapping**: Monte Carlo maakt geen gebruik van schattingen van toekomstige waarden; het wacht
  voor het echte resultaat!
- **ε-zacht beleid** (ε = epsilon): Voer meestal de bekendste actie uit, maar verken soms willekeurig

Het grote idee: **Monte Carlo leert door veel complete spellen te spelen. Het is alsof je ervan leert
ervaring — je herinnert je alles wat er is gebeurd en ontdekt wat tot de overwinning heeft geleid!**