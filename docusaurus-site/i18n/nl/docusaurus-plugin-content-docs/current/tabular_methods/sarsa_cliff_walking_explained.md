# SARSA voor Cliff Walking 🏔️

## Wat is het?

Stel je een **zeer lange gang** voor met een **verschrikkelijke klif** langs één rand. Als je van de
klif, je moet helemaal terug naar het begin! Je doel is om van het ene uiteinde naar het andere te lopen
andere zo snel mogelijk, zonder te vallen.

**SARSA** is een robot die door te oefenen door deze gang leert lopen. Het leert een
*veilig pad* dat de klif vermijdt — ook al is het iets langer — omdat het weet dat dit mogelijk is
glijd per ongeluk dicht bij de rand tijdens het verkennen!

---

## Het grote idee: leren van wat je feitelijk doet

SARSA staat voor: **S**tate → **A**ction → **R**eward → **S**tate → **A**ction

Dit zijn de vijf stukjes informatie die SARSA gebruikt om te leren:

1. **S** — Waar ben ik nu? (huidige staat)
2. **A** — Welke actie heb ik daadwerkelijk ondernomen?
3. **R** — Welke beloning heb ik gekregen?
4. **S** — Waar ben ik terechtgekomen?
5. **A** — Welke actie ga ik *eigenlijk hierna ondernemen*?

De laatste "A" maakt SARSA speciaal! Het wordt bijgewerkt met behulp van de actie die *daadwerkelijk wordt uitgevoerd
next* (zelfs als dat een willekeurige verkenningszet is), niet de perfecte ideale actie.

**Voorbeeld uit de praktijk:** Denk erover na om te leren fietsen. Als je weet dat je soms wiebelt
willekeurig (verkenning), blijf je wat verder van geparkeerde auto's — omdat je je kent
wiebelende zelf zou kunnen uitwijken! SARSA doet dit: het leert een veilig pad omdat het verantwoording aflegt
zijn eigen willekeurige fouten.

---

## De klifwandelkaart

```
[ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]
[ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]
[ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]
[S][C][C][C][C][C][C][C][C][C][C][G]
   ← ← ← ← CLIFF ← ← ← ← ←
```

- **S** = Start (linksonder)
- **G** = Doel (rechtsonder)
- **C** = Cliff — hier stappen = -100 beloning, herstart!
- Elke andere stap = -1 beloning

---

## Wat onze code heeft gevonden

Na SARSA-training voor 500 afleveringen:

| Result | Value |
|--------|-------|
| Laatste gemiddelde beloning van 50 afleveringen | **-21,6** |
| Optimale (risicovolle) padbeloning | -13 |

Het geleerde beleid van SARSA gaat **langs de top van het raster** – een veilige omweg! Het kost een paar
extra stappen (-21 in plaats van -13), maar valt tijdens de training vrijwel nooit van de klif.

---

## Voorbeelden uit het echte leven

- **Verpleegkundige die medicijnen toedient**: volgt het bewezen veilige protocol (veilig pad), zelfs als
  Er bestaat een iets snellere methode, omdat kleine fouten (verkenning) gevaarlijk kunnen zijn.
- **Luchtvaartpiloten**: volg strikte checklists (veilige paden), zelfs als er sluiproutes lijken
  sneller, rekening houdend met menselijke fouten.
- **Leren koken**: begin met goed geteste recepten (veilig), geen risicovolle snelkoppelingen.

---

## Sleutelwoorden om te onthouden

- **On-beleid**: leert over het beleid dat het daadwerkelijk gebruikt (inclusief de willekeurige fouten)
- **SARSA-update**: gebruikt de *werkelijke* volgende actie, niet de theoretisch beste
- **Veilig pad**: een langer pad dat gevaar vermijdt, rekening houdend met verkenningsfouten
- **TD (Temporal Difference)-controle**: waarden bijwerken na elke afzonderlijke stap (niet wachten op de hele aflevering)

Het grote idee: **SARSA is eerlijk: het leert van wat het feitelijk doet, niet van wat het wenst
het zou doen. Hierdoor bent u voorzichtig en veilig in de buurt van gevaar!**