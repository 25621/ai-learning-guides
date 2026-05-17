# SARSA versus Q-Learning: veilige versus optimale paden 🐢 versus 🐇

## Wat is het?

Twee robots moeten allebei langs een **klifrand** lopen om het doel te bereiken. Beide robots
zijn nog steeds aan het *leren* en maken soms willekeurige bewegingen (oeps!).

- 🐢 **SARSA-robot**: "Ik weet dat ik soms wiebel... dus ik loop ver weg van de klif
  om veilig te zijn, ook al duurt het langer."
- 🐇 **Q-Learning robot**: "Het kortste pad omhelst de klif - laten we gaan! (Valt eraf
  soms tijdens het leren, maar leert uiteindelijk de beste route.)"

Beide robots zijn slim, maar ze maken een **andere afweging**: veilig, maar langzamer vs
optimaal-maar-risicovol-tijdens-leren.

---

## Het belangrijkste verschil: welke "volgende actie" gebruikt u?

Bij het bijwerken van scores na elke stap vragen beide algoritmen:
> "Wat is de waarde van de *volgende staat*?"

| Algoritme | Gebruikt de volgende actie... | In lijn met het beleid? |
|-----------|------------------------|------------|
| **SARSA** | ...die ik *eigenlijk* zal nemen (misschien willekeurig!) | Ja |
| **Q-Learning** | ...dat is *theoretisch het beste* (altijd hebzuchtig) | Nee |

**Voorbeeld uit de praktijk:** Twee kinderen leren fietsen.

- **SARSA-kind**: Blijft dicht bij het gras omdat *ze weten* dat ze soms willekeurig wiebelen.
  Ze maken plannen voor hun eigenlijke wiebelende zelf.
- **Q-Learning Kid**: Oefeningen midden op het pad omdat ze zich een perfectie voorstellen
  toekomstige zelf die nooit wiebelt. Ze vallen nu soms, maar leren sneller het beste pad.

Beide kinderen leren het uiteindelijk, maar tijdens de training valt het SARSA-kind minder!

---

## Wat onze code heeft gevonden

Beide algoritmen draaiden **500 afleveringen** op Cliff Walking met ε=0,1 (ε = epsilon, "EP-sih-lon"; hier betekent het een kans van 10% op het maken van een willekeurige zet):

| Metrisch | SARSA | Q-Leren |
|--------|-------|------------|
| Gemiddelde beloning tijdens training (laatste 50 afleveringen) | **-19,7** | **-51,0** |
| Hebzuchtige evaluatie (geen verkenning) | -17 | **-13** |

- **Tijdens de training**: SARSA krijgt **veel betere beloningen** omdat het de klif vermijdt
  (rekening houdend met zijn eigen willekeurige bewegingen)
- **Na de training** (puur hebzuchtig): Q-Learning vindt het **kortere optimale pad** (-13)!

Naarmate ε kleiner wordt richting 0, convergeren beide algoritmen naar hetzelfde optimale beleid.

---

## Visuele samenvatting

```
SARSA path (during training):     Q-Learning path (greedy, after training):
[ ][→][→][→][→][→][→][→][→][→][→][↓]   [ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][↓]
[↑][→][→][→][→][→][→][→][→][→][→][↓]   [ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][↓]
[ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][↓]   [ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][↓]
[S][C][C][C][C][C][C][C][C][C][C][G]   [S][→][→][→][→][→][→][→][→][→][→][G]
     (safe detour, top rows)                 (optimal, hugs cliff)
```

---

## Voorbeelden uit het echte leven

- **Nieuwe chirurg versus ervaren chirurg**: Nieuwe chirurg (SARSA) blijft verre van riskant
  technieken tijdens het leren. Ervaren chirurg (Q-Learning hebzuchtig) gebruikt het meest efficiënt
  techniek nadat je deze onder de knie hebt.
- **Stadsrijden versus snelwegroute**: SARSA-achtige planning zorgt voor veiligere woonstraten;
  Q-Learning vindt de optimale, maar smalle snelweg.
- **Studenten studeren**: SARSA-studenten houden zich tijdens de praktijk bij goed begrepen onderwerpen.
  Q-Learning-student gaat door tot de moeilijkste problemen (faalt meer) maar leert de optimale strategie.

---

## Sleutelwoorden om te onthouden

- **On-policy** (SARSA): leert wat u *eigenlijk doet*, inclusief willekeurige verkenningen
- **Buiten het beleid** (Q-Learning): leert over het *best mogelijke* gedrag los van
  wat je eigenlijk doet
- **Veilig pad**: langere route die gevaar vermijdt, gebruikt wanneer verkenning ongelukken veroorzaakt
- **Optimaal pad**: kortste route/route met de hoogste beloning, gevonden als er geen verkenning plaatsvindt
- **Afweging tussen exploratie en exploitatie**: de balans tussen nieuwe dingen uitproberen en gebruiken
  wat je weet

Het grote idee: **SARSA is veiliger tijdens training (on-policy), Q-Learning vindt het optimale
pad sneller (buiten het beleid). Wat beter is, hangt ervan af of het vallen van de klif ertoe doet!**