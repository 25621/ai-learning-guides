# Doelnetwerk: de roos stabiliseren 🎯

## Het probleem van de bewegende doelpalen

Stel je voor dat je een roos probeert te raken met een pijl en boog. Je schiet, kijk waar
je pijl is geland en pas je doel aan voor de volgende keer. Simpel, toch?

Stel je nu voor dat de roos BEWEGT elke keer dat je schiet! Bij elke pijl schiet je een klein beetje
verandert waar de roos zal zijn voor het volgende schot. Je zou nooit verbeteren – dat zou wel zo zijn
een doelwit achtervolgen dat altijd wegrent.

Dat is precies het probleem met DQN zonder doelnetwerk!

---

## Waarom Q-Targets in beweging blijven

In DQN is het doel voor elke update:
> doel = beloning + γ × max(Q(next_state))

Hier is **γ (gamma)** de **discontofactor** — een getal tussen 0 en 1 (doorgaans 0,99)
dat bepaalt hoeveel de agent geeft om *toekomstige* beloningen versus *onmiddellijke* beloningen.

**Voorbeeld uit de praktijk:** Stel je voor dat iemand je nu een koekje aanbiedt, of morgen twee koekjes.
Als je nu echt cookies wilt, is je γ laag (je rekent zwaar af op de toekomst). Als je dat bent
geduldig en blij om te wachten, uw γ is hoog (toekomstige beloningen zijn bijna net zo belangrijk als nu).
In RL betekent γ = 0,99 "een volgende beloningsstap is op dit moment 99% van een beloning waard."

De Q-waarden aan de rechterkant komen van... hetzelfde netwerk dat we trainen!

Dus elke keer dat we het netwerk updaten (om de Q-waarden te verbeteren), veranderen we ook de
doelen. Het is een feedbackloop:

1. Update het netwerk → Q-waarden veranderen
2. Q-waarden veranderen → doelstellingen veranderen
3. Doelen veranderen → update het netwerk anders
4. Herhaal voor altijd - onstabiel!

**Voorbeeld uit de praktijk:** Je probeert jezelf te wegen op een weegschaal waarvan de meetwaarden veranderen
elke keer dat je erop stapt. Je zou nooit je ware gewicht weten!

---

## De oplossing: Bevries de Bullseye! ❄️

Het **Doelnetwerk** is een KOPIE van het hoofdnetwerk van Q-netwerk dat vastloopt.

- **Online netwerk** ( `qnet`): Elke trainingsstap bijgewerkt - leert snel
- **Doelnetwerk** ( `target_net`): Bevroren kopie — wordt alleen elke 100 stappen bijgewerkt

We gebruiken het FROZEN-doel om doelen te berekenen:
> doel = beloning + γ × max(Q_TARGET(volgende_staat))

Het doel blijft 100 stappen stil! Dat geeft het online netwerk een stabiel doel
doel voor. Vervolgens kopiëren we de online gewichten naar het doel, bevriezen opnieuw en herhalen.

**Voorbeeld uit de praktijk:** Denk aan een leerling en een leraar. De leraar geeft huiswerk
(het doel). De leerling leert en verbetert. Na 100 lessen UPDATES de leraar
het huiswerk wordt moeilijker. De leraar verandert niet elke minuut – dat
zou te chaotisch zijn!

---

## Het volledige DQN-recept 🍕

Het volledige DQN-algoritme (ervaringsherhaling + doelnetwerk) is:

```
1. Initialize online network Q and target network Q_target (same weights)
2. Create replay buffer (memory box)

Elke milieustap:
  een. Kies actie met ε-greedy met Q
  b. Opslaan (staat, actie, beloning, volgende_staat) in buffer

Elke 4 stappen:
  c. Bemonster een willekeurige minibatch uit de buffer
  d. Bereken doelen met Q_TARGET (bevroren!)
  e. Update Q om verlies te minimaliseren

Elke 100 stappen:
  f. Kopieer Q-gewichten → Q_TARGET (synchroniseer doel)
```

Dit is het exacte algoritme uit het DeepMind DQN-papier (2015)!

---

## Wat de vergelijking laat zien

Wanneer u `dqn_target_network.py` uitvoert, ziet u het volgende:

**Zonder doelnetwerk (alleen DQN + herhaling):**
- Trainen kan "oké" zijn, maar met periodieke instortingen
- Q-waarden kunnen uiteenlopen (exploderen of oscilleren)
- Leren is minder voorspelbaar

**Volledige DQN (herhaling + doelnetwerk):**
- Consistenter opwaarts leren
- Q-waarden blijven binnen een redelijk bereik
- Snellere convergentie naar de opgeloste drempel (195+ op CartPole)

---

## De "Dodelijke Triade" ☠️

Bij versterkend leren zorgt het combineren van drie dingen voor instabiliteit:

1. **Functiebenadering** (neuraal netwerk in plaats van tabel) ← we gebruiken dit
2. **Bootstrapping** (Q-waarden gebruiken om Q-waarden te schatten) ← we gebruiken dit
3. **Off-policy learning** (Q-learning gebruikt max, niet het daadwerkelijke beleid) ← wij gebruiken dit

Alle drie samen = de ‘dodelijke triade’. DQN temt dit met:
- Ervaringsherhaling → verbreekt correlaties
- Doelnetwerk → doorbreekt de feedbacklus

Het lost het probleem niet volledig op, maar maakt het wel beheersbaar!

---

## Sleutel Woordenschat

| Word | Meaning |
|------|---------|
| **Doelnetwerk** | Een bevroren kopie van het Q-netwerk dat alleen wordt gebruikt voor het berekenen van doelen |
| **Online netwerk** | Het Q-netwerk wordt actief getraind |
| **Synchroniseren** | Online netwerkgewichten kopiëren naar het doelnetwerk |
| **Feedbacklus** | Wanneer de uitvoer van een systeem terugkoppelt om de invoer te veranderen (kan instabiliteit veroorzaken) |
| **Dodelijke triade** | De combinatie van functiebenadering + bootstrapping + off-policy die instabiliteit veroorzaakt |

---

## Impact in de echte wereld

In 2015 publiceerde DeepMind hun DQN-paper waarin een AI werd getoond die 49 Atari kon spelen
games op bovenmenselijk niveau – met ENKEL deze twee trucs (herhaling + doelnetwerk).

Voordien dachten mensen dat je neurale netwerken niet met RL kon trainen vanwege de
instabiliteit. DeepMind bewees dat ze ongelijk hadden, en het was de start van de diepe RL-revolutie!

Vervolgens passen we dit volledige DQN-recept toe op Atari Pong – een echte videogame met onbewerkte pixels
als invoer!