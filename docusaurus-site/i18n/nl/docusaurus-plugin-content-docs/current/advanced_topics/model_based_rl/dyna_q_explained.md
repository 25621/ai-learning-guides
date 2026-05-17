# Dyna-Q: Sneller leren door je voor te stellen 🧠

## Wat is het? {#what-is-it}

Stel je een kind voor dat Mia heet en leert navigeren op haar nieuwe school. Elke dag loopt ze
door de gangen en ontdekt nieuwe dingen: "De bibliotheek is voorbij de cafetaria",
'De kamer van meneer Smith is boven, bij het trappenhuis.'

Een **gewone Q-learning** student leert alleen van wat hij *vandaag* doet. Als vandaag
ze liep gewoon van de klas naar de kantine, waar ze alleen haar geheugen over bijwerkt
dat ene pad.

Een **Dyna-Q**-student is anders. Na elke echte wandeling gaat ze even zitten
minuut en **herhalingen in haar hoofd** verschillende wandelingen uit het verleden die ze zich herinnert. Elke herhaling
versterkt haar mentale kaart. Na een paar weken kent ze de school door en door –
niet omdat ze meer liep, maar omdat ze meer nadacht over wat ze deed
al gezien**.

Dat is precies wat Dyna-Q doet voor een RL-agent: het leert van de werkelijkheid
ervaring **en** vanuit een ingebeelde ervaring, gebaseerd op een model waarmee het voortbouwt
de weg.

---

## De drie ingrediënten {#the-three-ingredients}

Dyna-Q is "Q-leren + model + plannen". Eén echte stap vervult **drie** taken:

1. **Directe RL** — de gebruikelijke Q-learning-update van `(s, a, r, s')` .
2. **Modelleren** — schrijf op: "Toen ik *a* in *s* deed, kreeg ik *r* en eindigde op *s'*."
3. **Planning** — kies *n* willekeurige `(s, a)` paren uit het geheugen van het model en doe
   *n* meer Q-learning-updates, **doen alsof** deze stappen zojuist zijn gebeurd.

Die derde stap is de magie. Met `n = 50` veroorzaakt elke echte stap ter wereld
**51 updates** voor de Q-tabel. De agent leert ~50x sneller – in reële termen –
dan een pure Q-leerling.

---

## Een foto van de lus {#a-picture-of-the-loop}

```
                   ┌────────────────────────────────────┐
                   │                                    │
   real world  ──► take action a ──► observe (r, s')    │
                            │                           │
              ┌─────────────┼──────────────┐            │
              ▼             ▼              ▼            │
        Q-learning      Model[s,a] ← (r,s')   Planning ─┘
         update                            (n imagined updates)
```

Het model is slechts een opzoektabel:
 `(state, action) → (reward, next_state)` . Goedkoop om te bouwen, goedkoop om te bevragen.

---

## Voorbeelden uit het echte leven {#real-life-examples}

- **Schakstudie.** Grootmeesters besteden uren aan het opnieuw spelen van hun eigen partijen en
  meesterspelen in hun hoofd. Elke herhaling is "planning" - er wordt extra van geleerd
  ervaringen die al gebeurd zijn.
- **Een muzikant die toonladders oefent.** Nadat ze een keer een lastige maat hadden gespeeld, gingen ze verder
  repeteer het mentaal nog tien keer voordat je verder gaat. De vingers niet
  beweegt, maar de hersenen worden bijgewerkt.
- **Een zelfrijdende auto.** Terwijl hij stationair draait voor een rood licht, speelt hij de laatste af
  honderd rijstrookwisselingen in de simulatie om zijn beleid te verfijnen zonder te verbranden
  banden.

---

## Wat onze code doet {#what-our-code-does}

We gebruiken het klassieke **Dyna Maze** ([Sutton & Barto, Figuur 8.2](http://incompleteideas.net/book/the-book.html)): een 6×9 raster met
enkele muren, een start `S` in het midden links en een doel `G` rechtsboven.

We hebben drie varianten, elk met gemiddeld meer dan 30 willekeurige zaden:

| Setting | Planning steps per real step | Meaning |
|---------|----------------------------|---------|
| `n = 0` | 0 | eenvoudige Q-learning |
| `n = 5` | 5 | een beetje ingebeelde praktijk |
| `n = 50` | 50 | veel denkbeeldige praktijk |

Het script rapporteert het **gemiddelde aantal echte stappen per aflevering** als
de opleiding vordert. Minder stappen betekent dat de agent een directere kennis heeft geleerd
pad naar het doel.

### Wat je zou moeten zien als je het uitvoert {#what-you-should-see-when-you-run-it}

Het kortste pad in dit doolhof is ~9 treden; met ε-hebzuchtige verkenning a
goed opgeleide agent gemiddeld ~10 stappen per aflevering. Run voor 50 afleveringen en
alle drie de instellingen komen daar samen - het verschil is *hoe snel*:

| Instelling | Stappen per aflevering (laatste 10 eps) | Wat het betekent |
|---------|------------------------------:|---------------|
| `n = 0`   | ~10 | Geconvergeerd – maar er waren ongeveer 30 tot 50 rondzwervingen nodig om hier te komen |
| `n = 5`   | ~10 | Geconvergeerd binnen ~ 10 afleveringen |
| `n = 50`  | ~10 | Geconvergeerd binnen ~ 3–5 afleveringen |

Het interessante signaal is de *leercurve*, niet het uiteindelijke getal. Het perceel
opgeslagen in `outputs/dyna_q.png` toont drie curven die op zeer hoge snelheid naar de vloer duiken
verschillende snelheden: `n = 50` bereikt het in een handvol afleveringen, terwijl `n = 0` dat wel is
klimt nog steeds goed naar beneden in de ren. (In een klein deterministisch doolhof als dit,
Duidelijke Q-learning komt er uiteindelijk wel, Dyna-Q heeft alleen veel minder echte leermiddelen nodig
afleveringen, wat het hele punt is in omgevingen waar echte stappen kostbaar zijn.)

---

## Waarom het zo goed werkt in dit doolhof {#why-it-works-so-well-on-this-maze}

Twee redenen:

1. **De omgeving is deterministisch.** Elke `(s, a)` geeft altijd hetzelfde
    `(r, s')` , zodat het model na één bezoek exact is. Ingebeelde ervaring is
   zo goed als echte ervaring.
2. **Echte stappen zijn duur, ingebeelde stappen zijn gratis.** Elke ingebeelde update
   zijn slechts een paar opzoekingen in de tabel, terwijl voor een echte stap de agent moet lopen.
   Wanneer echte interacties kostbaar zijn (denk aan een echte robot, een echt spel), kan Dyna-Q
   is enorm steekproefefficiënt.

---

## Waar Dyna-Q worstelt {#where-dyna-q-struggles}

- **Stochastische omgevingen.** Als `(s, a)` tot veel verschillende `s'` kan leiden 
  waarden, een ‘onthoud de laatste uitkomst’-model liegt tegen je. Oplossing: winkelbezoek telt
  of train een probabilistisch model.
- **Niet-stationaire omgevingen.** Als de wereld verandert, bijvoorbeeld:
  deuropening die open stond, gaat plotseling dicht, of er verschijnt een snelkoppeling: het model
  verouderd raakt en verkeerde voorspellingen doet. **Dyna-Q+** lost dit op door
  het toevoegen van een *verkenningsbonus*: staten die al lang niet meer zijn bezocht
  tijd een kleine extra beloning ontvangen, waardoor de agent wordt aangespoord terug te gaan en te controleren
  of de wereld veranderd is.
- **Grote statusruimten.** Een woordenboek ingetoetst op `(s, a)` schaalt niet naar
  miljoenen staten of aan continue staten. Dat is precies de kloof die ontstaat
  **geleerde (neurale netwerk) wereldmodellen** vullen — zie `world_model.py` hierna.

---

## Sleutelwoorden om te onthouden {#key-words-to-remember}

| Word | Meaning |
|------|---------|
| **Model**       | Geheugen van `(state, action) → (reward, next_state)` |
| **Planningsstap** | Een Q-update uitvoeren met behulp van ingebeelde gegevens uit het model |
| **Directe RL**   | Een Q-update op basis van echte data |
| **Monsterefficiëntie** | Meet hoe effectief een AI-model of algoritme trainingsgegevens gebruikt om een ​​specifiek prestatieniveau te bereiken |
| **Dina**        | Sutton's architectuur die leren en plannen met elkaar verweven |

---

## Samenvatting van één zin {#one-sentence-summary}

> **Dyna-Q leert van doen EN van verbeelden — en verbeelden is gratis.**

Dit idee, in zijn moderne neurale vorm, drijft enkele van de sterkste RL-agenten aan
ooit gebouwd (MuZero, Dreamer, World Models).