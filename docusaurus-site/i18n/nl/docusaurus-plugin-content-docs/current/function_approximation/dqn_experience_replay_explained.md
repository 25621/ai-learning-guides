# Ervaar herhaling: de robot leren onthouden 🎒

## Het probleem: vergeten (en verwarring)

Weet je nog hoe de naïeve DQN onstabiel was? De grootste reden is **gecorreleerd leren**.

Wanneer de robot het spel speelt, ervaart hij de dingen in volgorde:
> Stap 1 → Stap 2 → Stap 3 → Stap 4 → ...

Deze stappen zijn met elkaar verbonden! Als de robot in stap 10 naar links leunt, zal stap 11 ook leunen
links. Ze zijn niet onafhankelijk; ze zijn van elkaar afhankelijk.

Wanneer we het netwerk updaten met behulp van deze gecorreleerde stappen, is het alsof we proberen te leren
geschiedenis door steeds hetzelfde hoofdstuk te lezen. Je zou echt goed worden in één hoofdstuk
en vergeet al het andere!

**Voorbeeld uit de praktijk:** Stel je voor dat je studeert voor een toets door alleen die van gisteren te oefenen
huiswerk. Je wordt geweldig in precies die problemen, maar de test heeft andere vragen!
Je moet een MIX van verschillende problemen oefenen.

---

## De oplossing: een geheugenbox 📦

**Experience Replay** voegt een grote geheugenbox (de **herhalingsbuffer**) toe aan de robot.

In plaats van te leren van de allernieuwste ervaringen, kan de robot:
1. **Slaat** elke ervaring op in de geheugenbox: (status, actie, beloning, volgende status)
2. **Pakt willekeurig** een handvol herinneringen uit de doos
3. **Leert van die willekeurige mix** in plaats van alleen de laatste stap

```
Game Step 1 → [store in box]
Game Step 2 → [store in box]
Game Step 3 → [store in box]
...
Game Step 50 → [store in box] → pick 64 random memories → update network
Game Step 51 → [store in box] → pick 64 random memories → update network
```

**Voorbeeld uit de praktijk:** Denk aan een fotoalbum. Je leert niet alleen over je leven
kijkend naar de foto's van vandaag. Je bladert ook door OUDE foto's - een mix van goede herinneringen en
lastige momenten. Dit helpt je patronen gedurende je hele leven te begrijpen, niet alleen vandaag.

---

## Waarom willekeurige steekproeven helpen

Wanneer we willekeurig herinneringen kiezen, verbreken we de correlaties. De robot kan leren van:
- Een herinnering waar de paal perfect was (van 500 stappen geleden)
- Een herinnering waar de paal op het punt stond te vallen (van 20 stappen geleden)  
- Een herinnering waar het geluk had (vanaf stap 3)

Deze willekeurige mix betekent:
✅ De robot leert van verschillende situaties
✅ Elke herinnering kan vele malen worden "herspeeld" (efficiënt gebruik van ervaring)
✅ Het netwerk is niet overmatig aangepast aan recente gebeurtenissen

---

## Mini-batch leren

In plaats van ÉÉN ervaring tegelijk te updaten, updaten we over **64 ervaringen tegelijk**
(een "minibatch"). Dit is als:
- Oude manier: lees één flashcard, overhoor jezelf
- Nieuwe manier: lees 64 verschillende flashcards en overhoor jezelf vervolgens over de mix

Minibatches maken het leersignaal veel betrouwbaarder en minder luidruchtig.

---

## Opwarmperiode

We beginnen niet meteen met leren! De afspeelbuffer heeft eerst enkele herinneringen nodig.
We wachten tot er minimaal **500 ervaringen** in de doos zitten voordat de training begint.

**Voorbeeld uit de praktijk:** Je zou pas proberen een maaltijd te bereiden als je je spullen hebt verzameld
ingrediënten. De opwarmperiode is als boodschappen doen voordat u gaat koken!

---

## Wat de vergelijking laat zien

Wanneer u `dqn_experience_replay.py` uitvoert, ziet u twee leercurven:

| Naïeve DQN | DQN + opnieuw afspelen |
|-----------|-------------|
| Very bumpy | Smoother |
| Frequente crashes (vergeet alles) | Meer consistente verbetering |
| High variance | Lower variance |

De herhalingsversie is meestal:
- Behaalt betrouwbaarder goede scores
- Zakt niet zo vaak terug van 500 naar 30
- Toont een stabielere leervoortgang

---

## De herhalingsbuffer in code

```
ReplayBuffer:
  - capacity: 10,000 memories (oldest are forgotten when full)
  - push(state, action, reward, next_state, done)
  - sample(batch_size=64) → random batch
```

Zie het als een notitieboekje met 10.000 regels. Als deze vol is, wis je de oudste
regel en schrijf de nieuwste. Je studeert altijd vanaf een willekeurige pagina!

---

## Sleutel Woordenschat

| Word | Meaning |
|------|---------|
| **Ervaar herhaling** | Bewaar en hergebruik eerdere ervaringen willekeurig voor training |
| **Herhalingsbuffer** | Het geheugenvak waarin tupels uit het verleden (status, actie, beloning, volgende_status) worden opgeslagen |
| **Gecorreleerde updates** | Wanneer trainingsgegevens van zichzelf afhankelijk zijn (slecht voor het leren!) |
| **Minibatch** | Een kleine willekeurige steekproef van herinneringen die voor één updatestap worden gebruikt |
| **Decorrelatie** | Het verbreken van de verbindingen tussen opeenvolgende ervaringen |

---

## Wat ontbreekt er nog?

Zelfs met een herhalingsbuffer is er nog een probleem: het **bewegende doel**.

Elke keer dat we het netwerk updaten, veranderen de Q-waarden. Maar die bijgewerkte Q-waarden zijn dat wel
OOK gebruikt om het doel voor de VOLGENDE update te berekenen. Het is een cirkel van verwarring!

Dit wordt opgelost door het **Doelnetwerk** — een bevroren kopie van het netwerk dat alleen
wordt elke 100 stappen bijgewerkt. Dat zorgt ervoor dat de "bullseye" een tijdje stil blijft staan, dus de
robot kan er betrouwbaar op richten!