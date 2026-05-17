# Kinderboerderij-omgevingen verkennen 🦓

## Wat is kinderboerderij?

Als je single-agent RL hebt gedaan, heb je waarschijnlijk **Gymnasium** (de
opvolger van OpenAI Gym). Elke omgeving ziet er hetzelfde uit: `env.reset()` ,
 `env.step(action) → obs, reward, done, info` — een nieuwe *observatie* van de wereld,
een scalair *beloningssignaal*, een *klaar*-vlag met de tekst "game over", en een *info*
woordenboek voor het debuggen van extra's. Die uniformiteit zorgt ervoor dat RL-bibliotheken werken.

**PettingZoo** is precies hetzelfde idee, maar dan voor *meerdere agenten*. Het is een
dierentuin van multi-agentomgevingen — allemaal achter één goed gedefinieerde API:
- **Klassieke speelgoedproblemen**: eenvoudige omgevingen zoals Steen-Papier-Schaar om basisalgoritmen te testen.
- **Coöperatieve rasterwerelden**: agenten die door een raster navigeren om een ​​gedeeld doel te bereiken.
- **Atari-multiplayer**: klassieke competitieve spellen zoals Pong.
- **MPE (Multi-Particle Environment)**: fysische omgevingen met continue ruimte voor complexe coördinatie en competitie.

Als je daar code voor kunt schrijven
werkt op één PettingZoo-omgeving, u kunt inpluggen in een van de andere
met vrijwel geen wijzigingen.

---

## De twee API-stijlen

Instellingen voor meerdere agenten zijn rommeliger dan instellingen voor één agent, omdat er twee agenten zijn
kunnen tegelijkertijd, of op hun beurt, of zelfs in willekeurige volgorde handelen. Kinderboerderij
lost dit op met twee parallelle API's:

### 1) AEC (agent-omgeving-cyclus)

Er treedt één agent tegelijk op. In sommige gevallen loopt de omgeving door agenten
bestelling, en iedereen krijgt:
- een **observatie** — wat ze *nu* zien,
- een **beloning** — de uitbetaling verdiend door de *gezamenlijke* actie in de laatste volledige actie
  ronde (dat wil zeggen, wat er gebeurde als gevolg van het handelen van *alle* agenten, niet alleen
  jij; bij een schaakspel weerspiegelt uw beloning bijvoorbeeld de staat van het bord
  na de laatste zet van je tegenstander, niet alleen die van jou),
- een **beëindigingsvlag** — `True` wanneer de aflevering *natuurlijk* eindigt (bijv.
  schaakmat, iemand wint),
- een **truncatievlag** — `True` wanneer de aflevering een bepaalde tijd *afgebroken* wordt
  limiet voordat een natuurlijk einde wordt bereikt.

Dit is normaal voor **turn-based games** zoals schaken, Go en poker.

```python
env.reset()
for agent in env.agent_iter():
    obs, reward, term, trunc, info = env.last()
    if term or trunc:
        env.step(None)
        continue
    action = my_policy(obs, agent)
    env.step(action)
```

### 2) Parallel

Alle agenten observeren en handelen gelijktijdig bij elke stap.  `step()` duurt een
*woordenboek* van acties en geeft woordenboeken van observaties en terug
beloningen.

Dit is normaal voor **real-time games** zoals MPE (Multi-Particle
Omgevingen waarin alle dot-agents tegelijkertijd bewegen) of multi-agent
rasterwerelden.

```python
obs, info = env.reset()
while env.agents:
    actions = {a: my_policy(obs[a]) for a in env.agents}
    obs, rewards, terms, truncs, info = env.step(actions)
```

De twee stijlen zijn **isomorf** – structureel gelijkwaardig en
interconverteerbaar: elke AEC-omgeving kan automatisch worden ingepakt
eruit zien als een parallelle, en omgekeerd. Kinderboerderij verzendt de conversie
wrappers, zodat u slechts code voor één stijl hoeft te schrijven.

---

## Analogie uit het echte leven

- **AEC = een bordspelavond.** "Alice is aan de beurt. Nu Bob. Nu Carol. Terug naar
  Alice." Degene die daarna beweegt, ziet de laatste bordstatus.
- **Parallel = een videogame voor meerdere spelers.** Alle vier de spelers drukken
  knoppen tegelijkertijd; het spel vernieuwt de wereld 60 keer per seconde.
- **Waarom uniforme API's belangrijk zijn.** Stel je voor dat elke multiplayer-videogame dat zou doen
  had een eigen joystick nodig. Kinderboerderij is de ‘universele joystick’ van MARL.

---

## Wat onze code doet

We bouwen een omgeving in **PettingZoo-stijl** helemaal opnieuw: de **Iterated
Coördinatiespel**. Twee agenten kiezen herhaaldelijk kanaal `0` of `1`:

- Dezelfde keuze → beide krijgen +1
- Andere keuze → beide krijgen -1

De **observatie** die elke agent ontvangt, is de vorige *gezamenlijke actie* —
wat beide agenten de vorige ronde kozen, verpakt in één geheel getal.
Concreet: de laatste actie van elke agent is een van `{start, 0, 1}` (3 staten),
dus het paar codeert als `3 × agent_1_state + agent_2_state` , wat oplevert
9 mogelijke gehele getallen (0 – 8). Geheel getal 0 is de "start" -status - het signaleert
dat er nog geen actie is ondernomen (het allereerste begin van een aflevering).
Een aflevering duurt 25 stappen, dus het maximale totaalrendement is +25 per agent
en het minimum is −25. **Willekeurig spel scoort ≈ 0** omdat bij elke stap
twee onafhankelijke willekeurige agenten kiezen elk 0 of 1 met gelijke waarschijnlijkheid:
ze komen 50% van de tijd overeen (+1) en verschillen 50% van de tijd (−1), waardoor ze verschillen
een verwachte beloning per stap van 0,5 × (+1) + 0,5 × (−1) = **0**. Opgeteld
over 25 stappen is het verwachte episoderendement ook 0.

Wij dan:

1. **Demonstreer de AEC-interface** met een willekeurige uitrol – dit bevestigt
   de basis AEC-lus: `agent_iter()` levert de agent op wiens beurt het is,
    `last()` leest de huidige observatie en de verzamelde beloning van die agent,
   en `step()` levert de gekozen actie terug aan de omgeving.
2. **Train twee onafhankelijke Q-leerlingen via de parallelle interface**.
   Elke agent houdt zijn eigen Q-tabel bij, vastgelegd door de **gezamenlijke actie
   observatie** (het enkele gehele getal dat codeert voor wat *beide* agenten deden
   vorige ronde), zodat het kan leren: "Toen we de vorige keer allebei 0 hadden gekozen, zou ik dat moeten doen
   kies opnieuw 0."
3. **Probeer de echte `pettingzoo`-bibliotheek** te importeren en er een uit te rollen
   ingebouwde omgevingen (steen-papier-schaar) met een willekeurig beleid. Als
   PettingZoo is niet geïnstalleerd, bij een vriendelijk bericht slaan wij deze stap over.

### Wat je zou moeten zien

| Stage | Expected |
|-------|----------|
| Willekeurige uitrol (AEC)            | Gemiddelde (gemiddelde) episode-return in de buurt van **0**: willekeurige agenten kiezen onafhankelijk van elkaar kanalen, waarbij ze in ongeveer gelijke mate matchen en mismatchen. |
| Onafhankelijke Q-leerlingen (parallel) — eerste 100 afleveringen | Ongeveer **0** — nog steeds grotendeels willekeurig terwijl agenten verkennen. |
| Onafhankelijke Q-leerlingen — laatste 100 afleveringen             | Zeer positief, **+20 tot +25** — **coördinatie is ontstaan**: beide agenten hebben geleerd elke ronde betrouwbaar hetzelfde kanaal te kiezen. |

De plot `outputs/pettingzoo_coordination.png` toont individuele afleveringen
rendementen (grijs) en een rollende **Gemiddelde** curve (blauw). Het gemiddelde wordt luidruchtig gladgestreken
afleveringen zodat je de trend kunt zien: de agenten bewegen ongecoördineerd willekeurig
speel dichtbij ~0 richting stabiele **coördinatie** dichtbij ~+25. De onderbroken groene lijn
markeert het perfecte coördinatieplafond.

Als `pettingzoo` is geïnstalleerd, wordt het script ook uitgerold
 `pettingzoo.classic.rps_v2` om te bewijzen dat het script tegen de werkelijkheid werkt
bibliotheek op precies dezelfde manier als het werkt op onze handgerolde omgeving. Om in te schakelen
dat gedeelte:

```bash
source ../../venv/bin/activate
pip install "pettingzoo[classic]"
```

---

## Waarom eerst een aangepaste omgeving bouwen?

Omdat **de API de les is.** (Begrijpen hoe je de interactie tussen meerdere agenten en de omgeving structureert is belangrijker dan de specifieke spelregels.) Multi-agent RL heeft veel smaken
(turn-based, real-time, coöperatief, competitief, gemengd), en ze allemaal
passen in het AEC / Parallel-patroon. Zodra je die twee hebt geïmplementeerd
loops is elke PettingZoo-omgeving slechts een kwestie van het aansluiten van een
verschillende env-constructor - de trainingscode blijft hetzelfde.

Dit is precies hoe Gymnasium single-agent RL veranderde: door de
omgeving een zwarte doos achter een uniforme interface.

---

## Waar onafhankelijk Q-learning helpt en pijn doet

Coördinatiespellen zijn *vergevingsgezind* – de agenten delen het beloningsteken, dus
hun belangen komen overeen. Onafhankelijke leerlingen kunnen dit met plezier oplossen, omdat elke verbetering van de ene agent de andere helpt.

In **adversarial** games (RPS) oscilleert onafhankelijke Q-learning voor altijd (terwijl de ene agent zich aanpast, verandert de andere zijn strategie om tegen te gaan, wat leidt tot eindeloze achtervolging).
In **gedeeltelijk waarneembare** spellen kan het helemaal niet leren omdat de
"observatie" is slechts één onderdeel van de staat (een agent kan worden gestraft voor een goede actie alleen maar omdat hij niet kon zien wat de andere agent aan het doen was). Kinderboerderij omvat beide
soorten omgevingen, zodat u deze faalmodi zelf kunt zien.

---

## Sleutelwoorden om te onthouden

| Word | Meaning |
|------|---------|
| **Kinderboerderij**     | Het Gymnasium van multi-agent RL: een bibliotheek met gestandaardiseerde MARL-omgevingen |
| **AEC**            | Agent-omgeving-cyclus: één agent handelt per stap (turn-based) |
| **Parallelle API**   | Alle agenten handelen gelijktijdig bij elke stap |
| **MPE**            | Multi-Particle Environment, een populair coöperatief/competitief testbed dat wordt meegeleverd met PettingZoo (vaak met bewegende stippen die door op fysica gebaseerde taken navigeren). |
| **CTDE**           | Gecentraliseerde training, gedecentraliseerde uitvoering – train met een mondiaal perspectief (toegang tot alle staten), inzet met alleen lokale obs (elke agent handelt vanuit zijn eigen beperkte visie). |
| **Onafhankelijk Q-leren** | Elke agent voert standaard Q-learning uit (het standaard, ongewijzigde Q-learning-algoritme), waarbij wordt genegeerd dat er andere leerlingen bestaan. |

---

## Samenvatting van één zin

> **PettingZoo geeft elke omgeving met meerdere agenten dezelfde vorm — dus de
> code die je vandaag schrijft, werkt morgen nog steeds in een totaal ander spel.**

Zodra de twee API-stijlen een tweede natuur zijn, kunt u overstappen op MADDPG
(gecentraliseerde criticus voor middelen met continue controle), QMIX (waardemenging voor
coöperatieve teams), MAPPO (multi-agent PPO) of een andere moderne MARL
algoritme — de omgevingskant van uw code hoeft nooit te veranderen.