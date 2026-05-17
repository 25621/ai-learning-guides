# Optio-kriittinen arkkitehtuuri

## Suuri idea: Työskentely luvuissa, ei sana sanalta {#the-big-idea-working-in-chapters-not-word-by-word}

Kuvittele, että kirjoitat romaania. Et suunnittele jokaista sanaa ennen kuin aloitat. Sen sijaan ajattelet **luvuissa**: "Luku 1 esittelee sankarin. Luku 2 on tehtävä. Luku 3 on välienselvittely." Jokaisessa luvussa selvität yksityiskohdat edetessäsi.

Juuri näin Option-Critic-arkkitehtuuri ajattelee päätöksistä.

---

## Mikä on "tasainen" agentti? {#what-is-a-flat-agent}

Normaali RL-agentti (kuten opetussuunnitelman vaiheiden 3 ja 4 edustajat) päättää yhden toimenpiteen kerrallaan, jokaisessa vaiheessa. Se on kuin GPS, joka laskee koko reitin uudelleen alusta joka kerta, kun siirrät metrin. Se toimii, mutta se on uuvuttavaa ja hidasta oppia.

---

## What Is an "Option"? {#what-is-an-option}

**Vaihtoehto** on **niminen taito** – minikäytäntö, jonka agentti voi suorittaa useita vaiheita peräkkäin ennen kuin luovuttaa hallinnan takaisin.

Ajattele sitä kuin esimies, joka delegoi asiantuntijoille:

| Kuka | What they do |
|-----|-------------|
| **Ylläpitäjä (metapolitiikka)** | Päättää *kuka* asiantuntija lähettää työhön |
| **Specialist A** | Asiantuntija navigoimaan vasemmassa yläkulmassa |
| **B-asiantuntija** | Oviaukkojen ylittämisen asiantuntija |
| **C-asiantuntija** | Asiantuntija lataamaan kohti tavoitetta |
| **Erikoislääkäri D** | Backup generalist |

Johtaja valitsee asiantuntijan. Asiantuntija työskentelee, kunnes hän päättää, että he ovat valmiita (tätä kutsutaan **lopetuksi**). Sitten johtaja valitsee uudelleen.

---

## Kolme liikkuvaa osaa {#the-three-moving-parts}

Jokaisessa vaihtoehdossa on kolme osaa – ajattele niitä asiantuntijan **työnkuvauksena**:

1. **Aloitus**: Milloin tälle asiantuntijalle voidaan ottaa yhteyttä? *(esim. "Spesialisti A aktivoituu vain lähellä vasenta yläkulmaa.")*
2. **Vaihtoehtojen sisäinen käytäntö**: Mitä asiantuntija tekee työskennellessään? *(esim. "Kävele kohti vasenta yläkulmaa.")*
3. **Päättäminen**: Milloin asiantuntija ohjaa kättä takaisin? *(esim. "Pysäytä, kun olet saavuttanut oviaukon.")*

Option-Criticin kauneus on, että kaikki kolme **oppitaan automaattisesti** – et tee asiantuntijoita käsin. Algoritmi selvittää, että on hyödyllistä, että jokaiselle huoneelle on yksi vaihtoehto tai yksi vaihtoehto kiirehtiä maaliin, kaikki yksinään.

---

## Päivä vaihtoehtokritiikin agentin elämässä {#a-day-in-the-life-of-an-option-critic-agent}

1. Agentti astuu uuteen huoneeseen (tilaan).
2. **Ylläpitäjä** katsoo huonetta ja valitsee vaihtoehdon – esimerkiksi vaihtoehdon 2.
3. **Vaihtoehto 2:n asiantuntija** ottaa vallan: kävelee oviaukkoa kohti askel askeleelta.
4. Jossain vaiheessa vaihtoehto 2 sanoo "Olen valmis" (lopettaminen).
5. **Esimies** herää, valitsee uuden vaihtoehdon uuteen tilanteeseen.
6. Toista.

Vertaa tätä litteään aineeseen: litteä aine tuskaisee jokaisen askeleen. Optio-kriitikko delegoi kokonaisia ​​käyttäytymisosuuksia, jolloin jokainen asiantuntija pääsee hyvin kapeaan tehtäväänsä.

---

## Miksi tämä auttaa? {#why-does-this-help}

Labyrintissa agentin on saavutettava tavoite, joka voi olla 30–50 askeleen päässä. Tasaisessa oppimisessa jokainen askel polulla on yhtä "näkymätön", kunnes palkinto lopulta saapuu lopussa - signaalin on kuljettava taaksepäin kymmenien askeleiden läpi.

Vaihtoehtojen avulla polku jakautuu **alatehtäviin**. Jokainen alatehtävä saa oman minipalkintosignaalinsa (oviaukon saavuttaminen, seuraavaan huoneeseen pääsy). Oppiminen etenee lyhyempien osien kautta. **Agentti oppii nopeammin ongelmissa, jotka vaativat monia vaiheita.**

Tämä on kaiken Hierarchical RL:n perusidea – ja Option-Critic on yksi sen puhtaimmista toteutuksista.

---

## Mitä koodimme tekee {#what-our-code-does}

Käsikirjoitus `option_critic.py` asettaa Option-Critic-agentin **7x7 gridworld-maailmaan**, jolla on kiinteä tavoite. Agentti alkaa mistä tahansa ruudukosta ja sen on navigoitava tavoitesoluun.

Agentilla on neljä vaihtoehtoa, ja hänen on opittava samanaikaisesti:

- Käytäntö jokaiselle vaihtoehdolle (missä kävellä)
- Milloin kukin vaihtoehto on lopetettava (päättymisehto)
- Metapolitiikka, jolla valitaan vaihtoehtojen välillä

Palkinto käyttää **potentiaaliin perustuvaa muotoilua** – agentti saa pienen bonuksen joka askeleelta, jonka se siirtyy lähemmäs tavoitetta, plus +1 sen saavuttamisesta. Tämä tiheä palaute tekee oppimisesta riittävän vakaata, jotta vaihtoehdot toimivat 2 500 jaksossa.

Kukaan ihminen ei koskaan kerro sille, mitä kunkin vaihtoehdon tulisi tehdä. Algoritmi selvittää, mihin ruudukon alueisiin kukin vaihtoehto on erikoistunut.

---

## Mitä kaaviot osoittavat {#what-the-charts-show}

![Optiokritiikin oppimiskäyrät](outputs/option_critic.png)

**Left — Shaped Return:** Korkeampi tuotto tarkoittaa, että agentti saavuttaa tavoitteen luotettavammin *ja* kulkee lyhyempiä polkuja (muokkaus antaa bonuksen askelta lähempänä). Nouseva ja stabiloituva käyrä näyttää koordinoimaan oppivat vaihtoehdot.

**Oikein — askeleita tavoitteeseen:** Vähemmän vaiheita tarkoittaa, että agentti löysi tehokkaamman polun. Laskeva trendi osoittaa vaihtoehdot kypsyvät yhtenäisiksi taidoiksi, jotka ohjaavat agenttia suoremmin kohti tavoitetta.

Tasoitetut käyrät osoittavat yleisen trendin 100 jakson ikkunoissa – jonkin verran kohinaa on normaalia RL:ssä, varsinkin kun useat komponentit (vaihtoehdot, lopetus, metapolitiikka) oppivat samanaikaisesti.

---

## Yhden lauseen yhteenveto {#one-sentence-summary}

> **Option-Critic opettaa agenttia työskentelemään taitojen perusteella yksittäisten vaiheiden sijaan – johtaja valitsee, mikä asiantuntija juoksee, jokainen asiantuntija tekee työnsä ja koko järjestelmä oppii yhdessä samasta palkitsemissignaalista.**
