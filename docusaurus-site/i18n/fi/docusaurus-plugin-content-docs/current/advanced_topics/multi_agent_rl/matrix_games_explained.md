# Matrix Games: Simplest Multi-Agent World 🎲

## Mikä on Matrix-peli?

Kuvittele, että sinä ja ystäväsi valitsette kumpikin käsimerkin - **kiven, paperin tai sakset** -
*samaan aikaan*. Ette näe toistenne valintaa. Voittajan päättää
pieni pöytä:

|        | Rock | Paperi | Sakset |
|--------|:----:|:-----:|:--------:|
| Rock     |  0,0  | -1,+1 | +1,-1 |
| Paperi    | +1,-1 |  0,0  | -1,+1 |
| Sakset | -1,+1 | +1,-1 |  0,0  |

Tuo pöytä on pelin *koko maailma*. Ei liikettä, ei aikaa, ei karttaa.
Vain yksikertainen päätös. Kutsumme tätä **matriisipeliksi**, koska voitto
matriisi on koko ympäristö.

Matrix-pelit ovat siistein paikka opiskella **multi-agent RL** -peliä, koska
Ainoa asia, joka voi muuttua harjoituksen aikana, on jokaisen pelaajan *käytäntö* –
todennäköisyys valita jokainen toiminto.

---

## Miksi se on "Multi-Agent"

Yhden agentin RL:ssä ympäristö on kiinteä: tuuli puhaltaa aina samaa
tavalla, lattia ei koskaan liiku. Agentti parantaa ja lopulta voittaa.

Matriisipelissä "ympäristösi" on *toinen oppimisagentti*. Kuten he saavat
älykkäämpi, mikä on sinulle hyvä liike *muuttuu*. Tätä kutsutaan
**ei-stationaarisuus**, ja se on usean agentin RL:n keskeinen ongelma.

> Jos jatkat rockin pelaamista, vastustajasi alkaa lopulta pelata Paperia
> aina. Joten vaihdat Saksiin. Joten he siirtyvät Rockiin. Joten vaihdat
> Paperi... ja niin edelleen. "Paras liike" ei koskaan pysy paikallaan.

Klassinen ratkaisu on **sekoitetut strategiat**: älä valitse yhtä toimenpidettä
deterministisesti – satunnaista tavalla, jota vastustaja ei voi hyödyntää.

---

## Kolme peliä, joita pelaamme

### 1) kivi-paperi-sakset (nollasumma)
- Yhden pelaajan voitto on toisen tappio.
- **Nash-tasapaino** on: jokainen pelaaja valitsee jokaisen toiminnon todennäköisyydellä
  ⅓.  Mikä tahansa poikkeama on hyödynnettävissä.
- Odotamme kahden Q-oppijamme huojuvan noin ⅓-⅓-⅓ – ei koskaan täydellisesti
  tasaista, koska aina kun toinen ajautuu, toinen reagoi.

### 2) Vangin dilemma (yleinen summa)
Kaksi epäiltyä kuulustetaan erikseen:

|           | Tee yhteistyötä | Vika |
|-----------|:---------:|:------:|
| Tee yhteistyötä |   3, 3    |  0, 5  |
| Vika    |   5, 0    |  1, 1  |

- "Defect" voittaa "Yhteistyö" riippumatta siitä, mitä toinen tekee - se on a
  **dominoiva strategia**.
- Molemmat pelaajat ovat rationaalisia → molemmat viallisia → molemmat saavat 1, vaikka
  (Yhteistyö, Yhteistyö) oli 3 kutakin. Itsekäs paras vastaus tuhoaa ryhmän
  hyvinvointia.
- Odotamme Q-oppimisen konvergoivan selkeästi (vika, vika).

### 3) Hirvenmetsästys (koordinointi)
Kaksi metsästäjää voi yhdessä kaataa poltan (valtava palkinto) tai kumpikin tyytyä a
jänis (pieni mutta turvallinen palkinto):

|       | Stag | Jänis |
|-------|:----:|:----:|
| Stag  | 4, 4 | 0, 3 |
| Jänis  | 3, 0 | 2, 2 |

- (Stag, Stag) on ​​**voittodominoiva** – paras molemmille.
- (Hare, Hare) on **riskidominoiva** — turvallista, jos et luota kumppaniisi.
- Tulos riippuu alkuolosuhteista: itsenäiset Q-oppijat päätyvät usein
  *pahimmassa* (Hare, Hare) tasapainossa, koska jänikset ovat turvallisempia oppia.

---

## Esimerkkejä tosielämästä

- **Hinnoittelu duopolissa.** Kaksi kahvilaa samalla kadulla kukin poimii a
  hinta joka aamu. Voittomatriisin muoto ratkaisee, ovatko ne
  päätyä korkeaan "osuuskuntahintaan" (hyvä heille, huono asiakkaille) tai
  halpa hinta.
- **Verkkoprotokollat.** Reitittimet ja lähettäjät valitsevat ajoitusstrategiat; the
  verkon ruuhkautumisen lopputulos määräytyy matriisipelin kaltaisen voiton mukaan
  selviäminen vs. perääntyminen.
- **Tarjous huutokaupassa.** Jokainen tarjoaja valitsee tarjouksen, vaikka hän ei tiedä muita;
  voitot riippuvat koko vektorista. Nashin tasapaino on *tarjous
  strategia*, ei yksittäinen numero.

---

## Mitä koodimme tekee

Jokaiselle pelille me:
1. Luo kaksi valtiotonta Q-oppijaa (Q on vain yksi numero toimintoa kohden – siellä
   eivät ole osavaltioita yhden laukauksen pelissä).
2. Kierrä 20 000 askelta. Jokainen vaihe: molemmat agentit valitsevat ε-ahneen toiminnon
   samanaikaisesti saada palkinto, päivittää Q-arvonsa.
3. Seuraa kunkin agentin **empiiristä toimintataajuutta** 500 askeleen
   ikkuna. Sen sijaan, että katsoisimme vain abstrakteja todennäköisyyksiä, laskemme, mitä toimintoja he todella valitsivat äskettäin (esim. "viimeisen 500 kierroksen aikana he pelasivat rockia 40 % ajasta"). Tämä antaa meille reaaliaikaisen käytännön kuvan heidän muuttuvasta strategiastaan.
4. Piirrä taajuudet ajan mukaan, tallenna kohteeseen `outputs/<game>.png`ja tulostaa
   lopulliset Q-arvot.

### Mitä sinun pitäisi nähdä

| Peli | Juonen odotettu lopputulos |
|------|------------------------------|
| **kivi-paperi-sakset** | Molemmat pelaajat leijuvat lähellä ⅓-⅓-⅓, mutta tärisevät näkyvästi. Kaaret jahtaavat toisiaan – klassista pyöräilykäyttäytymistä. |
| **vangin dilemma** | Molempien pelaajien "vika"-taajuus nousee nopeasti ~1.0:aan. "Yhteistyö" murskataan. |
| **Stag metsästys** | Useimmat satunnaiset siemenet asettuvat (jänis, jänis). Jotkut onnensiemenet saavuttavat (Stag, Stag) – kokeile vaihtaa käsikirjoituksen siementä ja katsella sen kääntymistä. |

---

## Missä itsenäinen oppiminen taukoja

Agenttimme ovat *riippumattomia* – he näkevät vain oman palkkionsa, eivät koskaan
vastustajan toiminta tai Q-arvot. Tämä on yksinkertaisin lähtökohta ja se on
rajat:

- Se **ei voi taata konvergenssia** yleissummapeleissä.
- Se voi juuttua **huonoon tasapainoon** (Stag Hunt).
- Se **ei voi mallintaa vastustajaa**.

Todelliset monen agentin algoritmit korjaavat tämän perustelemalla selkeästi toista
oppija. Tässä on mitä kukin tekee, englanniksi:

| Algoritmi | Ydin idea                                                                                                                                                                                                                                                                               | Analogia tosielämästä |
|-----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|
| **Fiktiivinen leikki** | Pidä kirjaa siitä, kuinka usein vastustajasi on valinnut kunkin toiminnon. Oletetaan, että huomenna he tekevät sen, mitä he ovat aina tehneet – valitse sitten oma paras vastaus tähän uskomukseen.                                                                                                          | Tarkkaile vastustajan tottumuksia monien shakkipelien aikana ja säädä avauksesi sen mukaan. |
| **CFR (Counterfactual Regret Minimation)** | Kysy jokaisen kierroksen jälkeen *"Kuinka paljon kaduin, että en valinnut toistensa toimintaa?"* Siirrä vähitellen todennäköisyyttä toimiin, joita kadut ohittamista. Käytetään pokerissa, koska se käsittelee **epätäydellisen tiedon** pelejä (et näe vastustajan kortteja).                                  | Pokerikäden jälkeen pelaaminen uudelleen ja ajattelu: *"Olisi pitänyt panostaa enemmän - teen sen ensi kerralla."* |
| **LOLA (Oppiminen vastustajan oppimistietoisuuden kanssa)** | Gradienttiaskeli selittää sen tosiasian, että vastustaja on *myös* gradienttiaskeleva. Optimoi oman päivityksesi ennakoiden samalla vastustajan seuraavaa päivitystä – kaksi askelta edellä yhden sijasta.                                                                                    | Neuvotella sopimusta ja ajatellut: *"Jos tarjoan X, he vastasivat Y:llä, joten minun pitäisi aloittaa Z."* |
| **MADDPG (Multi-Agent Deep Deterministic Policy Gradient)** | Jokaisen agentin *kriitikko* (arvoestimaattori) on koulutettu **globaalinäkymällä**: se näkee jokaisen havainnot ja toimet. *toimija* (käyttöön otettava käytäntö) käyttää edelleen vain paikallisia tietoja – tämä on CTDE (Centralized Training with Decentralized Execution) -malli. | Koripallovalmentaja, joka katselee koko kenttää (keskitetty kriitikko), mutta opettaa jokaista pelaajaa reagoimaan vain siihen, mitä näkee (hajautettu näyttelijä). |

Mutta itsenäinen Q-oppiminen on oikea ensimmäinen askel. Sinä näet
ei-stationaarisuusongelma osui naamaan, ja korjaukset ovat järkeviä
jälkeenpäin.

---

## Avainsanat muistaa

| sana | Merkitys |
|------|---------|
| **Matoitusmatriisi** | Taulukko, joka määrittelee yhden laukauksen monen agentin pelin |
| **Nash-tasapaino** | Käytäntöprofiili, jossa yksikään toimija ei voi parantaa poikkeamalla |
| **Sekastrategia** | Käytäntö, joka satunnaistaa useita toimintoja |
| **Ei-stationaarisuus** | Ympäristö (= muut toimijat) muuttuu jatkuvasti oppiessaan |
| **itsenäinen oppija** | Agentti, joka jättää huomioimatta muiden oppijoiden olemassaolon |
| **nollasumma** | Yhden agentin voitto on täsmälleen toisen tappio |
| **Yleinen summa** | Molemmat agentit voivat voittaa, molemmat voivat hävitä tai mitä tahansa siltä väliltä |

---

## Yhden lauseen yhteenveto

> **Matriisipeleissä "ympäristö" on toinen oppija – siis paras liike
> jatkaa liikkeessä.**

Tämä on jokaisen tapaamasi moniagenttialgoritmin perusta
myöhemmin itsepelistä MADDPG:hen MARLiin viestinnällä.
