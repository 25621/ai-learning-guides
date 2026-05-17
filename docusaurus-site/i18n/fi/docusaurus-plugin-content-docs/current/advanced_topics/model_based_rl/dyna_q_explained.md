# Dyna-Q: oppii nopeammin kuvittelemalla 🧠

## Mikä se on? {#what-is-it}

Kuvittele Mia-nimistä lasta, joka oppii navigoimaan uudessa koulussaan. Joka päivä hän kävelee
käytävillä ja löytää uusia asioita: "Kirjasto on kahvilan ohi",
"Hra Smithin huone on yläkerrassa lähellä portaikkoa."

**Tavallinen Q-oppiva** opiskelija oppii vain siitä, mitä hän tekee *tänään*. Jos tänään
hän käveli juuri luokalta kahvilaan, hän päivittää vain muistiaan
tuo yksi polku.

**Dyna-Q** -opiskelija on erilainen. Jokaisen oikean kävelyn jälkeen hän istuu alas hetkeksi
minuutti ja **toistot päässään** useita menneitä kävelyjä hän muistaa. Jokainen toisto
vahvistaa hänen henkistä karttaansa. Muutaman viikon kuluttua hän tuntee koulun läpikotaisin -
ei siksi, että hän käveli enemmän, vaan koska hän ** ajatteli enemmän mitä hän
nähnyt jo**.

Juuri tätä Dyna-Q tekee RL-agentille: se oppii oikeasta
kokemus **ja** kuvitteellisesta kokemuksesta, joka on saatu mallista, jota se rakentaa
tapa.

---

## Kolme ainesosaa {#the-three-ingredients}

Dyna-Q on "Q-oppiminen + malli + suunnittelu". Yksi todellinen askel tekee **kolme** työtä:

1. **Direct RL** — tavallinen Q-learning-päivitys `(s, a, r, s')`.
2. **Mallioppiminen** — kirjoita ylös: "Kun tein *a*:lla *s*:lla, sain *r*:n ja päädyin *s'*:iin."
3. **Suunnittelu** — valitse *n* satunnainen `(s, a)` parit mallin muistista ja tee
   *n* lisää Q-learning-päivityksiä, **teeskelen** nämä vaiheet tapahtuivat juuri.

Kolmas askel on taikuutta. kanssa `n = 50`, jokainen todellinen askel maailmassa aiheuttaa
**51 päivitystä** Q-taulukkoon. Agentti oppii ~50x nopeammin - reaalivaiheisesti -
kuin puhdas Q-oppija.

---

## Kuva Loopista {#a-picture-of-the-loop}

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

Malli on vain hakutaulukko:
`(state, action) → (reward, next_state)`. Halpa rakentaa, halpa tiedustella.

---

## Esimerkkejä tosielämästä {#real-life-examples}

- **Shakin opiskelu.** Suurmestarit viettävät tuntikausia pelaamalla omia pelejään ja
  hallitsevat pelejä päässään. Jokainen uusinta on "suunnittelua" - ylimääräistä oppimista
  jo tapahtuneita kokemuksia.
- **Musiikko harjoittelee asteikkoja.** Soitettuaan kerran hankalaa baaria he
  harjoitella sitä henkisesti vielä kymmenen kertaa ennen kuin jatkat. Sormet eivät ole
  liikkuvat, mutta aivot päivittyvät.
- **Itse ajava auto.** Kun se käy joutokäynnillä punaisella valolla, se toistaa viimeisen
  sata kaistanvaihtoa simulaatiossa hienosäätääkseen sen politiikkaa polttamatta
  renkaat.

---

## Mitä koodimme tekee {#what-our-code-does}

Käytämme klassista **Dyna Maze** ([Sutton & Barto, kuva 8.2](http://incompleteideas.net/book/the-book.html)): 6×9 ruudukko
jotkut seinät, alku `S` keskellä vasemmalla ja maali `G` oikeassa yläkulmassa.

Käytämme kolmea muunnelmaa, joista jokainen on keskimäärin yli 30 satunnaista siementä:

| Asetus | Suunnittele vaiheita todellista askelta kohti | Merkitys |
|---------|------------------------------|---------|
| `n = 0` | 0 | tavallista Q-oppimista |
| `n = 5` | 5 | vähän kuviteltua harjoitusta |
| `n = 50` | 50 | paljon kuviteltua harjoitusta |

Käsikirjoitus raportoi **todellisten vaiheiden keskimääräisen määrän jaksoa kohden**
koulutus etenee. Vähemmän vaiheita tarkoittaa, että agentti on oppinut suoremman
polku päämäärään.

### Mitä sinun pitäisi nähdä, kun käytät sitä {#what-you-should-see-when-you-run-it}

Lyhin polku tässä sokkelossa on ~9 askelta; ε-ahneella etsinnällä a
hyvin koulutettu agentti tekee keskimäärin ~10 askelta jaksoa kohden. Juokse 50 jaksoa ja
kaikki kolme asetusta sulautuvat yhteen - ero on *kuinka nopeasti*:

| Asetus | Vaiheet per jakso (viimeiset 10 jaksoa) | Mitä se tarkoittaa |
|---------|--------------------------------:|---------------|
| `n = 0`   | ~10 | Lähestyi – mutta kesti noin 30–50 vaellusjaksoa päästä tänne |
| `n = 5`   | ~10 | Lähestyi ~10 jaksossa |
| `n = 50`  | ~10 | Lähestyi noin 3–5 jaksossa |

Mielenkiintoinen signaali on *oppimiskäyrä*, ei lopullinen luku. Juoni
tallennettu kohteeseen `outputs/dyna_q.png` näyttää kolme käyrää, jotka sukeltavat kohti lattiaa erittäin kohdalla
eri hinnat: `n = 50` saavuttaa sen muutamassa jaksossa `n = 0` on
edelleen kiipeämässä alas hyvin juoksuun. (Tällaisessa pienessä deterministisessä sokkelossa,
tavallinen Q-oppiminen onnistuu lopulta – Dyna-Q tarvitsee vain paljon vähemmän todellista
jaksot, mikä on koko pointti ympäristöissä, joissa todelliset askeleet ovat kalliita.)

---

## Miksi se toimii niin hyvin tässä sokkelossa {#why-it-works-so-well-on-this-maze}

Kaksi syytä:

1. **Ympäristö on deterministinen.** Jokainen `(s, a)` antaa aina saman
   `(r, s')`, joten malli on tarkka yhden käynnin jälkeen. Kuvitteellinen kokemus on
   yhtä hyvä kuin todellinen kokemus.
2. **Todelliset askeleet ovat kalliita, kuvitellut ovat ilmaisia.** Jokainen kuviteltu päivitys
   on vain muutama pöytähaku, kun taas todellinen askel vaatii agentin kävelemistä.
   Kun todellinen vuorovaikutus on kallista (ajattele: oikea robotti, oikea peli), Dyna-Q
   on erittäin näytetehokas.

---

## Missä Dyna-Q kamppailee {#where-dyna-q-struggles}

- **Stokastiset ympäristöt.** Jos `(s, a)` voi johtaa moniin erilaisiin `s'`
  arvot, "muista viimeinen tulos" -malli valehtelee sinulle. Korjaus: kaupassa käynnit lasketaan
  tai harjoittele todennäköisyysmallia.
- **Ei-kiinteät ympäristöt.** Jos maailma muuttuu – esimerkiksi a
  auki ollut oviaukko sulkeutuu yhtäkkiä tai näkyviin tulee pikakuvake – malli
  vanhenee ja antaa vääriä ennusteita. **Dyna-Q+** korjaa tämän
  *etsintäbonuksen* lisääminen: osavaltiot, joissa ei ole käyty uudelleen pitkään aikaan
  aika saada pieni ylimääräinen palkkio, joka tönäisi agenttia palaamaan tarkistamaan
  onko maailma muuttunut.
- **Suuret tilatilat.** Sanakirja kytkettynä `(s, a)` ei skaalaudu
  miljooniin valtioihin tai jatkuviin tiloihin. Se on juuri se aukko
  **oppineet (hermoverkon) maailmanmallit** täytä - katso `world_model.py` seuraavaksi.

---

## Avainsanat muistaa {#key-words-to-remember}

| sana | Merkitys |
|------|---------|
| **malli**       | Muisti `(state, action) → (reward, next_state)` |
| **Suunnitteluvaihe** | Q-päivityksen tekeminen mallin kuviteltujen tietojen avulla |
| **Suora RL**   | Q-päivitys, joka käyttää todellista dataa |
| **Näytteen tehokkuus** | Mittaa, kuinka tehokkaasti tekoälymalli tai -algoritmi käyttää harjoitustietoja tietyn suoritustason saavuttamiseksi |
| **Dyna**        | Suttonin arkkitehtuuri, joka yhdistää oppimisen + suunnittelun |

---

## Yhden lauseen yhteenveto {#one-sentence-summary}

> **Dyna-Q oppii tekemisestä JA kuvittelemisesta – ja kuvitteleminen on ilmaista.**

Tämä ajatus nykyaikaisessa hermomuodossaan antaa voimat joihinkin vahvimmista RL-agenteista
koskaan rakennettu (MuZero, Dreamer, World Models).
