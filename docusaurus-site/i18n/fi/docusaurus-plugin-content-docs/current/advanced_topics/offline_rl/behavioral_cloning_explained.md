# Behavioral Cloning (BC) 🐒

## Mikä se on?

Kuvittele, että haluat oppia pelaamaan tennistä. Katsot satoja tunteja
tallentaa Wimbledon-ottelut ja yksinkertaisesti **kopioi mitä pelaajat tekevät**. sinä
älä ajattele, oliko heidän laukauksensa *paras* laukaus – sovi vain yhteen
kehosi asento omaansa kohtaan ja heilu samalla tavalla.

Se on käyttäytymiskloonausta. **Ei palkintoa. Ei suunnittelua. Vain jäljitelmä.**

RL-termeillä: ota tietojoukko `(state, action)` parit ja harjoitus a
hermoverkko ennustaa tilan toimintaa, täsmälleen kuten an
kuvan luokittelumalli ennustaa kissa vs-koira. "Etiketti" on mikä tahansa
tietojen kerääjän toimenpiteisiin.

---

## Kuinka se eroaa "todellisesta" offline-RL:stä

| Lähestymistapa | Käyttääkö palkintoja? | Voiko data päihittää? |
|----------|---------------|---------------------|
| **BC**   | ❌ ei         | ❌ ei – parhaimmillaan vastaa keskimääräistä tiedon laatua |
| **CQL** (ja ystävät) | ✅ kyllä | ✅ kyllä – osaa ommella hyviä siirtymiä sekoitetuista tiedoista |

BC on RL:n "ohjatun oppimisen näkymä". Se on usein uskomattoman yksinkertaista
yllättävän vahva ja universaali lähtökohta. **Jos offline-RL
Algoritmi ei voi voittaa BC:tä samassa tietojoukossa, se ei ole tehnyt mitään.**

---

## Esimerkkejä tosielämästä

- **Oppii ajamaan kojelautakameramateriaalista.** Katso tietä, ennusta
  ihmisen käyttämä ohjauspyörän kulma. Kaksi maamerkkiesimerkkiä:
  - **ALVINN (1989)** — ensimmäinen hermoverkkoohjain; pieni 3-kerroksinen
    kameralla koulutettu verkko + lasersisääntulot pakettiauton ohjaamiseen valtateiden yli.
  - **NVIDIA PilotNet (2016)** – moderni syvällinen CNN-verkko, joka on koulutettu päästä päähän
    kojelautakameramateriaalia; kaistanpidon ja perusohjauksen oppinut puhtaasti
    jäljittelee ihmiskuljettajia, ei käsintehtyjä sääntöjä.
- **Oppipoika kopioimassa mestarikokkia.** "Mitä tahansa kokki tekee, minä teen."
  Toimii hyvin, jos kokki on loistava; tuottaa huonon kokin, jos kokki on
  huono.
- **GitHub Copilot.** Automaattinen täydennys on koulutettu ennustamaan "mitä koodia
  olisiko seuraava ihmistyyppi?" - puhdasta lähdekoodilokien jäljitelmää.
- **Jäljittelet vanhempaa sisarustasi.** Lapset tekevät tätä vuosia ennen kuin he tekevät
  ala pohtia *miksi* vanhempi sisarus tekee mitä tekee.

---

## Matematiikka (yksi rivi)

Jokaiselle `(s, a)` tietojoukossa minimoi:

```
loss = -log π(a | s)        (cross-entropy)
```

Siinä se. käytäntö `π` on vain MLP, joka tulostaa toimintalogit;
koulutus on sama kuin MNIST. Puretaan ammattikieltä:
- **`π` (Pi):** "käytännön" vakiosymboli – sääntö tai hermoverkko, joka päättää mitä tehdä.
- **MLP (Multi-Layer Perceptron):** Normaali perushermoverkko.
- **Logitit:** Raa'at, normalisoimattomat pisteet, joita verkko sylkee ennen kuin muutamme ne todennäköisyyksiksi.
- **Ristientropia:** Vakiokaava mallin rankaisemiseksi, kun se määrittää oikean vastauksen alhaisen todennäköisyyden.
- **MNIST:** Kuuluisa käsinkirjoitettujen numeroiden aloittelijatietojoukko.

Agentin kouluttaminen pelaamaan peliä BC:n kautta on kirjaimellisesti identtistä verkon kouluttamiseen tunnistamaan käsin kirjoitetut numerot MNIST:ssä. MNIST:ssä tulo on kuva ja lähtö on numero (0-9). BC:ssä syöte on pelin tila ja tulos on toiminta (esim. "siirrä vasemmalle").

---

## Mitä koodimme tekee

Skripti `behavioral_cloning.py`:

1. **Lataa kaikki neljä tietojoukkoa**, jonka on rakentanut `d4rl_dataset.py`
   (`random`, `medium`, `expert`, `medium-replay`).
2. Jokaiselle tietojoukolle **opettaa erillisen BC-käytännön** 10 000 gradientille
   ristientropian vaiheet. Palkinto-sarake jätetään kokonaan huomiotta.
3. Joka 2 500 askelta **arvioi** nykyisen käytännön ottamalla sen käyttöön
   ahneesti todellisessa CartPole-v1-ympäristössä (20 jaksoa, keskiarvo).
4. Tontit:
   - Pylväskaavio: lopullinen BC-tuotto tietojoukkoa kohti.
   - Oppimiskäyräkaavio: kuinka kukin BC-versio kiipeää harjoittelun yli.

---

## Mitä sinun pitäisi nähdä

Tyypillinen tulos tulostaa:

```
Final evaluation returns:
  BC on random          ->    ~20  ± a few   (≈ random play)
  BC on medium          ->   ~150  ± large   (≈ the medium policy)
  BC on expert          ->   ~480  ± small   (≈ the expert policy)
  BC on medium-replay   ->    ~60  ± large   (≈ the AVERAGE of mixed data)
```

Pylväskaavio tekee tarinasta ilmeisen: **BC:n paluu seuraa tietojoukon
keskimääräinen tuotto.** Se ei voi ylittää tätä kattoa, koska se ei voi ylittää
mieluummin sekalaisen tietojoukon "hyviä" osia "huonojen" osien sijaan - molemmat ovat
yhtä päteviä jäljitelmäkohteita.

Tämä on lyöntiviiva: **BC perii datan enimmäismäärän.**

---

## BC vs CQL – puhtain vertailu

**Keskitoiston** tietojoukossa (realistisin, sekalaatuisin tapaus):

| menetelmä | Suunnilleen lopullinen tuotto | Miksi? |
|--------|--------------------:|------|
| eKr     | ~60   | Jäljittelee epäonnistuneiden aikaisten ajojen *keskiarvoa* + myöhempiä hyviä |
| CQL    | ~400+ | Käyttää palkintoja suosiakseen korkean Q-siirtymiä; ompelee hyvän käytännön sekoitetuista tiedoista |

Joten CQL **päihittää tiedot**, BC ** vastaa tiedot**. Siinä kaikki
syy offline RL on tutkimusala eikä vain "teke jäljitelmiä".
oppiminen". Kun data on sekalaatuista (jota todelliset lokit ovat aina),
palkitsevat menetelmät toipuvat enemmän.

**asiantuntija**-tiedoissa vertailu kääntyy: BC vastaa asiantuntijaa (~480). Saatat ihmetellä, miksi CQL "sitoutuu" tähän häviämisen sijaan. Koska CQL on suunniteltu *konservatiiviseksi* ja rankaisemaan toimia, joita ei näy tietojoukossa, se päätyy tekemään täsmälleen sen, mitä asiantuntija teki. Se ei voi voittaa asiantuntijaa (koska maksimipistemäärä on jo saavutettu), mutta se ei myöskään riko asiantuntijan strategiaa aktiivisesti. Se liittyy vain BC:n suorituskykyyn.

Tämä on kuuluisa "datan laatu vs algoritmi" -kompromissi:

```
                            EXPERT  data  →  BC wins, CQL ties
   Algoritmin hienostuneisuus ↑
                            MIXED   data  →  CQL clearly beats BC

                            RANDOM  data  →  Everybody fails; need exploration
```

---

## Missä BC asuu modernissa RL:ssä

- **Esikoulutus online-RL:lle.** Monet nykyaikaiset järjestelmät (RT-1, Voyager,
  pelibotit) aloittavat BC:llä esittelyissä ja hienosäädä sitten
  verkossa PPO/SAC:n kautta.
- **RLHF.** InstructGPT:n vaihe 1 on valvottua hienosäätöä – puhdas BC päällä
  ihmisen kirjoittamia vastauksia. PPO + palkintomalli tulee myöhemmin.
- **DAgger (Ross et al., 2011).** Nerokas laajennus korjaamaan **kompositio-virhe** -ongelman.
  *Miksi yhdistämisvirhe on ongelma, jos BC-kloonat täydellisesti?* Vaikka BC-malli olisi 99 % tarkka, tuo 1 %:n virhe lopulta tapahtuu. Kun se tapahtuu, agentti siirtyy tilaan, jota se ei ole koskaan nähnyt täydellisesti ohjatussa tietojoukossa. Koska se on hämmentynyt, se tekee suuremman virheen, siirtyen vielä kauemmaksi tunnetusta tiedosta, mikä johtaa täydelliseen epäonnistumiseen (kuten kalliolta ajaminen).
  *Korjaus:* Voisimme vain pyytää asiantuntijaa ajamaan ikuisesti, mutta asiantuntija-aika on kallista. Sen sijaan DAgger antaa BC-käytännön ajaa. Kun käytäntö tekee virheen ja ajautuu omituiseen tilaan, pidämme tauon, kysymme asiantuntijalta "mitä sinä tekisit *täällä*?" ja lisäämme sen tietojoukkoon. Me vain "kysymme asiantuntijalta uudelleen BC-käytännön vierailemissa tiloissa", koska tarvitsemme asiantuntijan vain opettamaan meille, kuinka toipua omista virheistämme, sen sijaan, että kyselisimme niitä aina.
- **Decision Transformer (Chen et al., 2021).** "Älykäs" eKr.
  määrittää toiminnan ennusteen halutuksi *paluuksi*,
  Pohjimmiltaan offline RL:n muuttaminen takaisin seuraavan tunnuksen ennusteeksi.

---

## Avainsanat muistaa

| sana | Merkitys |
|------|---------|
| **Jäljitelmäoppiminen** | Kattotermi "kopioi mielenosoittaja"; BC on yksinkertaisin jäsen |
| **Korjausvirhe** | Pieni BC-virhe johtaa tiloihin, joita tietojoukko ei ole koskaan nähnyt, missä virheet lisääntyvät |
| **Esittelytiedot** | Asiantuntijan tuottamat liikeradat, joita käytetään BC-harjoitussarjana |
| **Datakatto** | BC:n tuotto rajoittuu tietojoukon keskimääräiseen tuottoon |
| **DAgger** | Interaktiivinen korjaus yhdistämisvirheeseen |

---

## Yhden lauseen yhteenveto

> **Käyttäytymiskloonaus on vain valvottua oppimista (tila, toiminta)
> paria — vahva, kun data on hyvää, avuton, kun dataa on sekoitettu.**
