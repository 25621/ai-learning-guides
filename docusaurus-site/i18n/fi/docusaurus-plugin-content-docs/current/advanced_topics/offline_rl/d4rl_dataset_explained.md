# D4RL-vertailutietojoukot 📦

## Mikä se on?

Kuvittele, että haluat opettaa robotin kääntämään pannukakkuja. Sen antaminen harjoitella todellisella liedellä kuukauden ajan olisi hidasta, vaarallista ja kallista. Mutta sinulla on kymmenen vuoden ajalta tallennettua videota siitä, kun kokit kääntävät pannukakkuja (osa hyvin, osa huonosti, jotkut satunnaisesti). Voitko opettaa robotin *pelkästään tämän datan avulla*, ilman että sen annetaan koskaan koskea oikeaan paistinpannuun?

Se on **offline-vahvistusoppimista**. Agentti oppii kiinteästä, aiemman kokemuksen aineistosta ilman suoraa vuorovaikutusta ympäristön kanssa. Vaikein osa on se, ettei agentti voi koskaan *kokeilla* oppimaansa ennen kuin vasta aivan lopussa.

Jotta tämä opiskelu olisi oikeudenmukaista, tutkimusyhteisö tarvitsi *standardin
tietojoukko*. Se on **D4RL** (**D**atasets for **D**eep **D**ata-**D**riven
**R**einforcement **L**earning): kokoelma ennalta tallennettuja siirtymiä
klassisissa ohjaustehtävissä, julkaistu UC Berkeleyn toimesta vuonna 2020. Jokainen tutkimus kouluttaa mallinsa
samoilla tiedoilla, joten tulokset ovat suoraan vertailukelpoisia.

---

## Mitä D4RL-tietojoukossa on?

D4RL toimittaa jokaiselle tehtävälle **neljä laatutasoa**:

| Taso | Mistä tiedot tulevat | Miksi sillä on väliä |
|-------|---------------------------|----------------|
| **satunnainen**        | Käytäntö, joka valitsee toiminnot tasaisen satunnaisesti | Pahimmassa tapauksessa: voitko silti oppia jotain hyödyllistä? |
| **keskikokoinen**        | Osittain koulutettu käytäntö (noin puolet asiantuntijapisteistä) | Realistinen: useimmat kirjatut tiedot ovat keskinkertaisia |
| **asiantuntija**        | Lähes asiantuntijatasoinen käytäntö | Paras tapaus: pystytkö saavuttamaan saman tason kuin datan tuottanut käytäntö? |
| **keskitason toisto** | Koko se toistopuskuri (replay buffer), jota käytettiin keskitasoisen käytännön kouluttamiseen | Mixed: sisältää varhaisia epäonnistumisia JA myöhempiä onnistumisia |

Ero välillä `medium` ja `medium-replay` on ratkaisevaa:
- **`medium`** luodaan ottamalla yksi kiinteä "keskimääräinen" käytäntö ja antamalla sen pelata monia pelejä. Kaikki tiedot heijastavat tätä tasaista, keskimääräistä taitotasoa.
- **`medium-replay`** on historiallinen loki. Se sisältää kaikki kokemukset, jotka on kerätty *oppiessaan* tyhjästä keskitasolle asti. Se sekoittaa **pahaa ja hyvää**
siirtymät – tarkalleen miltä todellinen loki näyttää (robotin ensimmäinen
kömpelöt yritykset *ja* sen myöhempi jalostettu käytös, kaikki samassa ämpärissä).

---

## Tosielämän esimerkkejä offline-tietojoukoista

- **Lääketieteelliset tiedot.** Vuodet (potilaan_tila, hoito, lopputulos) lukuja.
  Et voi satunnaistaa hoitoja eläville ihmisille, mutta voit oppia a
  parempi käytäntö lokitiedoista.
- **Asiakaspalvelun chat-lokit.** Miljoonat (user_message, agent_reply,
  tyytyväisyys) kirjaa. Kouluta parempi avustaja vaivaamatta enempää
  käyttäjiä.
- **Autonomisen ajon kalustotiedot.** Jokainen Tesla/Waymo-auto lataa tietonsa
  ajosta. Koko laivasto muodostaa jättimäisen "keskitason toisto" -tyyppisen aineiston.
- **Suositusjärjestelmät.** Viime vuoden napsautuslokit ovat staattinen aineisto:
  et voi näyttää samoja mainoksia uudelleen samoille käyttäjille.

Kaikissa neljässä tapauksessa **et voi pyytää ympäristöltä uutta näytettä.**
Tietojoukko on kaikki mitä sinulla on. Ikuisesti.

---

## Mitä koodimme tekee

Todelliset D4RL-tietojoukot tallennetaan MuJoCo:lle (Multi-Joint dynamics with
Yhteystiedot) liikkumistehtävät
(kuten HalfCheetah, Hopper, Walker2d, Ant – nämä ovat edistyneitä 3D-fysiikkasimulaatioita, joissa virtuaaliset robotit oppivat kävelemään ja juoksemaan). MuJoCo on raskas asentaa, joten me
luo uudelleen **sama nelitasoinen rakenne CartPole-v1:ssä** — standardi
aloittelijaympäristö aikaisemmista vaiheista. Oppitunnit siirtyvät suoraan.

Skripti `d4rl_dataset.py`:

1. **Harjoittaa DQN:ää** (Deep Q-Network, tavallinen RL-algoritmi) CartPolessa, kunnes se ratkaisee tehtävän (palautus ≥ 475).
2. **Ottaa tilannekuvia kahdesta tarkistuspisteestä** matkan varrella:
   - "keskikokoinen" - hetki, jolloin tuotto ylitti 150
   - "asiantuntija" - hetki, jolloin äskettäinen tuotto ylitti 475:n
3. **Ottaa tilannekuvia keskitasoisen käytännön koko toistopuskurista** – jokainen siirtymä
   se on koskaan nähnyt. Se on "keskitason toisto" -tietojoukkomme.
4. **Ottaa käyttöön kolme uutta käytäntöä** 10 000 siirtymän ajaksi:
   - `random`   — tasaisen satunnainen
   - `medium`   — keskitason tarkistuspiste + ε=0,10 kohina
   - `expert`   — asiantuntijatarkastuspiste + ε=0,02 kohinaa
5. **Tallentaa neljä `.npz`-tiedostoa** (NumPyn pakattu taulukkomuoto).
   `outputs/`, jokaisessa on taulukoita `obs / action / reward / next_obs / terminal`.

Nämä neljä tiedostoa toimivat syötteinä skripteille `cql.py` ja `behavioral_cloning.py`.

---

## Mitä sinun pitäisi nähdä, kun suoritat sen

Pelkkätekstinen yhteenveto tulostettu konsoliin ja tallennettu
`outputs/d4rl_summary.txt`:

```
dataset         |   N    |  mean return  |  min  |  max
------------------------------------------------------------
random          | 10000  |          ~22  |    ~9 |   ~80
medium          | 10000  |         ~180  |   ~50 |  ~500
expert          | 10000  |         ~490  |  ~400 |   500
medium-replay   | 10000  |          ~60  |    ~9 |  ~200
```

Se myös luo histogrammin (`outputs/d4rl_returns.png`), joka näyttää kuinka
neljä tietojoukkoa menevät päällekkäin. Tärkeimmät huomioitavat ominaisuudet:

- **Satunnaiset** klusterit noin 20 (satunnaisen CartPole-jakson keskimääräinen pituus).
- **Expert**-klusterit 500-katossa.
- **Keskitaso** on välissä, suurella varianssilla.
- **Keskitasoisella toistolla** on pitkä oikea takaosa – se koostuu enimmäkseen varhaisista epäonnistuneista suorituksista (pienet tuotot), mutta sen häntä ulottuu korkeampiin tuotoihin agentin oppien mukaan.

---

## Miksi tietojoukolla on merkitystä

Riippumatta siitä, millä tietojoukolla harjoitat offline-algoritmiasi, asetat
*katon* sille, mikä on mahdollista:

- **Asiantuntijadatan (expert) avulla** — jopa yksinkertainen algoritmi, kuten BC (Behavioral Cloning, joka vain kopioi tiedot tarkasti), voi toimia hyvin,
  koska kaikki tiedot ovat hyviä.
- **Satunnaisdatan (random) avulla** — tarvitset älykkään algoritmin, joka *ompelee yhteen*
  harvinaisia hyviä siirtymiä (polun löytäminen menestykseen yhdistämällä lyhyitä hyviä toimintoja eri yrityksistä). BC epäonnistuu täysin.
- **Keskitason toistodatan (medium-replay) avulla** — realistisin ja mielenkiintoisin.
  Hyvät algoritmit (kuten **CQL** – konservatiivinen Q-Learning, joka välttää
  liian itsevarma olemasta toimista, joita se ei ole koskaan nähnyt) voi joskus **päihittää
  keskimääräisen tietojen laadun**, koska ne poimivat rakenteen sekakuoresta
  signaaleja. Yksinkertaiset algoritmit (BC) taantuvat keskiarvoon.

Näemme täsmälleen tämän tarinan kahdessa seuraavassa skriptissä.

---

## Avainsanat muistaa

| sana | Merkitys |
|------|---------|
| **Offline RL**         | Koulutus kiinteästä tietojoukosta; ympäristövuorovaikutusta ei sallita |
| **Käyttäytymiskäytäntö**   | Käytäntö, joka *tuottaa* tietojoukon |
| **Tietojoukon laatu**    | Kuinka hyvä käyttäytymiskäytäntö oli (satunnainen / keskitaso / asiantuntija) |
| **Toista puskuri**      | Harjoittelun aikana nähty muutosten koko historia |
| **Jakelumuutos** | Tietojoukon toimien ja koulutetun käytäntösi haluamien toimien välinen kuilu. Koska tietojoukko ei koskaan näytä, mitä tapahtuu, kun uusi käytäntö yrittää jotain, jota ei tallennettu, algoritmin arvoarviot näille uusille toimille voivat olla vaarallisen väärin. |

---

## Yhden lauseen yhteenveto

> **D4RL jäädyttää RL:n valvotun oppimisen tyyliin: samat tavut
> kaikille, ei ympäristön huijausta, paras algoritmi voittaa.**
