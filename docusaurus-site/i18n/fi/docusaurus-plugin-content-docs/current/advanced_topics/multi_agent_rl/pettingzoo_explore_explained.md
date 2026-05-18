# PettingZoo-ympäristöihin tutustuminen 🦓

## Mikä on PettingZoo?

Jos olet tehnyt yhden agentin RL:n, olet todennäköisesti käyttänyt **Gymnasiumia** (
OpenAI Gymin seuraaja). Jokainen ympäristö näyttää samalta: `env.reset()`,
`env.step(action) → obs, reward, done, info` — uusi *havainto* maailmasta,
skalaari *palkinto* -merkki, *valmis* -lippu, jossa lukee "peli ohi" ja *info*
sanakirja lisäosien virheenkorjaukseen. Tämä yhtenäisyys saa RL-kirjastot toimimaan.

**PettingZoo** on täsmälleen sama idea, mutta *useita agentteja*. Se on a
moniagenttiympäristöjen eläintarha – kaikki yhden tarkasti määritellyn API:n takana:
- **Klassiset leluongelmat**: yksinkertaiset ympäristöt, kuten Rock-Paper-Scissors, perusalgoritmien testaamiseen.
- **Yhteistyöverkkomaailmat**: agentit, jotka navigoivat ruudukossa yhteisen tavoitteen saavuttamiseksi.
- **Atari-moninpeli**: klassisia kilpailupelejä, kuten Pong.
- **MPE (Multi-Particle Environment)**: jatkuvan avaruuden fysiikan ympäristöt monimutkaiseen koordinointiin ja kilpailuun.

Jos osaat kirjoittaa sen koodin
toimii yhdessä PettingZoo-ympäristössä, voit liittää mihin tahansa muuhun
lähes ilman muutoksia.

---

## Kaksi API-tyyliä

Monen agentin asetukset ovat sotkuisempia kuin yhden agentin asetukset, koska kaksi agenttia
voi toimia samanaikaisesti tai vuorotellen tai jopa mielivaltaisissa järjestyksessä. PettingZoo
ratkaisee tämän kahdella rinnakkaisella API:lla:

### 1) AEC (Agent-Environment-Cycle)

Yksi agentti toimii kerrallaan. Ympäristö kiertää joissakin tekijöissä
tilaa, ja jokainen saa:
- **havainto** - mitä he näkevät *juuri*,
- **palkkio** — *yhteisestä* toiminnasta viimeksi ansaittu voitto
  kierros (eli mitä tapahtui *kaikkien* agenttien toiminnan seurauksena, ei vain
  sinä; esimerkiksi shakkipelissä palkintosi heijastaa pelilaudan tilaa
  vastustajasi viimeisen liikkeen jälkeen, ei vain sinun)
- **lopetuslippu** - `True` kun jakso päättyy *luonnollisesti* (esim.
  matti, joku voittaa),
- **lyhennyslippu** - `True` kun jakso on *lyhennetty*
  raja ennen kuin luonnollinen loppu saavutetaan.

Tämä on luonnollista **vuoropohjaisissa peleissä**, kuten shakissa, gossa ja pokerissa.

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

### 2) Rinnakkais

Kaikki edustajat tarkkailevat ja toimivat samanaikaisesti jokaisessa vaiheessa. `step()` ottaa a
*sanakirja* toimista ja palauttaa havaintojen sanakirjat ja
palkintoja.

Tämä on luonnollista **reaaliaikaisissa peleissä**, kuten MPE (Multi-Particle
Ympäristöt, joissa kaikki pisteagentit liikkuvat samanaikaisesti) tai multi-agent
verkkomaailmat.

```python
obs, info = env.reset()
while env.agents:
    actions = {a: my_policy(obs[a]) for a in env.agents}
    obs, rewards, terms, truncs, info = env.step(actions)
```

Nämä kaksi tyyliä ovat **isomorfisia** — rakenteellisesti vastaavia ja
muunnettavissa: mihin tahansa AEC-ympäristöön voidaan automaattisesti kääriä
näyttää rinnakkaiselta ja päinvastoin. PettingZoo toimittaa muunnoksen
kääreitä, joten sinun tarvitsee kirjoittaa koodi vain yhdelle tyylille.

---

## Tosielämän analogia

- **AEC = lautapeliilta.** "Alicen vuoro. Nyt Bob. Nyt Carol. Takaisin
  Alice." Se, joka siirtyy seuraavaksi, näkee uusimman taulun tilan.
- **Rinnakkais = moninpeli.** Kaikki neljä pelaajaa painavat
  painikkeet samanaikaisesti; Peli päivittää maailmaa 60 kertaa sekunnissa.
- **Miksi yhtenäisillä API:lla on merkitystä.** Kuvittele, jos jokainen moninpeli
  tarvitsi oman ohjaussauvan. PettingZoo on MARLin "yleinen joystick".

---

## Mitä koodimme tekee

Rakennamme **PettingZoo-tyylisen** ympäristön tyhjästä: **Iterated
Koordinointipeli**. Kaksi agenttia valitsee toistuvasti kanavaa `0` tai `1`:

- Sama valinta → molemmat saavat +1
- Eri valinta → molemmat saavat -1

Jokaisen edustajan saama **havainto** on edellinen *yhteistoimi* —
mitä molemmat agentit valitsivat viime kierroksella, pakattuna yhdeksi kokonaisluvuksi.
Konkreettisesti: kunkin agentin viimeinen toiminta on yksi `{start, 0, 1}` (3 tilaa),
joten pari koodataan muodossa `3 × agent_1_state + agent_2_state`, jolloin saadaan
9 mahdollista kokonaislukua (0-8). Kokonaisluku 0 on "aloitus"-tila - se signaloi
että toimenpiteisiin ei ole vielä ryhdytty (jakson alussa).
Jakso kestää 25 vaihetta, joten maksimi kokonaistuotto on +25 agenttia kohti
ja minimi on −25. **Satunnaisen pelin tulokset ≈ 0**, koska jokaisessa vaiheessa
kaksi riippumatonta satunnaista agenttia valitsee kukin 0 tai 1 samalla todennäköisyydellä:
ne vastaavat 50 % ajasta (+1) ja eroavat 50 % ajasta (−1), antaen
odotettu askelkohtainen palkkio 0,5 × (+1) + 0,5 × (−1) = **0**. Summennettu
25 askeleen jälkeen odotettu jakson tuotto on myös 0.

Me sitten:

1. **Esittele AEC-liitäntä** satunnaisella käyttöönotolla – tämä vahvistaa
   AEC-perussilmukka: `agent_iter()` antaa agentin, jonka vuoro on,
   `last()` lukee agentin tämänhetkisen havainnon ja kertyneen palkkion,
   ja `step()` toimittaa valitsemansa toimintansa takaisin ympäristöön.
2. **Kouluta kaksi itsenäistä Q-oppijaa rinnakkaisliittymän kautta**.
   Jokainen agentti pitää oman Q-taulukkonsa, joka on **yhteistoiminnolla avattu
   havainto** (yksi kokonaisluku, joka koodaa sen, mitä *molemmat* agentit tekivät
   viime kierroksella), jotta se voi oppia "kun molemmat valitsimme 0 viime kerralla, minun pitäisi
   valitse uudelleen 0."
3. **Yritä tuoda oikea `pettingzoo` kirjasto** ja julkaise yksi niistä
   sisäänrakennetut ympäristöt (Rock-Paper-Scissors) satunnaisella käytännöllä. Jos
   PettingZooa ei ole asennettu, ohitamme tämän vaiheen ystävällisellä viestillä.

### Mitä sinun pitäisi nähdä

| Vaihe | Odotettu |
|-------|----------|
| Satunnainen käyttöönotto (AEC)            | Keskimääräinen (keskimääräinen) jakson tuotto lähellä **0** – satunnaiset agentit valitsevat kanavat itsenäisesti, täsmäävät ja eivät täsmää suunnilleen yhtä paljon. |
| Riippumattomat Q-oppijat (rinnakkaiset) – ensimmäiset 100 jaksoa | Noin **0** — edelleen enimmäkseen satunnainen agenttien tutkiessa. |
| Riippumattomat Q-oppijat – viimeiset 100 jaksoa             | Vahvasti positiivinen, **+20 - +25** — **koordinaatio on syntynyt**: molemmat agentit ovat oppineet valitsemaan luotettavasti saman kanavan joka kierroksella. |

Juoni `outputs/pettingzoo_coordination.png` näyttää yksittäisen jakson
palautukset (harmaa) ja liikkuva **Keskiarvo** käyrä (sininen). Keskiarvo tasoittaa meluisaa
jaksot, jotta voit nähdä trendin: agentit siirtyvät koordinoimattomasta satunnaisesta
pelata lähellä ~0:ta kohti vakaata **koordinaatiota** lähellä ~+25. Vihreä katkoviiva
merkitsee täydellisen koordinaation kattoa.

Jos `pettingzoo` on asennettu, myös komentosarja julkaistaan
`pettingzoo.classic.rps_v2` todistaa, että skripti toimii todellista vastaan
kirjasto täsmälleen samalla tavalla kuin se toimii käsin rullatussa kotelossamme. Ota käyttöön
tuo jakso:

```bash
source ../../venv/bin/activate
pip install "pettingzoo[classic]"
```

---

## Miksi rakentaa mukautettu kirjekuori ensin?

Koska **API on oppitunti.** (Useiden agenttien ja ympäristön välisen vuorovaikutuksen jäsentämisen ymmärtäminen on tärkeämpää kuin tietyt pelisäännöt.) Multi-agent RL:llä on monia makuja
(vuoropohjainen, reaaliaikainen, yhteistyökykyinen, kilpailukykyinen, sekoitettu) ja ne kaikki
sovi AEC / rinnakkaiskuvioon. Kun olet toteuttanut nämä kaksi
silmukoita, jokainen PettingZoo-ympäristö on vain liittäminen a
eri env-konstruktori - koulutuskoodi pysyy samana.

Juuri näin Gymnasium muutti yhden agentin RL:n: tekemällä
ympäristö musta laatikko yhtenäisen käyttöliittymän takana.

---

## Missä itsenäinen Q-oppiminen auttaa ja satuttaa

Koordinointipelit ovat *anteeksiantavia* – agentit jakavat palkintomerkin
heidän etunsa kohtaavat. Riippumattomat oppijat voivat ratkaista tämän onnellisesti, koska yhden edustajan tekemä parannus auttaa toista.

**Aversarial** -peleissä (RPS) riippumaton Q-oppiminen värähtelee ikuisesti (kun yksi agentti mukautuu, toinen muuttaa strategiaansa vastustaakseen, mikä johtaa loputtomaan jahtaamiseen).
**osittain havaittavissa** peleissä se ei voi oppia ollenkaan, koska
"havainnointi" on vain yksi osa tilaa (agenttia voidaan rangaista hyvästä toiminnasta vain siksi, että se ei voinut nähdä, mitä toinen agentti teki). PettingZoo sisältää molemmat
erilaisissa ympäristöissä, jotta voit nähdä nämä vikatilat itse.

---

## Avainsanat muistaa

| sana | Merkitys |
|------|---------|
| **eläineläintarha**     | The Gymnasium of multi-agent RL — kirjasto standardoituja MARL-ympäristöjä |
| **AEC**            | Agent-Environment-Cycle: yksi agentti toimii askelta kohti (vuoropohjainen) |
| **Parallel API**   | Kaikki agentit toimivat samanaikaisesti jokaisessa vaiheessa |
| **MPE**            | Multi-Particle Environment, suosittu yhteistyö/kilpaileva testipöytä, joka toimitetaan PettingZoon mukana (usein mukana liikkuvia pisteitä navigoimassa fysiikkapohjaisissa tehtävissä). |
| **CTDE**           | Keskitetty koulutus, hajautettu toteutus – harjoittele globaalilla näkymällä (pääsy kaikkiin tiloihin), ota käyttöön vain paikalliset obssit (jokainen agentti toimii oman rajoitetun näkemyksensä mukaan). |
| **Itsenäinen Q-oppiminen** | Jokainen agentti suorittaa vanilja Q-oppimista (standardi, muokkaamaton Q-oppimisalgoritmi) jättäen huomioimatta muiden oppijoiden olemassaolon. |

---

## Yhden lauseen yhteenveto

> **PettingZoo antaa jokaiselle moniagenttiympäristölle saman muodon – joten
> tänään kirjoittamasi koodi toimii edelleen huomenna täysin erilaisessa pelissä.**

Kun nämä kaksi API-tyyliä ovat toissijaisia, voit siirtyä käyttämään MADDPG:tä
(keskitetty kriitikko jatkuvatoimisille aineille), QMIX (arvosekoitus
yhteistyöryhmät), MAPPO (multi-agent PPO) tai mikä tahansa muu moderni MARL
algoritmi – koodisi ympäristöpuolen ei tarvitse koskaan muuttua.
